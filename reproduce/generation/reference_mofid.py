"""MOFid training-reference and identity helpers."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

TARGETS = set()
LMDB_ENV = None
DATA = Path('.')
setup_mofid = None

def identity(smiles, topology, cat):
    return f'{smiles} MOFid-v1.{topology}.cat{cat}'


def existence(smiles, cat):
    # Fragment/catenation success; topology labels are retained verbatim.
    # Known-topology-only results are reported separately as a sensitivity check.
    return bool(smiles and '*' not in smiles and cat is not None and str(cat).isdigit())


def known_topology(topology):
    bad = {'NA','ERROR','UNKNOWN','TIMEOUT','MISMATCH',''}
    return not any(t in bad for t in topology.split(','))


def train_init(targets):
    global TARGETS, LMDB_ENV
    setup_mofid()
    import lmdb
    TARGETS = targets
    LMDB_ENV = lmdb.open(str(DATA/'MetalOxo.lmdb'),subdir=False,readonly=True,lock=False,readahead=False)


def train_one(key):
    import pickle
    import warnings
    import numpy as np
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.cif import CifWriter
    from mofid.id_constructor import extract_fragments,extract_topology
    started = time.time()
    row = {'key':key,'status':'error','canonical':None,'smiles_sha256':None,'cat':None}
    try:
        with LMDB_ENV.begin() as txn: data = pickle.loads(txn.get(key.encode()))
        row['m_id'] = str(data.m_id)
        structure = Structure(Lattice(np.asarray(data.cell,dtype=float)),np.asarray(data.atom_types).tolist(),np.asarray(data.frac_coords,dtype=float))
        with tempfile.TemporaryDirectory(prefix='flexmof_vnu_',dir='/tmp') as temp:
            cif = Path(temp)/'training.cif'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                CifWriter(structure,significant_figures=10).write_file(cif)
            with open(os.devnull,'w') as null, contextlib.redirect_stderr(null):
                nodes,linkers,cat,_ = extract_fragments(str(cif),str(Path(temp)/'mofid'))
                smiles = '.'.join(sorted(nodes+linkers))
                row['smiles_sha256'] = hashlib.sha256(smiles.encode()).hexdigest()
                row['cat'] = cat
                if not existence(smiles,cat):
                    row['status'] = 'no_identifier'
                elif smiles not in TARGETS:
                    row['status'] = 'fragment_nonmatch'
                else:
                    sn = extract_topology(str(Path(temp)/'mofid/SingleNode/topology.cgd'))
                    an = extract_topology(str(Path(temp)/'mofid/AllNode/topology.cgd'))
                    topology = sn if sn==an or an=='ERROR' else sn+','+an
                    row.update(status='full_identifier',canonical=identity(smiles,topology,cat),topology=topology)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    row['seconds'] = time.time()-started
    return row


def prepare_reference_targets(db,targets):
    """Persist pending rechecks atomically with expansion of the target set."""
    fingerprint = hashlib.sha256(json.dumps(sorted(targets)).encode()).hexdigest()
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY,status TEXT,canonical TEXT,result TEXT)')
    old = db.execute("SELECT value FROM metadata WHERE key='targets'").fetchone()
    old_targets = set(json.loads(old[0])) if old else set()
    assert old_targets <= targets, 'MOFid target set unexpectedly shrank'
    added_hashes = {hashlib.sha256(s.encode()).hexdigest() for s in targets-old_targets}
    # Expansion of the query set only needs to revisit formerly nonmatching
    # fragments that now match a target hash. Identical strings necessarily
    # have identical hashes; any collision merely causes extra exact work.
    revisit = {key for key,result in db.execute("SELECT key,result FROM results WHERE status='fragment_nonmatch'")
               if json.loads(result)['smiles_sha256'] in added_hashes}
    db.executemany("UPDATE results SET status='pending_target_recheck' WHERE key=?",((k,) for k in revisit))
    db.execute("INSERT OR REPLACE INTO metadata VALUES ('targets',?)",(json.dumps(sorted(targets)),))
    db.execute("INSERT OR REPLACE INTO metadata VALUES ('target_sha256',?)",(fingerprint,))
    db.commit()
    return {key for (key,) in db.execute("SELECT key FROM results WHERE status='pending_target_recheck'")}
