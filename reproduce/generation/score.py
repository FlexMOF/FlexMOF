"""MOFid VNU and MOFFlow-2 validity with complete denominators."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback
import types

from worker import atomic

HERE = Path(__file__).resolve().parent
MOFID = Path(os.environ['FLEXMOF_MOFID_ROOT'])
JAVA = Path(os.environ['FLEXMOF_JAVA_BIN'])
DATA = Path(os.environ['FLEXMOF_BW_DB'])
MOFID_COMMIT = '36873683083c7bc62c1da2062df2833adc35b48a'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def verify_mofid(out):
    """Pin the source version and keep resumed scoring on the same build."""
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(MOFID), 'rev-parse', 'HEAD'], text=True).strip()
        subprocess.run(['git', '-C', str(MOFID), 'diff', '--quiet', 'HEAD', '--'],
                       check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('MOFid must be an unmodified Git checkout; see README') from exc
    if commit != MOFID_COMMIT:
        raise RuntimeError(f'Expected MOFid commit {MOFID_COMMIT}, found {commit}')
    binary = MOFID / 'bin/sbu'
    if not binary.is_file():
        raise RuntimeError('MOFid bin/sbu is missing; build MOFid as described in README')
    protocol = dict(mofid_commit=commit, mofid_sbu_sha256=sha(binary))
    path = out / 'scoring_protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError('MOFid build changed since scoring started; use a fresh run')
    else:
        if any((out / 'scores').glob('*.json')) or (out / 'training_reference.sqlite').exists():
            raise RuntimeError('Existing scores lack MOFid build metadata; use a fresh run')
        atomic(path, protocol)
    return protocol


def setup():
    os.environ['PATH'] = str(JAVA)+os.pathsep+os.environ['PATH']
    os.environ['JAVA_TOOL_OPTIONS'] = '-XX:ActiveProcessorCount=2'
    os.environ['BABEL_LIBDIR'] = str(MOFID/'openbabel/build/lib')
    os.environ['BABEL_DATADIR'] = str(MOFID/'openbabel/data')
    spec = importlib.util.spec_from_file_location('mofid', MOFID/'Python/__init__.py',
        submodule_search_locations=[str(MOFID/'Python')])
    module = importlib.util.module_from_spec(spec)
    sys.modules['mofid'] = module
    spec.loader.exec_module(module)
    paths = types.ModuleType('mofid.paths')
    paths.mofid_path = str(MOFID)
    paths.bin_path = str(MOFID/'bin')
    paths.openbabel_path = str(MOFID/'openbabel')
    paths.resources_path = str(MOFID/'Resources')
    sys.modules['mofid.paths'] = paths
    from mofid import id_constructor as ic
    original = ic.runcmd
    ic.runcmd = lambda cmd, timeout=None: original(cmd, timeout=90 if timeout is None else timeout)


def history():
    spec = importlib.util.spec_from_file_location('reference_mofid', HERE/'reference_mofid.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MOFID, mod.JAVA, mod.DATA, mod.setup_mofid = MOFID, JAVA, DATA, setup
    return mod


def sample_init():
    setup()
    import evaluate_mofflow2 as evaluation
    original = evaluation._mofchecker_descriptor_result
    def retain_descriptors(descriptors):
        result = original(descriptors)
        result.update({'mofchecker_'+k: None if descriptors[k] is None else int(bool(descriptors[k]))
            for k in evaluation.MOFFLOW2_EXPECTED_CHECK_VALUES})
        return result
    evaluation._mofchecker_descriptor_result = retain_descriptors


def one(task):
    out, i, record = task
    out=Path(out)
    row=dict(sample_id=i, cif=record.get('cif',''), exists=False, valid=False)
    started=time.time()
    from evaluate_mofflow2 import _evaluate_mofchecker, MOFFLOW2_EXPECTED_CHECK_VALUES
    result = _evaluate_mofchecker((str(i), row['cif'], 120))
    row.update(result)
    row['valid'] = bool(result['mofchecker_valid'])
    if row['cif'] and Path(row['cif']).is_file():
        row['cif_sha256']=sha(Path(row['cif']))
        try:
            from mofid.run_mofid import cif2mofid
            h=history()
            identifiers=cif2mofid(row['cif'], str(out/'mofid'/f'{i:05d}'))
            row.update(identifiers)
            row['exists']=h.existence(identifiers['smiles'],identifiers['cat'])
            row['known_topology']=h.known_topology(identifiers['topology'])
            row['canonical']=h.identity(identifiers['smiles'],identifiers['topology'],identifiers['cat']) if row['exists'] else None
        except Exception as e:
            row['mofid_error']=f'{type(e).__name__}: {e}'
    row.update(sample_id=i, seconds=time.time()-started)
    atomic(out/'scores'/f'{i:05d}.json',row)
    return row


REF=None
def ref_init(targets):
    global REF
    REF=history()
    REF.train_init(targets)


def ref_one(key):
    return REF.train_one(key)


def reference(out, rows, workers):
    cache=out/'training_reference.sqlite'
    db=sqlite3.connect(cache)
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY,status TEXT,canonical TEXT,result TEXT)')
    keys=(DATA/'train_split_filtered.txt').read_text().splitlines()
    assert sha(DATA/'train_split_filtered.txt') == 'b729ad872c56fdd4ec90251eeda7863755e6836bb86cbf5cf9e5899f19425b9d'
    assert len(keys)==len(set(keys)), 'Duplicate training keys'
    completed={k for k, in db.execute('SELECT key FROM results')}
    assert completed <= set(keys), 'Reference cache contains unexpected training keys'
    targets={r['smiles'] for r in rows if r['exists']}
    old=db.execute("SELECT value FROM metadata WHERE key='targets'").fetchone()
    if old:targets |= set(json.loads(old[0]))
    h=history()
    revisit=h.prepare_reference_targets(db,targets)
    # Retry unresolved execution failures from interrupted scoring, never silently
    # declare a structure novel against an incomplete reference.
    revisit |= {k for k, in db.execute("SELECT key FROM results WHERE status='error'")}
    pending=(set(keys)-completed) | revisit
    atomic(out/'reference_progress.json',dict(total=len(keys), pending=len(pending), status='running'))
    if pending:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=ref_init,
                initargs=(targets,)) as pool:
            remaining=iter(sorted(pending))
            outstanding=set()
            for _ in range(min(len(pending), max(1,workers*4))):
                outstanding.add(pool.submit(ref_one,next(remaining)))
            finished_count=0
            while outstanding:
                finished,outstanding=wait(outstanding,return_when=FIRST_COMPLETED)
                for future in finished:
                    r=future.result()
                    db.execute('INSERT OR REPLACE INTO results VALUES (?,?,?,?)',
                        (r['key'],r['status'],r.get('canonical'),json.dumps(r)))
                    finished_count+=1
                    try:
                        outstanding.add(pool.submit(ref_one,next(remaining)))
                    except StopIteration:
                        pass
                    if finished_count%50==0:db.commit()
                    if finished_count%1000==0:
                        atomic(out/'reference_progress.json',dict(total=len(keys),pending=len(pending)-finished_count,status='running'))
        db.commit()
    counts=dict(db.execute('SELECT status,COUNT(*) FROM results GROUP BY status'))
    assert not counts.get('error',0) and not counts.get('pending_target_recheck',0), counts
    assert {k for k, in db.execute('SELECT key FROM results')} == set(keys), 'Reference build is incomplete'
    ref={v for v, in db.execute('SELECT canonical FROM results WHERE canonical IS NOT NULL')}
    atomic(out/'reference_progress.json',dict(total=len(keys), pending=len(pending), status='complete', counts=counts))
    db.close()
    return ref, counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    out=a.root/('denovo_smoke' if a.smoke else 'denovo')
    expected=2 if a.smoke else 10000
    for n in ['scores','mofid']:(out/n).mkdir(exist_ok=True)
    mofid_protocol=verify_mofid(out)
    records=[json.loads((out/'records'/f'{i:05d}.json').read_text()) for i in range(expected)]
    assert all(r['status'] in ['generated','generation_error'] for r in records)
    tasks=[]
    for i,r in enumerate(records):
        if not (out/'scores'/f'{i:05d}.json').exists():tasks.append((str(out),i,r))
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=sample_init) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result()
            if i%100==0 or i==len(tasks):print(json.dumps(dict(scored=i,total=len(tasks),last=r['sample_id'])),flush=True)
    rows=[json.loads((out/'scores'/f'{i:05d}.json').read_text()) for i in range(expected)]
    if a.smoke:
        assert any(r['exists'] for r in rows), 'No accepted MOFid in scoring smoke'
        assert not all(r.get('mofid_error') for r in rows),rows
        # A generated CIF can fail a geometry or symmetry check; the full
        # protocol records that sample as invalid. Require at least one
        # successful checker evaluation to verify the installation.
        assert any(r['mofchecker_status'] in ('valid', 'invalid') for r in rows),rows
        atomic(out/'scoring_smoke.json',rows)
        return
    ref,ref_counts=reference(out,rows,a.workers)
    frequency=Counter(r['canonical'] for r in rows if r['exists'])
    for r in rows:
        r['unique']=r['exists'] and frequency[r['canonical']]==1
        r['novel']=r['exists'] and r['canonical'] not in ref
        atomic(out/'scores'/f"{r['sample_id']:05d}.json",r)
    accepted=sum(r['exists'] for r in rows)
    counts={k:sum(bool(r[k]) for r in rows) for k in ['valid','exists','unique','novel']}
    counts['valid_novel_unique']=sum(r['valid'] and r['novel'] and r['unique'] for r in rows)
    counts['distinct_mofids']=len(frequency)
    result=dict(complete=True,requested=expected,counts=counts,**mofid_protocol,
        percent_of_all={k:100*v/expected for k,v in counts.items()},
        conditional_percent_among_accepted_mofids={k:100*counts[k]/accepted if accepted else None for k in ['unique','novel']},
        distinct_identifier_uniqueness_percent=100*len(frequency)/accepted if accepted else None,
        generation_status=dict(Counter(r['status'] for r in records)),
        checker_status=dict(Counter(r['mofchecker_status'] for r in rows)),
        mofid_errors=sum(bool(r.get('mofid_error')) for r in rows),
        reference_count=247062,reference_status=ref_counts,
        validity='MOFFlow-2 exact 16 checks (MOFChecker 0.9.6), checker errors invalid',
        uniqueness='Singleton convention; distinct-ID convention also reported',
        novelty='Exact canonical MOFid against all training keys; wildcard identifiers not accepted')
    atomic(out/'results.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
