"""Extract known intramolecular adsorbate bonds for a sensitivity analysis.

The mapping uses element and occurrence order, not reference geometry. This
auxiliary analysis does not change AtomMOF's literal all-pair overlap rule.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'atommof_compat'))

import lmdb
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    args=p.parse_args();root=args.root;data=args.data
    meta=json.loads((root/'energy/inputs/manifest.json').read_text())
    if sha(data/'odac25_test.lmdb')!=meta['dataset_sha256'] or sha(data/'odac25_test_metadata.pkl')!=meta['metadata_sha256']:
        raise ValueError('dataset identity mismatch')
    with (data/'odac25_test_metadata.pkl').open('rb') as f:keys=list(pickle.load(f))
    if len(keys)!=3582:raise ValueError('unexpected record count')
    db=lmdb.open(str(data/'odac25_test.lmdb'),subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
    txn=db.begin();rows=[];n2_count=0;n2_bonded=0
    for row in meta['records']:
        index=row['dataset_index'];record=pickle.loads(txn.get(keys[index]));numbers=np.asarray(record.atomic_numbers)
        offset=0;excluded=set();n2_edges=0
        for block in record.blocks:
            local=np.asarray(block.atomic_numbers);n=len(local);global_values=numbers[offset:offset+n]
            if len(global_values)!=n or sorted(map(int,local))!=sorted(map(int,global_values)):
                raise ValueError(f'block elemental mismatch at {index}')
            mapping=np.full(n,-1,dtype=int)
            for z in np.unique(local):
                source=np.flatnonzero(local==z);target=np.flatnonzero(global_values==z)
                mapping[source]=target+offset
            if not np.array_equal(numbers[mapping],local):raise ValueError('block mapping mismatch')
            role=getattr(block.block_type,'name',str(block.block_type))
            if role=='ADSORBATE':
                bonds=np.asarray(block.bond_indices)
                if bonds.size:
                    if bonds.ndim!=2 or bonds.shape[0]!=2:raise ValueError('invalid guest bonds')
                    for a,b in bonds.T:
                        a=int(a);b=int(b)
                        if not 0<=a<n or not 0<=b<n:raise ValueError('guest bond index out of bounds')
                        if a!=b:
                            edge=tuple(sorted((int(mapping[a]),int(mapping[b]))));excluded.add(edge)
                            if int(local[a])==int(local[b])==7:n2_edges+=1
            offset+=n
        if offset!=len(numbers):raise ValueError('block atom count mismatch')
        if row['guest_counts']==[0,0,1,0]:
            n2_count+=1;n2_bonded+=int(n2_edges>0)
        rows.append({'dataset_index':index,'exclude_adsorbate_bonds':[list(x) for x in sorted(excluded)]})
    if n2_count!=487 or n2_bonded!=487:raise ValueError('unexpected nitrogen bond annotation')
    result={'records':rows,'dataset_sha256':meta['dataset_sha256'],'metadata_sha256':meta['metadata_sha256'],
            'source':'Known direct bonds within ADSORBATE blocks only; same-element stable occurrence mapping.',
            'n2_reference_records':n2_count,'n2_reference_records_with_guest_bond':n2_bonded,
            'purpose':'Auxiliary sensitivity only; the primary overlap rule still considers all atom pairs.'}
    path=root/'energy/inputs/guest_bonds.json';temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('x') as f:json.dump(result,f,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()
    print('GUEST_BONDS_PREPARED',len(rows),'n2',n2_count,'bonded',n2_bonded,flush=True)


if __name__=='__main__':main()
