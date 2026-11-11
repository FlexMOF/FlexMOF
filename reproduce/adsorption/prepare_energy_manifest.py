"""Build reference host groups for ODAC adsorption-energy evaluation.

The dataset's recorded DFT bare-MOF energy is used only as an identity key,
never subtracted from UMA energies. Bare geometries are constructed by removing
the predicted ADSORBATE blocks from the input reference structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'atommof_compat'))

import lmdb
import numpy as np
import torch


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args();root=args.root;data=args.data
    out=root/'energy/inputs';out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        raise FileExistsError('energy input manifest already exists; verify before reuse')
    with (data/'odac25_test_metadata.pkl').open('rb') as f:metadata=pickle.load(f)
    with (data/'test_system_indices.pkl').open('rb') as f:systems=pickle.load(f)
    keys=list(metadata)
    if len(keys)!=3582 or len(systems)!=683:
        raise ValueError('unexpected test or system count')
    reverse={};system_rows=[]
    for j,(system,indices) in enumerate(systems.items()):
        ids=[int(i) for i in indices]
        for i in ids:
            if i in reverse:raise ValueError('test index appears in multiple systems')
            reverse[i]=j
        system_rows.append({'system_index':j,'system_id':str(system),'indices':ids})
    if sorted(reverse)!=list(range(3582)):
        raise ValueError('system index does not cover the full test set')
    db=lmdb.open(str(data/'odac25_test.lmdb'),subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
    txn=db.begin();host_ids={};hosts=[];records=[];system_guests={}
    for index,key in enumerate(keys):
        record=pickle.loads(txn.get(key));numbers=np.asarray(record.atomic_numbers,dtype=np.int64)
        frac=np.asarray(record.frac_coords,dtype=np.float64);cell=np.asarray(record.cell,dtype=np.float64).reshape(3,3)
        if frac.shape!=(len(numbers),3) or not np.isfinite(frac).all() or not np.isfinite(cell).all():
            raise ValueError(f'invalid reference geometry at {index}')
        mask=np.ones(len(numbers),dtype=bool);offset=0;guest_blocks=0
        for block in record.blocks:
            count=len(block.atomic_numbers);role=getattr(block.block_type,'name',str(block.block_type))
            if role=='ADSORBATE':mask[offset:offset+count]=False;guest_blocks+=1
            offset+=count
        if offset!=len(numbers) or not guest_blocks or mask.all() or not mask.any():
            raise ValueError(f'host/guest partition invalid at {index}')
        info=record.info
        guest=tuple(int(info.get(k,0)) for k in ('nco2','nh2o','nn2','no2'))
        system=reverse[index]
        if system in system_guests and system_guests[system]!=guest:
            raise ValueError(f'guest counts vary within system {system}')
        system_guests[system]=guest
        host_numbers=numbers[mask];formula=sorted(Counter(map(int,host_numbers)).items())
        mof_energy=float(info['energy_mof'])
        if not math.isfinite(mof_energy):raise ValueError(f'nonfinite host identity at {index}')
        host_key={'mof_name':str(info['mof_name']),'supercell':[int(x) for x in info['supercell']],
                  'dataset_bare_energy_identity_eV':mof_energy,'host_formula':formula}
        canonical=json.dumps(host_key,sort_keys=True,separators=(',',':'))
        if canonical not in host_ids:
            host_id=len(hosts);host_ids[canonical]=host_id
            geometry={'host_id':host_id,'source_test_index':index,'atomic_numbers':torch.from_numpy(host_numbers.copy()),
                      'positions':torch.from_numpy((frac[mask]@cell).copy()),'cell':torch.from_numpy(cell.copy())}
            host_path=out/f'host_{host_id:04d}.pt'
            if host_path.exists():
                prior=torch.load(host_path,map_location='cpu',weights_only=True)
                if (prior['host_id']!=host_id or prior['source_test_index']!=index or
                    any(not torch.equal(prior[k],geometry[k]) for k in ('atomic_numbers','positions','cell'))):
                    raise ValueError(f'existing host input mismatch at {host_id}')
            else:
                temp=host_path.with_name(host_path.name+f'.tmp.{os.getpid()}')
                with temp.open('xb') as f:torch.save(geometry,f);f.flush();os.fsync(f.fileno())
                os.link(temp,host_path);os.unlink(temp)
            hosts.append({'host_id':host_id,'key':host_key,'source_test_index':index,
                          'geometry_path':str(host_path),'geometry_sha256':sha256(host_path),'records':[]})
        host_id=host_ids[canonical];hosts[host_id]['records'].append(index)
        records.append({'dataset_index':index,'system_index':system,'host_id':host_id,
                        'guest_counts':list(guest),'atoms':len(numbers),'host_atoms':len(host_numbers)})
        if (index+1)%500==0:print(json.dumps({'processed':index+1,'total':3582,'host_groups':len(hosts)}),flush=True)
    result={'status':'complete','records':records,'systems':system_rows,'hosts':hosts,
            'dataset_sha256':sha256(data/'odac25_test.lmdb'),
            'metadata_sha256':sha256(data/'odac25_test_metadata.pkl'),
            'system_indices_sha256':sha256(data/'test_system_indices.pkl'),
            'reference_policy':'Host group keyed by MOF name, supercell, formula and dataset bare-energy identity. '
              'Energies are recomputed with MLIPs; DFT metadata values are never mixed into MLIP differences.',
            'gas_cancellation':'All records in each system have identical per-species guest counts, so the same isolated-gas '
              'reference cancels exactly in the system minimum adsorption-energy difference.'}
    atomic_json(manifest_path,result)
    print('ENERGY_INPUTS_COMPLETE',json.dumps({'records':len(records),'systems':len(system_rows),'host_groups':len(hosts)}),flush=True)


if __name__=='__main__':main()
