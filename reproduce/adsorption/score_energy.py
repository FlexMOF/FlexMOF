"""Aggregate persisted eSEN/UMA outcomes without dropping failed systems."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from energy_eval import CHECKPOINT_SHA, ENERGY_CODE_SHA, ESEN_SHA, PROFILE, UMA_SHA, VERSION

NUM_RECORDS=3582
NUM_SYSTEMS=683
NUM_SAMPLES=10
NUM_SURROGATE_HOSTS=71


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def score_one(indices, host_ids, references, candidates, hosts, k, radii):
    """Return conservative success status and minimum-energy difference.

    Each sample whose calculator failed is unknown, not an automatic failure.
    The common host and isolated-guest references cancel for one-host systems.
    For multiple hosts we use relaxed reconstructed-host energies, which is an
    explicitly approximate reference policy.
    """
    system_hosts = {host_ids[i] for i in indices}
    multiple_hosts = len(system_hosts) > 1
    result = {'host_policy':'reconstructed_relaxed_surrogate' if multiple_hosts else 'host_cancels_exactly'}
    if any(references[i]['status'] != 'complete' for i in indices):
        return {**result,'state':'unknown','reason':'reference_calculation_error'}
    if multiple_hosts and any(hosts[h]['status'] != 'complete' for h in system_hosts):
        return {**result,'state':'unknown','reason':'surrogate_host_unavailable'}

    def adjusted(i, row):
        value = float(row['uma_energy_eV'])
        if multiple_hosts:
            value -= float(hosts[host_ids[i]]['uma_energy_eV'])
        if not math.isfinite(value):
            raise ValueError('nonfinite energy')
        return value

    minimum_ref = min(adjusted(i,references[i]) for i in indices)
    valid = []
    unknown = 0
    counts = Counter()
    for i in indices:
        for j in range(k):
            row = candidates[(i,j)]
            status = row['status']
            flag = row.get('overlap_' + radii + '_radii')
            if flag is True:
                counts['overlap_filtered'] += 1
            elif status == 'complete' and flag is False:
                valid.append(adjusted(i,row))
                counts['converged'] += 1
            elif status in ('generation_failed','relax_not_converged') and flag is not True:
                counts[status] += 1
            else:
                counts['unknown'] += 1
                unknown += 1
    result.update(candidate_counts=dict(counts), candidates_unknown=unknown,
                  converged_candidates=len(valid))
    if not valid:
        return {**result,'state':'unknown' if unknown else 'no_converged_candidate',
                'success':None if unknown else False,'negative':None if unknown else False}
    epsilon = min(valid) - minimum_ref
    success = True if epsilon <= 0.1 else (None if unknown else False)
    negative = True if epsilon < 0 else (None if unknown else False)
    return {**result,'state':'upper_bound' if unknown else 'exact_for_policy',
            'epsilon_eV':epsilon,'success':success,'negative':negative}


def aggregate(rows, metric, subset=None):
    selected = [r for r in rows if subset is None or r['host_policy'] == subset]
    n = len(selected)
    if not n:
        return {'systems':0}
    yes = sum(r.get(metric) is True for r in selected)
    unknown = sum(r.get(metric) is None for r in selected)
    valid_eps = [r['epsilon_eV'] for r in selected
                 if r['state']=='exact_for_policy' and 'epsilon_eV' in r]
    return {'systems':n,'yes':yes,'unknown':unknown,'no':n-yes-unknown,
            'percent_lower_bound':100*yes/n,'percent_upper_bound':100*(yes+unknown)/n,
            'mean_epsilon_eV':sum(valid_eps)/len(valid_eps) if valid_eps else None,
            'mean_epsilon_systems':len(valid_eps),
            'state_counts':dict(Counter(r['state'] for r in selected))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    root=args.root
    inputs=json.loads((root/'energy/inputs/manifest.json').read_text())
    done=json.loads((root/'energy/evaluation_manifest.json').read_text())
    if (done['status']!='completed' or done['protocol']!=VERSION or
        done['checkpoint_sha256']!=CHECKPOINT_SHA or done['uma_sha256']!=UMA_SHA or done['esen_sha256']!=ESEN_SHA or
        done['energy_code_sha256']!=ENERGY_CODE_SHA or done['sampling_profile']!=PROFILE or
        done['counts']!={'hosts':NUM_SURROGATE_HOSTS,'references':NUM_RECORDS,'candidates':NUM_RECORDS*NUM_SAMPLES}):
        raise ValueError('energy evaluation identity or output count mismatch')
    if done['inputs_manifest_sha256']!=digest(root/'energy/inputs/manifest.json'):
        raise ValueError('energy input manifest changed after evaluation')
    host_ids={r['dataset_index']:r['host_id'] for r in inputs['records']}
    guest_counts={r['dataset_index']:tuple(r['guest_counts']) for r in inputs['records']}
    guest_bonds_sha=digest(root/'energy/inputs/guest_bonds.json')
    if done['guest_bonds_sha256']!=guest_bonds_sha:
        raise ValueError('guest bond input changed after evaluation')
    required_hosts=set()
    for system in inputs['systems']:
        group={host_ids[i] for i in system['indices']}
        if len(group)>1:
            required_hosts.update(group)
    if len(required_hosts)!=NUM_SURROGATE_HOSTS:
        raise ValueError('unexpected surrogate host count')
    references={};candidates={};hosts={};status_counts=Counter();ref_errors=0
    for host in inputs['hosts']:
        h=host['host_id']
        if h not in required_hosts:
            continue
        row=json.loads((root/'energy/hosts'/f'{h:04d}.json').read_text())
        if (row['host_id']!=h or row['input_sha256']!=host['geometry_sha256'] or row['protocol']!=VERSION or
            row['energy_code_sha256']!=ENERGY_CODE_SHA or row['sampling_profile']!=PROFILE or
            row['checkpoint_sha256']!=CHECKPOINT_SHA or row['uma_sha256']!=UMA_SHA or row['esen_sha256']!=ESEN_SHA):
            raise ValueError(f'host identity mismatch: {h}')
        hosts[h]=row
    for i in range(NUM_RECORDS):
        source_hash=digest(root/'generation/records'/f'{i:05d}.pt')
        ref=json.loads((root/'energy/references'/f'{i:05d}.json').read_text())
        if (ref['dataset_index']!=i or ref['source_sha256']!=source_hash or ref['protocol']!=VERSION or
            ref['energy_code_sha256']!=ENERGY_CODE_SHA or ref['sampling_profile']!=PROFILE or
            ref['checkpoint_sha256']!=CHECKPOINT_SHA or ref['uma_sha256']!=UMA_SHA or ref['esen_sha256']!=ESEN_SHA):
            raise ValueError(f'reference identity mismatch: {i}')
        references[i]=ref
        ref_errors += ref['status']!='complete'
        for j in range(NUM_SAMPLES):
            row=json.loads((root/'energy/candidates'/f'{i:05d}_{j:02d}.json').read_text())
            if (row['dataset_index']!=i or row['sample_index']!=j or row['source_sha256']!=source_hash or
                row['checkpoint_sha256']!=CHECKPOINT_SHA or row['uma_sha256']!=UMA_SHA or row['esen_sha256']!=ESEN_SHA or
                row['protocol']!=VERSION or row['energy_code_sha256']!=ENERGY_CODE_SHA or row['sampling_profile']!=PROFILE or
                row['guest_bonds_sha256']!=guest_bonds_sha):
                raise ValueError(f'candidate identity mismatch: {i}, {j}')
            candidates[(i,j)]=row
            status_counts[row['status']]+=1
    system_results=[]
    for system in inputs['systems']:
        indices=system['indices']
        groups={guest_counts[i] for i in indices}
        if len(groups)!=1:
            raise ValueError('guest counts vary within a system')
        item={'system_index':system['system_index'],'system_id':system['system_id'],'indices':indices,
              'guest_counts':list(next(iter(groups))),
              'host_ids':sorted(set(host_ids[i] for i in indices)),'results':{}}
        for radius in ('author','ase','author_guest_bond_exempt','ase_guest_bond_exempt'):
            for k in (1,10):
                item['results'][f'{radius}_k{k}']=score_one(indices,host_ids,references,candidates,hosts,k,radius)
        system_results.append(item)
    if len(system_results)!=NUM_SYSTEMS:
        raise ValueError('system count mismatch')
    summaries={}
    for radius in ('author','ase','author_guest_bond_exempt','ase_guest_bond_exempt'):
        for k in (1,10):
            key=f'{radius}_k{k}'
            data=[r['results'][key] for r in system_results]
            summaries[key]={metric:aggregate(data,metric) for metric in ('success','negative')}
            summaries[key]['one_host_subset']={metric:aggregate(data,metric,'host_cancels_exactly')
                                               for metric in ('success','negative')}
            summaries[key]['by_guest_counts']={','.join(map(str,group)):
                {metric:aggregate([item['results'][key] for item in system_results if tuple(item['guest_counts'])==group],metric)
                 for metric in ('success','negative')}
                for group in sorted({tuple(item['guest_counts']) for item in system_results})}
    result={'status':'completed','protocol':VERSION,'sampling_profile':PROFILE,'energy_code_sha256':ENERGY_CODE_SHA,'checkpoint_sha256':CHECKPOINT_SHA,
            'uma_sha256':UMA_SHA,'esen_sha256':ESEN_SHA,'systems':len(system_results),
            'records':NUM_RECORDS,'candidate_slots':NUM_RECORDS*NUM_SAMPLES,'candidate_status_counts':dict(status_counts),
            'reference_calculation_errors':ref_errors,
            'host_status_counts':dict(Counter(r['status'] for r in hosts.values())),
            'multi_host_systems':sum(len(r['host_ids'])>1 for r in system_results),
            'metrics':summaries,
            'energy_method':'UMA 1.2 energy of eSEN-FIRE-relaxed predictions vs UMA energy of dataset reference geometry',
            'sampling':'5000-step DDPM without steering',
            'host_policy':'Same-host energies cancel; multiple-host systems use reconstructed, eSEN-relaxed host surrogates.',
            'gas_reference':'Guest species counts match within each system; the same isolated-gas reference cancels.',
            'guest_count_order':['CO2','H2O','N2','O2'],
            'radii_policy':'Both source-code and ASE radii are reported. Direct-adsorbate-bond-exempt results are auxiliary only; primary results retain all pairs.',
            'guest_bond_exemption_caveat':'This sensitivity rule also ignores excessively short directly bonded adsorbate pairs; it is not the published all-pair rule.',
            'unknown_policy':'Success/negative percentages are intervals when calculations are missing; means use known minima only.',
            'inputs_manifest_sha256':digest(root/'energy/inputs/manifest.json'),
            'evaluation_manifest_sha256':digest(root/'energy/evaluation_manifest.json')}
    output=root/'energy/summary.json'; detail=root/'energy/systems.json'
    for path,value in ((detail,system_results),(output,result)):
        if path.exists():
            if json.loads(path.read_text())!=value:
                raise ValueError(f'existing energy score differs from recomputation: {path}')
            continue
        temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
        with temp.open('x',encoding='utf-8') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temp,path);temp.unlink()
    print('ENERGY_SCORING_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
