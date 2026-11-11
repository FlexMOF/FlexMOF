"""Group only missing (sample, candidate) pairs into bounded same-structure batches."""
from __future__ import annotations

CANDIDATES=(0,1,2,3,4)
BUCKETS=((0,96,32),(96,192,16),(192,384,8),(384,512,4),(512,768,4),(768,4096,1))


def make_plan(assigned, manifest, completed, capacities):
    assigned=list(assigned)
    if len(assigned)!=len(set(assigned)):
        raise ValueError('Rank assignment contains duplicate sample IDs')
    allowed={(i,c) for i in assigned for c in CANDIDATES}
    completed=set(completed)
    if not completed<=allowed:
        raise ValueError('Completed set contains an unknown sample or candidate')
    plan=[]
    for lower,upper,baseline in BUCKETS:
        cap=capacities.get((lower,upper),baseline)
        if type(cap) is not int or cap<1:
            raise ValueError('Batch capacity must be positive')
        ordered=sorted((i for i in assigned if lower<manifest[i]['num_atoms']<=upper),
                       key=lambda i:(manifest[i]['num_atoms'],i))
        batch=[]
        for sample in ordered:
            missing=[(sample,c) for c in CANDIDATES if (sample,c) not in completed]
            while missing:
                available=cap-len(batch)
                # Keep a sample together if it fits in a fresh batch. For very
                # large structures split only when the cap cannot fit all seeds.
                if batch and len(missing)>available and len(missing)<=cap:
                    plan.append(batch);batch=[];continue
                count=min(len(missing),cap-len(batch))
                batch.extend(missing[:count]);missing=missing[count:]
                if len(batch)==cap:
                    plan.append(batch);batch=[]
        if batch:plan.append(batch)
    flat=[pair for batch in plan for pair in batch]
    if len(flat)!=len(set(flat)) or set(flat)!=(allowed-completed):
        raise ValueError('Plan must cover missing candidates exactly once')
    return plan
