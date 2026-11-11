"""Resumable 32-rank worker for grouped seed-forward reuse."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import time

from pending_plan import BUCKETS,make_plan


def atomic_create(path,value):
    """Publish a complete record without ever replacing an existing path."""
    import worker
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value,stream,allow_nan=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def choose_smoke_pairs(context):
    selected=set(context['selected'])
    entry=min((e for e in context['manifest'] if e['sample_id'] not in selected),
              key=lambda e:(e['num_atoms'],e['sample_id']))
    return [(entry['sample_id'],0),(entry['sample_id'],1)]


def run(args,context):
    import sys
    import torch
    import worker
    sys.path[:0]=[str(context['run']/'source/src'),str(context['run']/'cuda_extension'),str(context['run'])]
    from flexmof.runtime import build_config
    from flexmof.data import MOFDataset,collate_mofs
    from flexmof.sampler import MOFSampler
    from flexmof.model.diffusion import DDPM
    from noise import PerStructureNoise
    from group_sampling import sample_and_match_grouped
    import flexmof.sampler as sampler_module
    sampler_module.tqdm=lambda it,**kwargs:it
    rank=args.rank;local_gpu=rank%8 if args.local_gpu is None else args.local_gpu
    worker.require(0<=local_gpu<torch.cuda.device_count(),'Requested local GPU is unavailable')
    torch.cuda.set_device(local_gpu);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    isolated=args.smoke
    if isolated:
        output=context['attempt']/'smoke'
        output.mkdir(parents=True,exist_ok=False)
    else:
        output=context['run']/'csp'
        claim=context['attempt']/'claims'/f'rank{rank}.json';claim.parent.mkdir(parents=True,exist_ok=True)
        with claim.open('x') as stream:json.dump({'pid':os.getpid(),'rank':rank,'attempt_id':context['attempt_id']},stream)
    for directory in ('structures','records','metrics','evidence'):(output/directory).mkdir(parents=True,exist_ok=True)
    protocol=context['protocol'];overrides=protocol['overrides'].copy()
    if isolated:overrides['num_timesteps_stepsize']=-100
    config=build_config(protocol['checkpoint'],protocol['data_root'],str(output/'structures'),
                        overrides=overrides,seed=12347)
    config.metric_path=str(output/'metrics'/f'rank{rank}')
    worker.require(config.coordinate_sampler=='ddpm' and config.num_timesteps==5000,'Expected DDPM5000')
    worker.require(config.num_timesteps_stepsize==(-100 if isolated else -1),'Wrong DDPM step size')
    worker.require(not config.mof_disable_pbc and config.num_recycles==0,'PBC protocol changed')
    sampler=MOFSampler(config).cuda().eval();sampler.requires_grad_(False)
    worker.require(isinstance(sampler.diffusion_process,DDPM),'Sampler is not DDPM')
    steps=50 if isolated else 5000
    worker.require(len(sampler.diffusion_process.timesteps)==steps,'Unexpected DDPM timestep count')
    worker.require(sampler.converter.protocol.matcher_kwargs()==worker.MATCHER,'Matcher protocol changed')
    worker.require(sampler.net.joint_composition and not sampler.net.gaussian_composition and
                   not sampler.net.lgflow_composition,'Unsupported composition mode')
    dataset=MOFDataset(Path(protocol['data_root'])/'bw_db',split='test',max_atoms=4096,composition_config=config)
    manifest=context['manifest'];worker.require(len(dataset)==worker.FULL_COUNT,'Dataset length changed')
    for e in manifest:
        worker.require(str(dataset.keys[e['dataset_index']])==str(e['data_key']),'Dataset key ordering changed')
    worker.atomic(context['attempt']/'configs'/f"{'smoke' if isolated else 'rank'}{rank}.json",
                  {'config':vars(config),'evaluation_identity':context['identity']})
    assigned=context['assignments'][rank]
    completed=set()
    if isolated:
        queue=[choose_smoke_pairs(context)]
        worker.require(queue[0][0][0] not in set(context['selected']),'Smoke must be outside evaluated cohort')
    else:
        for i in assigned:
            for c in worker.ACTIVE_CANDIDATES:
                path=output/'records'/f'{i:05d}_{c}.json'
                if path.exists() or path.is_symlink():
                    worker.validate_committed(context,manifest[i],c,output=output)
                    completed.add((i,c))
        caps={(lo,hi):worker.batch_cap(context,lo,hi,base) for lo,hi,base in BUCKETS}
        queue=make_plan(assigned,manifest,completed,caps)
    reused=len(completed);generated=0
    def move(v):
        if isinstance(v,torch.Tensor):return v.cuda()
        if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
        if isinstance(v,list):return [move(x) for x in v]
        if isinstance(v,tuple):return tuple(move(x) for x in v)
        return v
    while queue:
        worker.check_failures(context)
        pairs=queue.pop(0)
        worker.require(len(pairs)==len(set(pairs)) and all(p not in completed for p in pairs),'Duplicate planned pair')
        for i,c in pairs:
            path=output/'records'/f'{i:05d}_{c}.json'
            worker.require(not path.exists() and not path.is_symlink(),'Refusing to resample an existing candidate')
        batch=move(collate_mofs([copy.deepcopy(dataset[manifest[i]['dataset_index']]) for i,c in pairs]))
        batch['idx']=torch.tensor([i for i,c in pairs],device='cuda',dtype=torch.long)
        entries=[dict(num_atoms=manifest[i]['num_atoms'],csp_seed=manifest[i]['candidate_seeds'][c]) for i,c in pairs]
        noise=PerStructureNoise(sampler,entries);started=time.monotonic()
        try:
            try:
                values,stats=sample_and_match_grouped(sampler,batch,pairs,steps)
                worker.require(noise.calls==steps+1,'Per-structure noise call count changed')
                anchors=dict(stats['sequence']['anchor_pairs'])
                for (i,c),metrics in zip(pairs,values):
                    worker.require(metrics.get('match') in (0,1),'Invalid match value')
                    hashes={}
                    for kind in ('original','sampled'):
                        cif=output/'structures'/f'{kind}_{i}_{c}.cif'
                        worker.require(cif.is_file() and cif.stat().st_size>0,f'Missing sampled CIF: {cif}')
                        hashes[kind+'_sha256']=worker.sha(cif)
                    seqstats={k:stats['sequence'][k] for k in ('calls','source_forwards','reused_candidate_forwards')}
                    seqstats.update(anchor_candidate_id=anchors[i],applied=(sum(x==i for x,_ in pairs)>1),
                                    module_sha256=worker.sha(worker.VARIANT_DIR/'group_seq_adapter.py'))
                    atomic_create(output/'records'/f'{i:05d}_{c}.json',
                        dict(sample_id=i,data_key=manifest[i]['data_key'],candidate_id=c,
                             seed=manifest[i]['candidate_seeds'][c],status='complete',metrics=metrics,
                             batch_size=len(pairs),**hashes,producer_attempt_id=context['attempt_id'],
                             evaluation_identity=context['identity'],
                             graph_bias_cache=dict(module_sha256=worker.sha(worker.VARIANT_DIR/'graph_bias_cache.py'),
                                                   **stats['graph']),seq_forward_reuse=seqstats))
                    completed.add((i,c));generated+=1
            except torch.cuda.OutOfMemoryError as error:
                sampler.diffusion_process.reset();torch.cuda.empty_cache()
                written=any((output/'structures'/f'{kind}_{i}_{c}.cif').exists()
                            for i,c in pairs for kind in ('original','sampled'))
                worker.require(not written,'OOM after structure writes: manual reconciliation required')
                if len(pairs)>1:
                    cut=max(1,len(pairs)//2);queue=[pairs[:cut],pairs[cut:]]+queue
                    print(json.dumps({'event':'oom_split','rank':rank,'old_batch_size':len(pairs),'new_sizes':[cut,len(pairs)-cut]}),flush=True)
                    continue
                i,c=pairs[0]
                atomic_create(output/'records'/f'{i:05d}_{c}.json',
                    dict(sample_id=i,data_key=manifest[i]['data_key'],candidate_id=c,
                         seed=manifest[i]['candidate_seeds'][c],status='generation_error',
                         error=f'{type(error).__name__}: {error}',producer_attempt_id=context['attempt_id'],
                         evaluation_identity=context['identity']))
                completed.add((i,c));generated+=1
            worker.atomic(context['attempt']/'progress'/f"{'smoke' if isolated else 'rank'}{rank}.json",
                dict(rank=rank,last_pairs=pairs,elapsed_seconds=time.monotonic()-started,
                     remaining_batches=len(queue),generated=generated,reused=reused,attempt_id=context['attempt_id']))
            print(json.dumps({'rank':rank,'last_pairs':pairs,'generated':generated,'reused':reused,
                              'remaining_batches':len(queue),'attempt_id':context['attempt_id']}),flush=True)
        finally:noise.remove()
    if isolated:
        for i,c in completed:
            row=worker.validate_committed(context,manifest[i],c,output=output,allow_imported=False)
            worker.require(row['status']=='complete','Smoke generation failed')
        worker.require(generated==2 and reused==0,'Smoke must generate two independent candidates')
        worker.atomic(output/'passed.json',dict(attempt_id=context['attempt_id'],steps=steps,generated=generated,
            selected_ids=sorted({i for i,c in completed}),candidate_ids=sorted(c for i,c in completed),
            outside_evaluation_cohort=True,evaluation_identity=context['identity']))
    else:
        counts=worker.completed_counts(context,rank)
        worker.atomic(context['attempt']/'done'/f'rank{rank}.json',
            dict(attempt_id=context['attempt_id'],rank=rank,assigned=len(assigned),
                 assignment_sha256=worker.ids_sha(assigned),records=len(worker.ACTIVE_CANDIDATES)*len(assigned),
                 counts=counts,generated=generated,reused=reused,evaluation_identity=context['identity']))
