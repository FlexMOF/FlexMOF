"""One independent FlexMOF training trial on a fixed, paired test structure."""
import argparse
import hashlib
import json
import os
import signal
import tempfile
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from training_common import OUT, HERE, CHECKPOINT_SHA256, build, instrument_neighbors, make_batch, optimizer_and_ema, sample_for, seed_all, tensor_digest

def write(path, value):
    temporary=path.with_suffix('.json.tmp')
    with temporary.open('x') as f:json.dump(value,f,indent=2,allow_nan=False,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--batch-size',type=int,required=True,choices=(1,4))
    p.add_argument('--time-limit',type=int,default=900);p.add_argument('--warmup',type=int,default=2);p.add_argument('--steps',type=int,default=3)
    a=p.parse_args();out=OUT/'results';out.mkdir(exist_ok=True);dest=out/f'train_flex_key{a.key}_b{a.batch_size}.json'
    assert not dest.exists()
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.90,0);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status={'model':'FlexMOF','key':a.key,'batch_size':a.batch_size,'status':'starting','phase':'initialization',
            'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'memory_fraction_cap':.90,'precision':'FP32, TF32 disabled','torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'warmup_steps':a.warmup,'measured_steps':a.steps,'time_limit_seconds':a.time_limit,
            'scope':'synchronized model forward/loss + backward + clipping + AdamW + EMA; batch construction, input loading and logging excluded',
            'memory_scope':'peak allocated/reserved includes model, gradients, AdamW state and EMA; NCCL and other non-PyTorch allocations excluded',
            'data_split':'test','batch_construction':'four or one independent-noise copies of the same structure',
            'checkpoint_sha256':CHECKPOINT_SHA256,
            'cohort_sha256':hashlib.sha256((HERE/'cohort.json').read_bytes()).hexdigest()}
    start=time.monotonic()
    def alarm(*_):raise TimeoutError(f'{a.time_limit}s trial limit')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(a.time_limit)
    try:
        model,config,run=build();row,item=sample_for(config,a.key);status['sample']=row
        assert config.pbc_cutoff==40 and config.use_per_atom_outcell_index and not config.mof_disable_pbc
        optim,ema,lr=optimizer_and_ema(model,run)
        joint=model.net.mof_joint;flow=joint.flow_loss;predict=joint.denoiser.predict_clean
        trace={'in_flow':False,'calls':0}
        def wrapped_predict(*args,**kwargs):
            if trace['in_flow']:trace['calls']+=1
            return predict(*args,**kwargs)
        def wrapped_flow(*args,**kwargs):
            trace['calls']=0;trace['in_flow']=True
            try:return flow(*args,**kwargs)
            finally:trace['in_flow']=False
        joint.denoiser.predict_clean=wrapped_predict;joint.flow_loss=wrapped_flow
        status.update(pbc_cutoff_angstrom=40,periodic_neighbor_cap=None,
                      activation_checkpoint_note='original conditional periodic-edge activation checkpoint policy',
                      pbc_activation_checkpoint_edges=config.pbc_activation_checkpoint_edges,
                      optimizer='fused AdamW; beta=(0.9,0.999); weight_decay='+str(run['training']['weight_decay']),
                      learning_rate=lr,ema_decay=run['training']['ema_decay'],gradient_clip=run['training']['gradient_clip'],
                      model_parameters=sum(x.numel() for x in model.net.parameters()))
        def step(seed, label):
            seed_all(seed);model.diffusion_noise.torch_generator=torch.Generator(device='cuda').manual_seed(seed+200000003)
            optim.zero_grad(set_to_none=True);batch=make_batch(item,a.batch_size)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            events=[torch.cuda.Event(enable_timing=True) for _ in range(6)]
            status['phase']=label+'_forward';start_step=time.perf_counter();events[0].record()
            result=model(batch);loss=result['loss'];events[1].record()
            status['phase']=label+'_backward';loss.backward();events[2].record()
            status['phase']=label+'_clip';norm=torch.nn.utils.clip_grad_norm_(model.parameters(),run['training']['gradient_clip']);events[3].record()
            status['phase']=label+'_optimizer';optim.step();events[4].record()
            status['phase']=label+'_ema';ema.update(model);events[5].record();torch.cuda.synchronize()
            wall=time.perf_counter()-start_step
            if not torch.isfinite(loss).item() or not torch.isfinite(norm).item():raise ValueError('Nonfinite loss or gradient')
            return {'seed':seed,'coordinate_noise_generator_seed':seed+200000003,
                    'wall_seconds':wall,'forward_gpu_seconds':events[0].elapsed_time(events[1])/1000,
                    'backward_gpu_seconds':events[1].elapsed_time(events[2])/1000,
                    'clip_gpu_seconds':events[2].elapsed_time(events[3])/1000,
                    'optimizer_gpu_seconds':events[3].elapsed_time(events[4])/1000,
                    'ema_gpu_seconds':events[4].elapsed_time(events[5])/1000,
                    'loss':float(loss.detach().item()),'gradient_norm':float(norm.detach().item()),
                    'noised_positions_sha256':tensor_digest(batch['pos']),
                    'noise_sigma_sha256':tensor_digest(batch['sqrt_one_minus_alphas_cumprod_t']),
                    'sampled_times_sha256':tensor_digest(result['time']),
                    'flow_predict_clean_calls':trace['calls'],'flow_self_conditioning_used':trace['calls']==2,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        for i in range(a.warmup):step(row['seed']+i,'warmup_'+str(i))
        torch.cuda.empty_cache();status['steps']=[]
        for i in range(a.steps):status['steps'].append(step(row['seed']+1000+i,'measured_'+str(i)))
        status['peak_allocated_bytes']=max(x['peak_allocated_bytes'] for x in status['steps'])
        status['peak_reserved_bytes']=max(x['peak_reserved_bytes'] for x in status['steps'])
        status['status']='ok';status['phase']='complete'
        # Probe exactly the first measured random seed after timing.  Input-noise
        # digest must match before its graph can be associated with that step.
        counts={'batch_size':a.batch_size,'atoms':row['atoms']};enabled={'value':True}
        try:
            status['neighbor_probe_status']='starting';instrument_neighbors(model,counts,enabled)
            seed=row['seed']+1000;seed_all(seed);model.diffusion_noise.torch_generator=torch.Generator(device='cuda').manual_seed(seed+200000003)
            probe=make_batch(item,a.batch_size)
            with torch.no_grad():model(probe)
            enabled['value']=False;torch.cuda.synchronize()
            same=tensor_digest(probe['pos'])==status['steps'][0]['noised_positions_sha256']
            status['neighbor_probe_status']='matched' if same else 'noise_hash_mismatch'
            if same:status['neighbor_profile_first_measured_seed']=counts
        except Exception as error:
            enabled['value']=False;status['neighbor_probe_status']='error';status['neighbor_probe_error']=f'{type(error).__name__}: {error}'
    except Exception as error:
        status.update(status='oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'timeout' if isinstance(error,TimeoutError) else 'error',
                      error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()[-10000:],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:signal.alarm(0)
    status['total_process_seconds']=time.monotonic()-start;write(dest,status)
    print('TRIAL '+json.dumps({'key':a.key,'batch_size':a.batch_size,'model':'flex','status':status['status'],'phase':status['phase']}),flush=True)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as d:
        dist.init_process_group('gloo',init_method=f'file://{d}/init',rank=0,world_size=1)
        try:main()
        finally:
            if dist.is_initialized():dist.destroy_process_group()
