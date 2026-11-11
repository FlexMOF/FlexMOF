"""One independent official AtomMOF-L training trial on a paired test structure."""
import argparse
import copy
import hashlib
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import lmdb
import numpy as np
import torch
from omegaconf import OmegaConf

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['FLEXMOF_TRAIN_MEMORY_RUN']).resolve()
SOURCE=Path(os.environ['ATOMMOF_ROOT']).resolve()
ATOM_DATA=Path(os.environ['ATOMMOF_BW_TEST']).resolve()
CKPT=Path(os.environ['ATOMMOF_CHECKPOINT']).resolve()
sys.path.insert(0,str(SOURCE))
from src.data.bwdb_dataset import BWDBDataset
from src.data.bwdb_datamodule import collate_fn
from src.data.feature.featurizer import Featurizer
from src.models.simplemof_module import SimpleMOFModule

def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def move(x):
    if torch.is_tensor(x):return x.cuda(non_blocking=False)
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    if isinstance(x,list):return [move(v) for v in x]
    return x
def digest(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.json.tmp')
    with temporary.open('x') as f:json.dump(value,f,indent=2,allow_nan=False,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--batch-size',type=int,required=True,choices=(1,4))
    p.add_argument('--time-limit',type=int,default=900);p.add_argument('--warmup',type=int,default=2);p.add_argument('--steps',type=int,default=3)
    a=p.parse_args();out=ROOT/'results';out.mkdir(exist_ok=True);dest=out/f'train_atommof_key{a.key}_b{a.batch_size}.json'
    assert not dest.exists()
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.90,0);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status={'model':'AtomMOF-L BWDB official','key':a.key,'batch_size':a.batch_size,'status':'starting','phase':'initialization',
            'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'memory_fraction_cap':.90,'precision':'FP32, TF32 disabled','torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'warmup_steps':a.warmup,'measured_steps':a.steps,'time_limit_seconds':a.time_limit,
            'scope':'synchronized official structure forward/loss + backward + clipping + AdamW; batch construction, input loading and logging excluded',
            'memory_scope':'peak allocated/reserved includes model, gradients and AdamW state; NCCL and other non-PyTorch allocations excluded',
            'data_split':'test','batch_construction':'four or one independently sampled priors for the same structure',
            'checkpoint_sha256':'b077fd91b9d1fe9363428fe8fc4b050a1395a6ecd2219c014583cb551465d303',
            'cohort_sha256':hashlib.sha256((HERE/'cohort.json').read_bytes()).hexdigest(),
            'periodic_neighbor_note':'Official dense atom-pair attention/lattice conditioning; no FlexMOF 40 Å periodic multigraph or directly comparable periodic-edge count is built.',
            'seed_note':'Same integer seed convention as FlexMOF; model-specific priors, objectives and RNG algorithms differ.'}
    start=time.monotonic()
    def alarm(*_):raise TimeoutError(f'{a.time_limit}s trial limit')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(a.time_limit)
    try:
        cohort=json.loads((HERE/'cohort.json').read_text());row=next(x for x in cohort['samples'] if x['data_key']==a.key);status['sample']=row
        digest_ckpt=hashlib.sha256()
        with CKPT.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest_ckpt.update(block)
        if digest_ckpt.hexdigest() != status['checkpoint_sha256']: raise ValueError('unexpected AtomMOF checkpoint')
        assert (ATOM_DATA.parent/'bwdb_test_metadata.pkl').is_file(),'Do not create metadata in the original data directory'
        env=lmdb.open(str(ATOM_DATA),subdir=False,readonly=True,lock=False,readahead=False,meminit=False)
        with env.begin(write=False) as txn:payload=txn.get(row['atom_lmdb_key'].encode())
        assert payload and hashlib.sha256(payload).hexdigest()==row['atom_input_sha256'];env.close()
        pred_args=OmegaConf.create({'num_samples':1,'sampling_steps':10,'use_sde_coords':False,
                                   'sampling_center_coords':True,'refine_final':False,
                                   'steering_args':{'fk_steering':False,'energy_fn':'mlip'}})
        model=SimpleMOFModule.load_from_checkpoint(str(CKPT),map_location='cpu',predict_args=pred_args)
        defaults=OmegaConf.load(str(SOURCE/'configs/model/simple.yaml'))['training_args']
        t=model.training_args;missing={}
        for key in ('loss_type','coord_loss_weight','lengths_loss_weight','angle_loss_weight','use_dist_loss',
                    'dist_cutoff','use_dist_schedule','dist_loss_weight','lr','weight_decay'):
            if key not in t:missing[key]=defaults[key];t[key]=defaults[key]
        original_multiplicity=t.get('train_multiplicity',1);t['train_multiplicity']=1
        assert not model.use_ema,'Checkpoint unexpectedly enables EMA; implement and measure its official callback first'
        prior=model.structure_module.prior_sampler
        dataset=BWDBDataset(str(ATOM_DATA),Featurizer(),prior_sampler=None,max_num_atoms=None,random_transform=False)
        index=int(row['atom_dataset_index']);stored=dataset.keys[index]
        if isinstance(stored,bytes):stored=stored.decode('utf8')
        assert str(stored)==row['atom_lmdb_key'];seed_all(row['seed']);item=dataset[index]
        assert int(item['ref_element'].numel())==row['atoms']
        model=model.cuda().float().train()
        optim=torch.optim.AdamW(model.parameters(),lr=float(t['lr']),weight_decay=float(t['weight_decay']))
        clip=float(OmegaConf.load(str(SOURCE/'configs/trainer/default.yaml')).get('gradient_clip_val',0))
        status.update(model_parameters=sum(x.numel() for x in model.parameters()),
                      optimizer='official PyTorch AdamW defaults; fused not forced',learning_rate=float(t['lr']),
                      optimizer_weight_decay=float(t['weight_decay']),ema_enabled=False,ema_gpu_seconds=0.0,
                      gradient_clip=clip,gradient_clip_source='repository configs/trainer/default.yaml; trainer clip not encoded in checkpoint',
                      checkpoint_training_args=dict(t),checkpoint_missing_training_defaults=missing,
                      checkpoint_train_multiplicity=original_multiplicity,benchmark_train_multiplicity=1,
                      use_kernels=bool(model.use_kernels),dense_attention_valid_pairs_per_item=int(row['atoms'])**2)
        def step(seed,label):
            seed_all(seed);optim.zero_grad(set_to_none=True);items=[]
            for _ in range(a.batch_size):
                feature=copy.deepcopy(item);feature.update(prior.sample(feature));items.append(feature)
            batch=move(collate_fn(items));prior_coord_sha=digest(batch['cart_coords_0']);prior_lattice_sha=digest(batch['lattice_0'])
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
            status['phase']=label+'_forward';start_step=time.perf_counter();events[0].record()
            output=model.structure_module(feats=batch,multiplicity=1)
            losses=model.structure_module.compute_loss(batch,output,multiplicity=1,loss_type=t['loss_type'],
                use_dist_loss=bool(t['use_dist_loss']),dist_cutoff=float(t['dist_cutoff']),use_dist_schedule=bool(t['use_dist_schedule']))
            loss=(float(t['coord_loss_weight'])*losses['coord_loss'].mean()+float(t['lengths_loss_weight'])*losses['length_loss'].mean()
                  +float(t['angle_loss_weight'])*losses['angle_loss'].mean()+float(t['dist_loss_weight'])*losses['dist_loss'].mean())
            events[1].record();status['phase']=label+'_backward';loss.backward();events[2].record()
            status['phase']=label+'_clip';norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip) if clip>0 else None;events[3].record()
            status['phase']=label+'_optimizer';optim.step();events[4].record();torch.cuda.synchronize();wall=time.perf_counter()-start_step
            if not torch.isfinite(loss).item() or (norm is not None and not torch.isfinite(norm).item()):raise ValueError('Nonfinite loss or gradient')
            return {'seed':seed,'prior_coordinate_sha256':prior_coord_sha,'prior_lattice_sha256':prior_lattice_sha,
                    'sampled_times_sha256':digest(output['times']),
                    'wall_seconds':wall,'forward_gpu_seconds':events[0].elapsed_time(events[1])/1000,
                    'backward_gpu_seconds':events[1].elapsed_time(events[2])/1000,
                    'clip_gpu_seconds':events[2].elapsed_time(events[3])/1000,
                    'optimizer_gpu_seconds':events[3].elapsed_time(events[4])/1000,'ema_gpu_seconds':0.0,
                    'loss':float(loss.detach().item()),'gradient_norm':float(norm.detach().item()) if norm is not None else None,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        for i in range(a.warmup):step(row['seed']+i,'warmup_'+str(i))
        torch.cuda.empty_cache();status['steps']=[]
        for i in range(a.steps):status['steps'].append(step(row['seed']+1000+i,'measured_'+str(i)))
        status['peak_allocated_bytes']=max(x['peak_allocated_bytes'] for x in status['steps'])
        status['peak_reserved_bytes']=max(x['peak_reserved_bytes'] for x in status['steps'])
        status['status']='ok';status['phase']='complete'
    except Exception as error:
        status.update(status='oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'timeout' if isinstance(error,TimeoutError) else 'error',
                      error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()[-10000:],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:signal.alarm(0)
    status['total_process_seconds']=time.monotonic()-start;write(dest,status)
    print('TRIAL '+json.dumps({'key':a.key,'batch_size':a.batch_size,'model':'atommof','status':status['status'],'phase':status['phase']}),flush=True)

if __name__=='__main__':main()
