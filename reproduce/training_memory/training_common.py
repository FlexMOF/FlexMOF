"""Shared training-memory benchmark helpers."""
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ['FLEXMOF_TRAIN_MEMORY_RUN']).resolve()
CKPT = Path(os.environ['FLEXMOF_STRUCTURE_CHECKPOINT']).resolve()
DATA = Path(os.environ['FLEXMOF_DATA']).resolve() / 'bw_db'
CHECKPOINT_SHA256 = '351404ef67ab445591c5639f550176ca14caf565272ddda40decf1a03b4f35e6'
from flexmof.checkpoint import load_checkpoint, load_model_weights
from flexmof.config import config_from_checkpoint
from flexmof.data import MOFDataset, collate_mofs
from flexmof.training import MOFTrainingModel, optimizer_parameter_groups
from flexmof.ema import ModelEMA


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def move(x):
    if torch.is_tensor(x):return x.cuda(non_blocking=False)
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    if isinstance(x,list):return [move(v) for v in x]
    if isinstance(x,tuple):return tuple(move(v) for v in x)
    return x


def build(dropout_off=False):
    original=json.loads((HERE/'train_config.json').read_text())
    if sha(CKPT) != CHECKPOINT_SHA256: raise ValueError('unexpected structure checkpoint')
    saved=load_checkpoint(CKPT)
    runtime_keys={'checkpoint_path','data_root','dataset_subdir','output_dir','metric_path',
                  'micro_batch_size','max_batches','num_samples','seed','local_rank','rank','world_size'}
    overrides={k:v for k,v in original['architecture'].items() if k not in runtime_keys}
    if dropout_off:overrides.update(dropout=0.0,act_dropout=0.0,attention_dropout=0.0)
    config=config_from_checkpoint(saved['args'],str(CKPT),str(DATA.parent),str(OUT/'scratch'),
                                  'bw_db',4,-1,12347,overrides=overrides,num_samples=1)
    assert config.pbc_cutoff==40 and config.use_per_atom_outcell_index and not config.mof_disable_pbc
    assert config.pbc_sparse_edge_chunk_size==2097152 and config.pbc_activation_checkpoint_edges==8000000
    assert not config.mof_joint_denoiser_only and config.mof_joint_composition
    assert config.attention_dropout==0.0
    if not dropout_off: assert config.dropout==config.act_dropout==0.1
    model=MOFTrainingModel(config)
    load_model_weights(model,saved,CKPT);del saved
    return model.cuda().train(),config,original


def sample_for(config, key):
    cohort=json.loads((HERE/'cohort.json').read_text())
    row=next(x for x in cohort['samples'] if x['data_key']==str(key))
    dataset=MOFDataset(DATA,split='test', max_atoms=4096,composition_config=config)
    index=row['train_dataset_index']
    assert str(dataset.keys[index])==row['data_key'] and int(dataset.sizes[index])==row['atoms']
    payload=dataset._transaction.get(row['data_key'].encode())
    if payload is None:raise ValueError(f'Missing LMDB payload for frozen key {row["data_key"]}')
    assert hashlib.sha256(payload).hexdigest()==row['input_sha256']
    return row,dataset[index]


def make_batch(item,batch_size):
    return move(collate_mofs([copy.deepcopy(item) for _ in range(batch_size)]))


def optimizer_and_ema(model,run):
    t=run['training'];step=713433
    progress=min(1.0,(step-t['warmup_steps'])/max(1,t['max_steps']-t['warmup_steps']))
    lr=t['learning_rate']*0.5*(1+math.cos(math.pi*progress))
    optimizer=torch.optim.AdamW(optimizer_parameter_groups(model,t['composition_lr_multiplier']),
                                lr=lr,betas=(0.9,0.999),eps=1e-8,
                                weight_decay=t['weight_decay'],fused=True)
    return optimizer,ModelEMA(model,t['ema_decay']),lr


def instrument_neighbors(model,counts,enabled):
    expander=model.net.cell_expander;original=expander.expand_per_atom_streaming
    def wrapped(*args,**kwargs):
        result=original(*args,**kwargs)
        if enabled['value']:
            edges=result['packed_edges'];n=int(result['max_num_atoms']);k=int(result['candidate_count'])
            q=torch.div(edges,k,rounding_mode='floor'); c=torch.bincount(q,minlength=counts['batch_size']*n).reshape(-1,n)
            physical=c[:,:counts['atoms']]
            counts.update(edges_all_tokens=int(edges.numel()),physical_neighbor_mean=float(physical.float().mean().item()),
                          physical_neighbor_max=int(physical.max().item()),translation_candidates=int(result['selected_cells'].shape[0]))
            positions=args[0].float(); atoms=args[4]
            central=(torch.cdist(positions,positions)<float(result['cutoff']))
            central &= atoms.ne(0)[:,:,None] & atoms.ne(0)[:,None,:]
            physical_all=(c+central.sum(dim=-1))[:,:counts['atoms']]
            cell_index=torch.div(edges.remainder(k),n,rounding_mode='floor')
            source=edges.remainder(n)
            eb=torch.div(q,n,rounding_mode='floor')
            per_sample=[]
            for i in range(counts['batch_size']):
                select=eb.eq(i)
                unique_cells=torch.unique(cell_index[select])
                translations=result['selected_cells'][unique_cells].cpu().tolist()
                image_nodes=int(torch.unique(cell_index[select]*n+source[select]).numel())
                ranges=[]
                for axis in range(3):
                    vals=[int(x[axis]) for x in translations]+[0]
                    ranges.append([min(vals),max(vals)])
                lengths=[hi-lo+1 for lo,hi in ranges]
                per_sample.append({'periodic_edges':int(select.sum().item()),
                    'central_edges':int(central[i].sum().item()),
                    'physical_neighbor_mean_including_central':float(physical_all[i].float().mean().item()),
                    'physical_neighbor_max_including_central':int(physical_all[i].max().item()),
                    'central_queries_including_corners':int(atoms[i].ne(0).sum().item()),
                    'distinct_periodic_image_nodes':image_nodes,
                    'used_image_cells':len(translations),
                    'used_translation_bounds':ranges,
                    'rectangular_cover_shape':lengths,
                    'rectangular_cover_cells':math.prod(lengths),
                    'used_translations':translations})
            counts['per_sample']=per_sample
            counts['neighbor_definition']='Periodic edges plus central pairs within 40 A; includes corner tokens as neighbors; padded tokens excluded.'
        return result
    expander.expand_per_atom_streaming=wrapped


def tensor_digest(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
