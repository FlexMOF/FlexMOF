"""Approximate sequence-forward sharing, restricted to seeds of the same sample."""
from __future__ import annotations


class GroupedSeedSequenceReuse:
    def __init__(self, module, pairs):
        pairs=list(pairs)
        if not pairs or len(pairs)!=len(set(pairs)):
            raise ValueError('Batch must contain unique nonempty (sample, candidate) pairs')
        if any(type(i) is not int or type(c) is not int or c not in range(5) for i,c in pairs):
            raise ValueError('Invalid sample or candidate identity')
        self.module=module
        self.original=module.forward
        self.pairs=pairs
        anchors=[]; inverse=[]; source_index={}
        for position,(sample,_) in enumerate(pairs):
            if sample not in source_index:
                source_index[sample]=len(anchors);anchors.append(position)
            inverse.append(source_index[sample])
        self.anchor_positions=anchors
        self.inverse=inverse
        self.calls=0
        self.reused_candidate_forwards=0
        self.source_forwards=0
        self.anchor_index=self.inverse_index=self.lookup=None

    def __call__(self,x,pair,batch,periodic,padding_mask):
        import torch
        if self.module.training or not torch.is_inference_mode_enabled():
            raise RuntimeError('Grouped sequence reuse requires evaluation/inference mode')
        n=len(self.pairs)
        if batch['pos'].shape[0]!=n or x.shape[1]!=n or pair.shape[2]!=n or padding_mask.shape[0]!=n:
            raise ValueError('Sequence batch dimensions differ from planned identities')
        self.calls+=1
        self.source_forwards+=len(self.anchor_positions)
        self.reused_candidate_forwards+=n-len(self.anchor_positions)
        if n==len(self.anchor_positions):
            return self.original(x,pair,batch,periodic,padding_mask)
        if not periodic.get('streaming_per_atom') or 'packed_edges' not in periodic:
            raise ValueError('Expected validated streaming PBC geometry')
        packed=periodic['packed_edges'];distances=periodic['edge_distance']
        if packed.numel()!=distances.numel():
            raise ValueError('Periodic edge and distance counts differ')
        device=packed.device
        if self.anchor_index is None:
            self.anchor_index=torch.tensor(self.anchor_positions,dtype=torch.long,device=device)
            self.inverse_index=torch.tensor(self.inverse,dtype=torch.long,device=device)
            self.lookup=torch.full((n,),-1,dtype=torch.long,device=device)
            self.lookup[self.anchor_index]=torch.arange(len(self.anchor_positions),device=device)
        if self.anchor_index.device!=device:
            raise ValueError('Batch device changed')
        span=periodic['max_num_atoms']*periodic['candidate_count']
        old_batch=torch.div(packed,span,rounding_mode='floor')
        mapped=self.lookup[old_batch]
        mask=mapped.ge(0)
        sliced={'streaming_per_atom':True,'max_num_atoms':periodic['max_num_atoms'],
                'candidate_count':periodic['candidate_count'],
                'edge_chunk_size':periodic['edge_chunk_size'],'cutoff':periodic['cutoff'],
                'packed_edges':packed[mask]+(mapped[mask]-old_batch[mask])*span,
                'edge_distance':distances[mask]}
        batch0={'pos':batch['pos'].index_select(0,self.anchor_index)}
        if self.module.use_rope:
            batch0['position_ids']=batch['position_ids'].index_select(0,self.anchor_index)
        source_x,source_pair=self.original(x.index_select(1,self.anchor_index),
            pair.index_select(2,self.anchor_index).clone(),batch0,sliced,
            padding_mask.index_select(0,self.anchor_index))
        return source_x.index_select(1,self.inverse_index),source_pair.index_select(0,self.inverse_index)

    def stats(self):
        return {'calls':self.calls,'source_forwards':self.source_forwards,
                'reused_candidate_forwards':self.reused_candidate_forwards,
                'pairs':self.pairs,'anchor_pairs':[self.pairs[i] for i in self.anchor_positions]}
