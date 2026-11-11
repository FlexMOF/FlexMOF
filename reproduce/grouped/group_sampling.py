"""Sample mixed candidate IDs and commit per-candidate structures without clobbering."""
from __future__ import annotations
import math
from pathlib import Path
from graph_bias_cache import CentralGraphBiasCache
from group_seq_adapter import GroupedSeedSequenceReuse


def sample_and_match_grouped(sampler,batch,pairs,expected_steps,*,enable_sequence=True):
    import torch
    pairs=list(pairs)
    if not pairs or len(pairs)!=len(set(pairs)):
        raise ValueError('Expected unique (sample, candidate) pairs')
    if batch['idx'].tolist()!=[i for i,_ in pairs]:
        raise ValueError('Sampler row identities do not match planned pairs')
    if sampler.net.gaussian_composition or sampler.net.lgflow_composition or not sampler.net.joint_composition:
        raise ValueError('Mixed candidate output adapter is validated only for joint composition')
    directory=Path(sampler.config.output_dir)
    for sample,candidate in pairs:
        for kind in ('original','sampled'):
            path=directory/f'{kind}_{sample}_{candidate}.cif'
            if path.exists() or path.is_symlink():
                raise FileExistsError(f'Refusing to replace existing structure {path}')
    seq_module=sampler.net.seq_encoder
    original_seq=seq_module.forward
    seq=GroupedSeedSequenceReuse(seq_module,pairs)
    graph_module=sampler.net.structure_encoder.embedding.graph_2d_attention_bias
    original_graph=graph_module.forward
    graph=CentralGraphBiasCache(graph_module)
    original_positions=batch['pos'].clone()
    original_cell=batch['cell'].clone()
    if enable_sequence:
        seq_module.forward=seq
    graph_module.forward=graph
    try:
        with torch.inference_mode():
            batch['pos']=torch.zeros_like(batch['pos'])
            batch['cell']=torch.zeros_like(batch['cell'])
            sampled=sampler.sample(batch)
            structures=sampler.converter._convert(batch,batch['pos'],sampled.get('pred_mof_cell_atom_types'))
            references=sampler.converter._convert(batch,original_positions)
            if len(structures)!=len(pairs) or len(references)!=len(pairs):
                raise ValueError('Sampler returned the wrong number of structures')
            results=[sampler.converter._match(structure,reference,sample_id=sample,
                       sample_time_index=candidate)
                     for (sample,candidate),structure,reference in zip(pairs,structures,references)]
            for candidate in range(5):
                indices=[j for j,(_,c) in enumerate(pairs) if c==candidate]
                if not indices:continue
                index=torch.tensor(indices,dtype=torch.long,device=batch['pos'].device)
                subset={key:batch[key].index_select(0,index) for key in ('idx','num_atoms','pos','cell')}
                sampler._dump_metrics(subset,[results[j] for j in indices],candidate)
        if graph.calls!=expected_steps or graph.fallbacks or graph.rebuilds<1:
            raise RuntimeError('Graph bias cache did not execute for the complete schedule')
        if enable_sequence and seq.calls!=expected_steps:
            raise RuntimeError('Sequence reuse did not execute once per step')
        values=[]
        for row in results:
            converted={}
            for key,value in row.items():
                # The previous worker converted matcher values through a
                # float32 tensor before recording; preserve that convention.
                scalar=float(torch.tensor(value,dtype=torch.float32).item())
                converted[key]=scalar if math.isfinite(scalar) else None
            values.append(converted)
        return values,{'graph':graph.stats(),'sequence':seq.stats()}
    finally:
        batch['pos']=original_positions
        batch['cell']=original_cell
        seq_module.forward=original_seq
        graph_module.forward=original_graph
        graph.saved=None
