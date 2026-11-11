"""Explicit periodic-image attention used for the efficiency comparison."""
import math
import types
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from flexmof.model.attention import PrecomputedLogAttentionWeights
from flexmof.model.periodic import iter_periodic_edge_slices, polynomial_cutoff


def _cat(parts):
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def install(net, backend='edge'):
    """Patch only this process; return a stats object. Never edits source files."""
    if backend not in ('edge', 'sdpa', 'sdpa_ckpt'): raise ValueError(backend)
    encoder = net.structure_encoder
    emb = encoder.embedding
    original_sdpa = F.scaled_dot_product_attention
    original_net_forward = net.forward
    state = {'edge_data': None, 'attention_calls': 0, 'evaluations': 0,
             'last_edge_count': None, 'dropout_call_counts': {}}

    def image_attention(q, k, v, data, scale=None, dropout_p=0.0):
        eb, eq, es, logmass, central, group, target, n = data
        batch, heads, targets, width = q.shape
        assert targets == n and k.shape[:3] == (batch, heads, n)
        assert v.shape[:3] == (batch, heads, n) and heads == logmass.shape[1]
        qk = torch.matmul(q, k.transpose(-1, -2)) * (scale if scale is not None else 1.0 / math.sqrt(width))
        c_score = qk + central
        e_score = qk[eb, :, eq, es] + logmass
        maxima = c_score.detach().amax(dim=-1).reshape(-1).clone()
        maxima.scatter_reduce_(0, group, e_score.detach().reshape(-1), reduce='amax', include_self=True)
        c_exp = torch.exp(c_score - maxima.view(batch, heads, n, 1))
        e_exp = torch.exp(e_score.reshape(-1) - maxima[group])
        denom = c_exp.sum(dim=-1).reshape(-1).scatter_add(0, group, e_exp)
        assert torch.isfinite(denom).all() and bool((denom > 0).all())
        c_prob = c_exp / denom.view(batch, heads, n, 1)
        e_prob = e_exp / denom[group]
        probabilities = c_prob.reshape(-1).scatter_add(0, target, e_prob).view(batch, heads, n, n)
        if dropout_p:
            probabilities = F.dropout(probabilities, p=dropout_p, training=True)
        return torch.matmul(probabilities, v)

    def image_attention_sdpa(q, k, v, data, scale=None, dropout_p=0.0):
        sources, bias, n = data
        batch, heads, targets, width = q.shape
        assert targets == n and k.shape[:3] == (batch, heads, n)
        assert v.shape[:3] == (batch, heads, n) and heads == bias.shape[-1]
        # Bound expanded K/V temporaries; use the original optimized PyTorch
        # SDPA on each query-specific expanded image sequence.
        queries_per_chunk = 4 if width <= 32 else 2
        chunks = []
        flat_q = q.permute(0, 2, 1, 3).reshape(batch*n, heads, width)
        # One dropout draw per (batch, head, query, source), shared by every
        # periodic image of that source, as in source-reduced attention.
        source_dropout = None
        if dropout_p:
            source_dropout = F.dropout(torch.ones((batch, heads, n, n), device=q.device, dtype=q.dtype),
                                       p=dropout_p, training=True).permute(0, 2, 1, 3).reshape(batch*n, heads, n)
        for begin in range(0, batch*n, queries_per_chunk):
            end = min(begin + queries_per_chunk, batch*n)
            count = end - begin
            source_chunk = sources[begin:end]
            bias_chunk = bias[begin:end]
            batch_chunk = torch.arange(begin, end, device=q.device) // n
            dropout_chunk = source_dropout[begin:end] if source_dropout is not None else q.new_empty(0)
            qg = flat_q[begin:end].unsqueeze(2).contiguous()
            def evaluate(qpart, kfull, vfull, biaspart, sourcepart, batchpart, drop_part):
                indices = sourcepart[:, None, :, None].expand(-1, heads, -1, kfull.size(-1))
                kg = kfull.index_select(0, batchpart).gather(2, indices).contiguous()
                vg = vfull.index_select(0, batchpart).gather(2, indices).contiguous()
                if drop_part.numel():
                    dindex = sourcepart[:, None, :].expand(-1, heads, -1)
                    vg = vg * drop_part.gather(2, dindex).unsqueeze(-1)
                mask = biaspart.permute(0, 2, 1).unsqueeze(2).contiguous()
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    return original_sdpa(qpart, kg, vg, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale)
            if backend == 'sdpa_ckpt' and torch.is_grad_enabled():
                out = checkpoint(evaluate, qg, k, v, bias_chunk, source_chunk, batch_chunk, dropout_chunk, use_reentrant=False)
            else:
                out = evaluate(qg, k, v, bias_chunk, source_chunk, batch_chunk, dropout_chunk)
            chunks.append(out.squeeze(2))
        return torch.cat(chunks, dim=0).reshape(batch, n, heads, width).permute(0, 2, 1, 3).contiguous()

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        payload = state['edge_data']
        if payload is None or attn_mask is not payload['sentinel']:
            return original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        if not 0.0 <= dropout_p < 1.0 or is_causal or enable_gqa:
            raise ValueError('Expanded benchmark requires noncausal attention, dropout in [0,1), and no GQA')
        state['attention_calls'] += 1
        key = str(float(dropout_p))
        state['dropout_call_counts'][key] = state['dropout_call_counts'].get(key, 0) + 1
        fn = image_attention if backend == 'edge' else image_attention_sdpa
        return fn(q, k, v, payload['values'], scale=scale, dropout_p=dropout_p)

    def encoder_forward(self, batch, invariant, time_embedding, padding_mask,
                        pair_features, periodic, clean_mask):
        if not periodic.get('streaming_per_atom') or not periodic.get('central_cell_graph'):
            raise ValueError('Expected frozen E370 streaming/central-cell graph')
        if bool(padding_mask.any()):
            raise ValueError('Only batches without padded tokens are supported')
        invariant = invariant.transpose(0, 1)
        pair = self.pair_feat_bias(pair_features).permute(0, 3, 1, 2)
        position_embedding = emb.embed(batch['masked_token_type'])
        bsz, n = padding_mask.shape
        bond = None
        if not emb.config.mof_disable_graphormer_2d:
            bond = emb.graph_2d_attention_bias(batch, periodic)
            emb._set_degrees(batch, periodic)
        cur_bias, cur_features, cur_dist = emb.pos_embedding_bias.central_terms(batch, padding_mask)
        init_bias, init_features, _ = emb.init_pos_embedding_bias.central_terms(batch, padding_mask)
        init_present = (batch['init_pos'] != 0.0).any(dim=-1).any(dim=-1)
        init_bias = init_bias.masked_fill(~init_present[:, None, None, None], 0.0)
        local = polynomial_cutoff(cur_dist, periodic['cutoff']).to(cur_features.dtype)
        local = local.masked_fill(padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(1), 0.0)
        current_sum = (cur_features * local[..., None]).sum(dim=2)
        initial_sum = (init_features * local[..., None]).sum(dim=2)
        central_logits = cur_bias + init_bias + pair
        if bond is not None: central_logits = central_logits + bond['central']
        central_logits = central_logits.masked_fill(padding_mask[:, None, None, :], float('-inf'))
        central_logits = central_logits.masked_fill(padding_mask[:, None, :, None], 0.0)
        central = central_logits + local.clamp_min(1e-30).log()[:, None]
        central = central.masked_fill(local[:, None] <= 0, float('-inf'))

        batches, queries, sources, masses = [], [], [], []
        for offset, edges in iter_periodic_edge_slices(batch['pos'], batch['masked_token_type'], periodic):
            def compute(edge_chunk, edge_offset=offset):
                eb, eq, _, es, geom, cf, inf, cutoff = emb._streaming_edge_terms(
                    batch, periodic, init_present, edge_offset, edge_chunk)
                extra = torch.zeros_like(geom) if bond is None else bond['disconnected'][None].expand_as(geom)
                extra = extra.masked_fill(clean_mask[eb, eq, None], 0.0)
                lm = geom + pair[eb, :, eq, es] + extra + cutoff.clamp_min(1e-30).log()[:, None]
                lm = lm.masked_fill(cutoff[:, None] <= 0, float('-inf'))
                index = (eb * n + eq)[:, None].expand(-1, cf.size(-1))
                c = torch.zeros(bsz * n, cf.size(-1), device=cf.device, dtype=cf.dtype).scatter_add(
                    0, index, (cf * cutoff[:, None]).to(cf.dtype)).view(bsz, n, -1)
                j = torch.zeros_like(c).view(bsz * n, -1).scatter_add(
                    0, index, (inf * cutoff[:, None] * init_present[eb, None]).to(cf.dtype)).view_as(c)
                return eb, eq, es, lm, c, j
            if self.training and torch.is_grad_enabled() and periodic['checkpoint_edge_activations']:
                eb, eq, es, lm, ca, ia = checkpoint(compute, edges, use_reentrant=False)
            else:
                eb, eq, es, lm, ca, ia = compute(edges)
            batches.append(eb); queries.append(eq); sources.append(es); masses.append(lm)
            current_sum = current_sum + ca
            initial_sum = initial_sum + ia
        if not batches: raise ValueError('Empty periodic edge set is not yet supported')
        eb, eq, es, logmass = map(_cat, (batches, queries, sources, masses))
        heads = self.num_heads
        if backend == 'edge':
            head_ids = torch.arange(heads, device=logmass.device)[None, :]
            group = ((eb[:, None] * heads + head_ids) * n + eq[:, None]).reshape(-1)
            target = (group.view(-1, heads) * n + es[:, None]).reshape(-1)
            payload = (eb, eq, es, logmass, central, group, target, n)
        else:
            global_query = eb * n + eq
            counts = torch.bincount(global_query, minlength=bsz*n)
            max_images = int(counts.max().item())
            starts = torch.cumsum(counts, dim=0) - counts
            order = torch.argsort(global_query, stable=True)
            ordered_query, ordered_source = global_query[order], es[order]
            ranks = torch.arange(global_query.numel(), device=eq.device) - starts[ordered_query]
            destination = ordered_query * max_images + ranks
            image_sources = torch.zeros(bsz * n * max_images, dtype=es.dtype, device=es.device)
            image_sources = image_sources.scatter(0, destination, ordered_source).view(bsz*n, max_images)
            image_bias = torch.full((bsz * n * max_images, heads), float('-inf'),
                                    dtype=logmass.dtype, device=logmass.device)
            image_bias = image_bias.index_copy(0, destination, logmass[order]).view(bsz*n, max_images, heads)
            central_sources = torch.arange(n, device=es.device).expand(bsz*n, -1)
            all_sources = torch.cat([central_sources, image_sources], dim=1)
            all_bias = torch.cat([central.permute(0, 2, 3, 1).reshape(bsz*n, n, heads), image_bias], dim=1)
            payload = (all_sources, all_bias, n)
        sentinel = torch.empty(0, device=logmass.device, dtype=logmass.dtype)
        state['edge_data'] = {'sentinel': sentinel, 'values': payload}
        state['evaluations'] += 1
        state['last_edge_count'] = int(eb.numel())
        position = emb.pos_embedding_bias.pos_embedding_proj(current_sum)
        initial = emb.init_pos_embedding_bias.pos_embedding_proj(initial_sum)
        position += initial * init_present[:, None, None]
        position_embedding += position.masked_fill(padding_mask[..., None], 0.0)
        weights = PrecomputedLogAttentionWeights(sentinel)
        for layer in self.layers:
            position_embedding = layer(position_embedding + time_embedding, invariant, weights)
        return position_embedding, central_logits, weights

    def net_forward(*args, **kwargs):
        try:
            return original_net_forward(*args, **kwargs)
        finally:
            state['edge_data'] = None

    encoder.forward = types.MethodType(encoder_forward, encoder)
    net.forward = net_forward
    F.scaled_dot_product_attention = sdpa
    return state
