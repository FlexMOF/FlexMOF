"""Inference-only experimental cache for an unchanged central chemical graph.

This is an isolated benchmark prototype, not a modification of the running job.
The underlying graph-bias implementation reads node count, central adjacency,
edge types, and expanded lengths.  Reuse requires exact equality of those
inputs, unchanged weights, evaluation mode, and a central streaming graph.
The periodic geometric neighbors are still rebuilt by the existing code.
"""
from __future__ import annotations
import torch


class CentralGraphBiasCache:
    INPUTS = ('all_adj', 'all_attn_edge_type', 'all_expand_len')

    def __init__(self, module):
        self.module = module
        self.original = module.forward
        self.saved = None
        self.calls = 0
        self.hits = 0
        self.rebuilds = 0
        self.fallbacks = 0
        self.invalidations = 0

    @staticmethod
    def copy_output(output):
        return {key: value.clone() for key, value in output.items()}

    def __call__(self, batch, periodic):
        self.calls += 1
        eligible = (not self.module.training and torch.is_inference_mode_enabled()
                    and not torch.is_autocast_enabled('cuda')
                    and periodic.get('streaming_per_atom', False)
                    and periodic.get('central_cell_graph', False)
                    and 'expand_len' not in periodic
                    and all(key in periodic for key in self.INPUTS))
        if not eligible:
            self.fallbacks += 1
            self.saved = None
            return self.original(batch, periodic)
        graph = tuple(periodic[key] for key in self.INPUTS)
        version = (self.module.num_heads, self.module.multi_hop_max_dist,
                   self.module.edge_hidden_dim, torch.are_deterministic_algorithms_enabled(),
                   torch.get_default_dtype(),
                   torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                   tuple((id(p), p._version, p.dtype, p.device) for p in self.module.parameters()),
                   tuple((id(b), b._version, b.dtype, b.device) for b in self.module.buffers()))
        width = batch['node_attr'].size(1)
        if self.saved is not None:
            same = (width == self.saved['width'] and version == self.saved['version']
                    and all(x.dtype == y.dtype and x.device == y.device and torch.equal(x, y)
                            for x, y in zip(graph, self.saved['graph'])))
            if same:
                self.hits += 1
                # Preserve the cached tensors if downstream code changes its output in place.
                return self.copy_output(self.saved['output'])
            self.invalidations += 1
        result = self.original(batch, periodic)
        if not (isinstance(result, dict) and all(isinstance(x, torch.Tensor) for x in result.values())):
            self.saved = None
            self.fallbacks += 1
            return result
        self.rebuilds += 1
        self.saved = {'width': width, 'version': version,
                      'graph': tuple(x.detach().clone() for x in graph),
                      'output': self.copy_output(result)}
        return result

    def stats(self):
        return {key: getattr(self, key) for key in
                ('calls', 'hits', 'rebuilds', 'fallbacks', 'invalidations')}
