"""Inference-only ODAC adapter with deterministic same-element atom mapping.

Some test NODE blocks list elements in a different order from the global cell
array. Their elemental multisets agree. Resolve each block-local atom to the
same-element global position by stable occurrence order, using no coordinates.
The resulting bond and occurrence maps are consistent with the cell atom order.
"""
from __future__ import annotations

import pickle

import numpy as np
import torch

from flexmof.model.gaussian_composition import AtomVocabulary
from odac_adapter import OdacDataset
from odac_multigraph import JointOdacDataset


def block_maps(record):
    numbers = torch.as_tensor(np.asarray(record.atomic_numbers), dtype=torch.long)
    maps, offset, changed = [], 0, 0
    for block in record.blocks:
        local = torch.as_tensor(np.asarray(block.atomic_numbers), dtype=torch.long)
        count = int(local.numel())
        global_slice = numbers[offset:offset + count]
        if count == 0 or global_slice.numel() != count:
            raise ValueError('invalid or empty block in test record')
        mapping = torch.full((count,), -1, dtype=torch.long)
        for z in torch.unique(local, sorted=True):
            src = torch.nonzero(local == z).flatten()
            dst = torch.nonzero(global_slice == z).flatten()
            if src.numel() != dst.numel():
                raise ValueError('block/cell elemental multisets disagree')
            mapping[src] = dst + offset
        if mapping.lt(0).any() or not torch.equal(numbers[mapping], local):
            raise ValueError('invalid stable element mapping')
        if not torch.equal(mapping, torch.arange(offset, offset + count)):
            changed += 1
        maps.append(mapping)
        offset += count
    if offset != numbers.numel():
        raise ValueError('blocks do not cover cell atoms')
    return numbers, maps, changed


class EvaluationJointOdacDataset(JointOdacDataset):
    def _bond_graph(self, record, cell, fractional):
        _, maps, _ = block_maps(record)
        edges = set()
        for block, mapping in zip(record.blocks, maps):
            bonds = np.asarray(block.bond_indices)
            if not self.drop_bonds and bonds.size:
                if bonds.ndim != 2 or bonds.shape[0] != 2:
                    raise ValueError('invalid block bond shape')
                for source, target in zip(bonds[0], bonds[1]):
                    a, b = int(mapping[int(source)]), int(mapping[int(target)])
                    if a != b:
                        edges.add((min(a, b), max(a, b)))
        if not edges:
            return torch.zeros([2, 0], dtype=torch.long), torch.zeros([0, 3], dtype=torch.long)
        ordered = sorted(edges)
        source_index = torch.tensor([a for a, _ in ordered], dtype=torch.long)
        target_index = torch.tensor([b for _, b in ordered], dtype=torch.long)
        normalized = fractional % 1.0 % 1.0
        cart = normalized.to(dtype=cell.dtype) @ cell
        image_offsets = self.neighbor_cells.to(dtype=cell.dtype) @ cell
        distances = (cart[source_index, None, :] -
                     (cart[target_index, None, :] + image_offsets[None, :, :])).norm(dim=-1, p=2)
        closest = distances.argmin(dim=-1)
        target_cell = torch.stack([closest // 9 - 1, (closest % 9) // 3 - 1, closest % 3 - 1], dim=-1).long()
        return torch.stack([source_index, target_index], dim=0), target_cell

    def __getitem__(self, index):
        # OdacDataset uses the overridden bond graph above; the original
        # JointOdacDataset.__getitem__ cannot handle NODE atom reordering.
        item = OdacDataset.__getitem__(self, index)
        record = pickle.loads(self._txn().get(self.keys[index]))
        numbers, maps, changed = block_maps(record)
        n = int(numbers.numel())
        ids = torch.zeros((n, 3), dtype=torch.float32)
        ids[:, 0] = (torch.arange(n, dtype=torch.float32) + 1) / (n + 1)
        ids[:, 1], ids[:, 2] = 0.25, 0.5
        blocks, roles = [], []
        for block, mapping in zip(record.blocks, maps):
            atomic = torch.as_tensor(np.asarray(block.atomic_numbers), dtype=torch.long)
            name = getattr(block.block_type, 'name', str(block.block_type))
            if name not in ('NODE', 'LINKER', 'SOLVENT', 'ADSORBATE'):
                raise ValueError(f'unknown block role: {name}')
            edges = torch.as_tensor(np.asarray(block.bond_indices), dtype=torch.long)
            if edges.numel() == 0:
                edges = torch.zeros((2, 0), dtype=torch.long)
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError('invalid block bond shape')
            if edges.numel() and (edges.min() < 0 or edges.max() >= atomic.numel()):
                raise ValueError('block bond index out of range')
            blocks.append({'atomic_numbers': atomic, 'edge_index': edges, 'frac_coords': ids[mapping]})
            roles.append(name != 'NODE')
        composition = self._composition_builder._composition_targets(
            {'pyg_mols': blocks, 'is_linker': roles}, numbers, ids)
        occurrences = composition['mof_bb_occ_atom_index']
        valid = (composition['mof_bb_occ_mask'].unsqueeze(-1) &
                 composition['mof_bb_atom_mask'].unsqueeze(1) & occurrences.ge(0))
        chosen = occurrences[valid]
        if chosen.numel() != n or not torch.equal(torch.sort(chosen).values, torch.arange(n)):
            raise ValueError('occurrence map does not cover every cell atom exactly once')
        decoded = AtomVocabulary(self.composition_config.mof_composition_atom_vocab).decode(
            composition['mof_bb_atom_types']).unsqueeze(1).expand_as(occurrences)[valid]
        if not torch.equal(decoded, numbers[chosen]):
            raise ValueError('occurrence mapping changes an atom element')
        item.update(composition)
        item['evaluation_reordered_blocks'] = changed
        return item
