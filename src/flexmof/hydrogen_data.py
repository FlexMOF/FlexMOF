"""Lossless graph representation with terminal hydrogens stored as node counts.

Only a degree-one H attached to a non-H node is collapsed. Other hydrogens
remain ordinary core nodes; no exceptional source graph is filtered or repaired.
Expansion is the representation's decoder, not a modification of a sampled
explicit-H graph. Core labels, core bonds and attached-H counts are generated.
"""

import torch

from .linker_data import collate_linkers


REPRESENTATION = 'terminal_h_counts_v1'


def factor_terminal_hydrogens(numbers, adjacency):
    numbers, adjacency = numbers.detach().cpu(), adjacency.detach().cpu()
    n = len(numbers)
    if n < 1 or adjacency.shape != (n, n) or not torch.equal(adjacency, adjacency.T):
        raise ValueError('Factorization requires a nonempty undirected graph')
    if adjacency.diag().any() or not ((adjacency == 0) | (adjacency == 1)).all():
        raise ValueError('Factorization requires binary bonds without self loops')
    degree = adjacency.sum(-1)
    parent = adjacency.long().argmax(-1)
    collapsed = (numbers == 1) & (degree == 1) & (numbers[parent] != 1)
    core = (~collapsed).nonzero().flatten()
    old_to_core = torch.full((n,), -1, dtype=torch.long)
    old_to_core[core] = torch.arange(len(core))
    hydrogen = collapsed.nonzero().flatten()
    parent_core = old_to_core[parent[hydrogen]]
    counts = torch.bincount(parent_core, minlength=len(core))
    hydrogen = hydrogen[torch.argsort(parent_core, stable=True)]
    return dict(numbers=numbers[core].clone(), adjacency=adjacency[core][:, core].clone(),
                hydrogen_counts=counts, original_order=torch.cat([core, hydrogen]))


def expand_terminal_hydrogens(numbers, adjacency, hydrogen_counts):
    """Expand every predicted count; no chemistry checks, rejection or repair."""
    numbers, adjacency = numbers.detach().cpu(), adjacency.detach().cpu()
    counts = hydrogen_counts.detach().cpu()
    n = len(numbers)
    if n < 1 or adjacency.shape != (n, n) or counts.shape != (n,):
        raise ValueError('Core labels, bonds and hydrogen counts must align')
    if (counts < 0).any() or not torch.equal(counts, counts.long()):
        raise ValueError('Hydrogen counts must be nonnegative integers')
    counts = counts.long()
    total = n+int(counts.sum())
    expanded_numbers = torch.cat([numbers, torch.ones(total-n, dtype=numbers.dtype)])
    expanded_adjacency = adjacency.new_zeros(total, total)
    expanded_adjacency[:n, :n] = adjacency
    parents = torch.arange(n).repeat_interleave(counts)
    children = torch.arange(n, total)
    expanded_adjacency[parents, children] = expanded_adjacency[children, parents] = 1
    return dict(numbers=expanded_numbers, adjacency=expanded_adjacency)


def collate_hydrogen_graphs(records):
    batch = collate_linkers(records)
    counts = torch.zeros_like(batch['atoms'])
    for row, record in enumerate(records):
        value = record['hydrogen_counts']
        if value.shape != record['numbers'].shape:
            raise ValueError('Attached-H counts must align with core atoms')
        counts[row, :len(value)] = value
    return dict(batch, hydrogen_counts=counts)


def assemble_hydrogen_composition(shape, core_graphs):
    """Expand generated linker attributes into the ordinary explicit-atom CSP input.

    Non-linker blocks are singleton atom identities by definition; an attached-H
    attribute is only defined for linker blocks. No source H counts are read.
    """
    from .joint_data import assemble_joint_graphs
    if len(core_graphs) != len(shape['sizes']):
        raise ValueError('Generated core blocks must match the supplied shape')
    expanded, all_counts = [], []
    for size, role, graph in zip(shape['sizes'], shape['is_linker'], core_graphs):
        if len(graph['numbers']) != size:
            raise ValueError('Generated core atom count differs from its shape')
        counts = graph['hydrogen_counts'] if role else torch.zeros(size, dtype=torch.long)
        expanded.append(expand_terminal_hydrogens(graph['numbers'], graph['adjacency'], counts))
        all_counts.append(counts.tolist())
    expanded_shape = dict(shape, sizes=[len(graph['numbers']) for graph in expanded])
    composition = assemble_joint_graphs(expanded_shape, expanded)
    composition.update(representation=REPRESENTATION, generated_hydrogen_counts=all_counts,
                       expanded_block_sizes=expanded_shape['sizes'])
    return composition, expanded


def verify_hydrogen_expansion(composition):
    """Check that recorded generated attributes explain the complete CSP graph."""
    try:
        shape = composition['shape_condition']
        counts = composition['generated_hydrogen_counts']
        blocks = composition['blocks']
        if not (len(blocks) == len(counts) == len(shape['sizes']) == len(shape['is_linker']) == len(shape['repetitions'])):
            return False
        cores = []
        for size, role, block, hydrogen in zip(shape['sizes'], shape['is_linker'], blocks, counts):
            if len(hydrogen) != size or (not role and any(hydrogen)):
                return False
            labels = torch.tensor(block['atomic_numbers'])
            adjacency = torch.zeros(len(labels), len(labels), dtype=torch.long)
            for first, second in block['bonds']:
                adjacency[first, second] = adjacency[second, first] = 1
            restored = expand_terminal_hydrogens(labels[:size], adjacency[:size, :size], torch.tensor(hydrogen))
            if not torch.equal(restored['numbers'], labels) or not torch.equal(restored['adjacency'], adjacency):
                return False
            cores.append(dict(numbers=labels[:size], adjacency=adjacency[:size, :size], hydrogen_counts=torch.tensor(hydrogen)))
        expected, _ = assemble_hydrogen_composition(shape, cores)
        return (composition['representation'] == REPRESENTATION
                and composition['expanded_block_sizes'] == expected['expanded_block_sizes']
                and torch.equal(composition['numbers'], expected['numbers'])
                and torch.equal(composition['edge_index'], expected['edge_index']))
    except (KeyError, ValueError, IndexError, RuntimeError, TypeError):
        return False


def verify_hydrogen_partitions(corpus, original):
    """Audit every codec graph identity against the original MOF partitions."""
    source = {split: {i for mof in original['mofs'] if mof['split'] == split for i in mof['graph_ids']
                      if original['graphs'][i]['is_linker']} for split in ('train', 'valid', 'test')}
    expected = dict(train=source['train'], valid=source['valid']-source['train'],
                    test=source['test']-source['train']-source['valid'])
    ids = [g['source_graph_id'] for g in corpus['graphs']]
    if len(ids) != len(set(ids)):
        return False
    return all({g['source_graph_id'] for g in corpus['graphs'] if g['split'] == split} == expected[split]
               for split in expected) and set(ids) == set.union(*source.values())
