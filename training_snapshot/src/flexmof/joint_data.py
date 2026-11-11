"""Pack whole-MOF block sets without embedding arbitrary block or atom indices."""

import torch


def collate_joint(records, graphs=None, *, device='cpu'):
    """Inference needs only sizes/repetitions/roles, never graph IDs or targets."""
    sizes = [size for record in records for size in record['sizes']]
    maximum = max(sizes)
    block_count = max(len(record['sizes']) for record in records)
    mask = torch.arange(maximum)[None] < torch.tensor(sizes)[:, None]
    owner, slot, repetitions, roles = [], [], [], []
    block_mask = torch.zeros(len(records), block_count, dtype=torch.bool)
    counts = []
    atom_owner, atom_slot, compact_sizes = [], [], []
    for i, record in enumerate(records):
        n = len(record['sizes'])
        if not (n == len(record['repetitions']) == len(record['is_linker'])):
            raise ValueError('Block sizes, multiplicities, and roles must align')
        compact = sum(record["sizes"])
        compact_sizes.append(compact)
        atom_owner.extend([i]*compact)
        atom_slot.extend(range(compact))
        owner.extend([i]*n)
        slot.extend(range(n))
        repetitions.extend(record['repetitions'])
        roles.extend(record['is_linker'])
        block_mask[i, :n] = True
        counts.append([n, sum(s*r for s, r in zip(record['sizes'], record['repetitions'])), sum(record['repetitions'])])
    batch = dict(mask=mask, owner=torch.tensor(owner), slot=torch.tensor(slot),
                 repetitions=torch.tensor(repetitions), is_linker=torch.tensor(roles, dtype=torch.bool),
                 block_mask=block_mask, mof_counts=torch.tensor(counts),
                 atom_owner=torch.tensor(atom_owner), atom_slot=torch.tensor(atom_slot),
                 joint_atom_mask=torch.arange(max(compact_sizes))[None] < torch.tensor(compact_sizes)[:, None])
    if any('_transport_group' in record for record in records):
        if not all('_transport_group' in record for record in records):
            raise ValueError('Conditional training batches need a group label for every MOF')
        batch['transport_group'] = torch.tensor([record['_transport_group'] for record in records])
    if graphs is not None:
        latents = [graphs[index]['latent'] for record in records for index in record['graph_ids']]
        if [len(latent) for latent in latents] != sizes:
            raise ValueError('Latent targets do not match their block sizes')
        batch['clean'] = torch.nn.utils.rnn.pad_sequence(latents, batch_first=True)
    return {key: value.to(device) for key, value in batch.items()}


def shape_record(record):
    return {key: list(record[key]) for key in ('sizes', 'repetitions', 'is_linker')}


def assemble_joint_graphs(record, graphs):
    """Expand generated block atoms and bonds into the composition format."""
    if len(graphs) != len(record['sizes']):
        raise ValueError('Generated block count differs from its shape condition')
    numbers, edges, blocks, inorganic = [], [], [], []
    for size, repetitions, linker, graph in zip(record['sizes'], record['repetitions'], record['is_linker'], graphs):
        labels = graph['numbers'].tolist()
        if len(labels) != size:
            raise ValueError('Generated graph size differs from its shape condition')
        if not linker and size != 1:
            raise ValueError('Inorganic blocks use the pretrained singleton representation')
        local_edges = torch.triu(graph['adjacency'], diagonal=1).nonzero().tolist()
        offsets = []
        for _ in range(repetitions):
            offset = len(numbers)
            offsets.append(offset)
            numbers.extend(labels)
            edges.extend([first+offset, second+offset] for first, second in local_edges)
        block = dict(atomic_numbers=labels, bonds=local_edges, repetitions=repetitions, cell_offsets=offsets, is_linker=linker)
        blocks.append(block)
        if not linker:
            inorganic.extend(labels*repetitions)
    return dict(numbers=torch.tensor(numbers, dtype=torch.long),
                edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                inorganic_numbers=inorganic, linkers=[b for b in blocks if b['is_linker']], blocks=blocks)
