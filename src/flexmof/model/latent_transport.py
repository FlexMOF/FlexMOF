"""Permutation optimal-transport coupling for equivariant graph latent flows."""

import torch


@torch.no_grad()
def align_atom_noise(clean, noise, mask):
    """Minimize each block's clean/noise distance by reordering noise rows.

    The unordered Gaussian point set, every clean target, and all block/MOF
    boundaries are preserved. This changes only training path coupling; it
    introduces no inference conditions, labels, graph edits or noise filtering.
    """
    from scipy.optimize import linear_sum_assignment

    if clean.shape != noise.shape or clean.shape[:2] != mask.shape:
        raise ValueError('Clean latents, noise and mask must align')
    with torch.autocast(clean.device.type, enabled=False):
        costs = torch.cdist(clean.float(), noise.float()).square().cpu()
    masks = mask.cpu()
    permutation = torch.arange(noise.size(1)).expand(noise.size(0), -1).clone()
    for block in range(noise.size(0)):
        positions = masks[block].nonzero().flatten()
        if len(positions) > 1:
            rows, columns = linear_sum_assignment(costs[block][positions][:, positions].numpy())
            permutation[block, positions[rows]] = positions[columns]
    indices = permutation.to(noise.device)[..., None].expand_as(noise)
    return noise.gather(1, indices).masked_fill(~mask[..., None], 0)


@torch.no_grad()
def align_mof_noise(clean, noise, batch, group_size):
    """Match whole noise samples only among identical shape conditions.

    A sliced point-set distance supplies the assignment cost. This first
    assigns complete MOF noise tensors; atom matching can then align rows.
    The noise inventory is unchanged within each conditional minibatch.
    """
    from scipy.optimize import linear_sum_assignment

    counts = batch['block_mask'].sum(-1).cpu()
    if group_size < 2:
        raise ValueError('Conditional transport group size must be at least two')
    labels = batch.get('transport_group')
    if labels is None:
        if len(counts) % group_size:
            raise ValueError('Complete conditional transport groups are required')
        labels = torch.arange(len(counts))//group_size
    else:
        labels = labels.cpu()
        if labels.shape != counts.shape:
            raise ValueError('One transport-group label is required per MOF')
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    permutation = torch.arange(noise.size(0))
    with torch.autocast(clean.device.type, enabled=False):
        for label in labels.unique(sorted=True):
            members = (labels == label).nonzero().flatten()
            size = len(members)
            if size > group_size:
                raise ValueError('Conditional group exceeds the configured maximum')
            if size == 1:
                continue
            blocks = int(counts[members[0]])
            if not (counts[members] == blocks).all():
                raise ValueError('Transport groups must share one complete shape condition')
            indices = offsets[members, None]+torch.arange(blocks)[None]
            packed_indices = indices.flatten().to(noise.device)
            masks = batch['mask'][packed_indices].reshape(size, blocks, -1)
            for values in (masks, batch['repetitions'][packed_indices].reshape(size, blocks),
                           batch['is_linker'][packed_indices].reshape(size, blocks)):
                if not torch.equal(values, values[:1].expand_as(values)):
                    raise ValueError('Transport groups must share sizes, repetitions and roles')
            targets = clean[packed_indices].float().reshape(size, blocks, clean.size(1), clean.size(2))
            sources = noise[packed_indices].float().reshape_as(targets)
            cost = clean.new_zeros(size, size, dtype=torch.float32)
            for block in range(blocks):
                mask = masks[0, block]
                # Sort each latent coordinate separately: a fixed set of sliced
                # Wasserstein projections, invariant to atom-row order.
                target = targets[:, block, mask].sort(dim=1).values.flatten(1)
                source = sources[:, block, mask].sort(dim=1).values.flatten(1)
                cost += torch.cdist(target, source).square()/(target.size(1)*blocks)
            rows, columns = linear_sum_assignment(cost.cpu().numpy())
            for row, column in zip(rows, columns):
                permutation[indices[int(row)]] = indices[int(column)]
    return noise[permutation.to(noise.device)].masked_fill(~batch['mask'][..., None], 0)
