"""Strict two-stage generation: composition first, then coordinates.

Stage one runs the latent reverse process to completion and decodes once, at
the end. Stage two receives nothing but the decoded graph: atomic numbers,
bonds and block multiplicities. No coordinate, lattice or reference-composition
tensor crosses the boundary, and the composition is frozen once stage two
starts. The handoff is shared by the fused model and sampling scripts.
"""

from __future__ import annotations

import torch

from .data import collate_mofs
from .model.gaussian_composition import AtomVocabulary

LATTICE_TOKENS = 8
LATTICE_TOKEN_ID = 128


def blocks_from_stage_one(expanded, packed):
    """Group stage one's per-block graphs by the MOF that requested them."""
    atoms = expanded["atomic_numbers"]
    bonds = expanded["bond_adj"]
    mask = expanded["atom_mask"].bool()
    owner = packed["owner"]
    repetitions = packed["repetitions"]
    roles = packed["is_linker"]
    mofs = packed["block_mask"].size(0)

    grouped = [[] for _ in range(mofs)]
    for row in range(atoms.size(0)):
        present = mask[row].nonzero().flatten()
        local = bonds[row][present][:, present]
        edges = torch.triu(local > 0, diagonal=1).nonzero().tolist()
        grouped[int(owner[row])].append(
            {
                "atomic_numbers": atoms[row][present].tolist(),
                "bonds": edges,
                "repetitions": int(repetitions[row]),
                "is_linker": bool(roles[row]),
            }
        )
    return grouped


def composition_from_blocks(blocks):
    """Lay a MOF's blocks out into one cell-level graph."""
    numbers, edges, placed = [], [], []
    for block in blocks:
        offsets = []
        for _ in range(block["repetitions"]):
            offset = len(numbers)
            offsets.append(offset)
            numbers.extend(block["atomic_numbers"])
            edges.extend(
                [first + offset, second + offset] for first, second in block["bonds"]
            )
        placed.append({**block, "cell_offsets": offsets})
    return {
        "numbers": torch.tensor(numbers, dtype=torch.long),
        "edge_index": torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        "blocks": placed,
    }


def csp_item(composition, index):
    """A CSP input built from a generated composition and nothing else.

    Coordinates and the cell are zero: the sampler initializes both. Keeping
    them zero here is what makes the leak impossible rather than unlikely.
    """
    numbers = composition["numbers"]
    edges = composition["edge_index"]
    tokens = torch.cat(
        [numbers, torch.full((LATTICE_TOKENS,), LATTICE_TOKEN_ID, dtype=torch.long)]
    )
    nodes = len(tokens)
    adjacency = torch.ones(nodes, nodes, dtype=torch.bool)
    return {
        "idx": index,
        "token_type": tokens,
        "num_atoms": len(numbers),
        "coords": torch.zeros(nodes, 3, dtype=torch.float64),
        "cell": torch.zeros(3, 3, dtype=torch.float64),
        "pbc": torch.ones(3, dtype=torch.bool),
        "node_attr": torch.cat(
            [tokens[:, None], torch.zeros(nodes, LATTICE_TOKENS, dtype=torch.long)],
            dim=-1,
        ),
        "adj": adjacency,
        "attn_bias": torch.zeros(nodes + 1, nodes + 1),
        "in_degree": adjacency.long().sum(1),
        "edge_index": edges,
        "edge_attr": torch.ones(edges.shape[1], 1, dtype=torch.long),
        "target_cell_index": torch.zeros(edges.shape[1], 3, dtype=torch.long),
    }


def _composition_fields(blocks, vocabulary):
    """Rebuild the ``mof_bb_*`` view of a generated composition.

    Atom labels are vocabulary ids, matching what ``MOFDataset`` emits under a
    latent strategy, so the joint encoder folds a generated graph exactly as it
    folds a corpus one.
    """
    sizes = [len(block["atomic_numbers"]) for block in blocks]
    width = max(sizes) if sizes else 1
    count = len(blocks)
    repetitions = max((block["repetitions"] for block in blocks), default=1)
    atoms = torch.zeros(count, width, dtype=torch.long)
    atom_mask = torch.zeros(count, width, dtype=torch.bool)
    bonds = torch.zeros(count, width, width, dtype=torch.long)
    occurrence = torch.full((count, repetitions, width), -1, dtype=torch.long)
    occurrence_mask = torch.zeros(count, repetitions, dtype=torch.bool)
    for index, block in enumerate(blocks):
        size = sizes[index]
        atoms[index, :size] = vocabulary.encode(
            torch.tensor(block["atomic_numbers"], dtype=torch.long)
        )
        atom_mask[index, :size] = True
        for first, second in block["bonds"]:
            bonds[index, first, second] = 1
            bonds[index, second, first] = 1
        for repeat, offset in enumerate(block["cell_offsets"]):
            occurrence[index, repeat, :size] = torch.arange(offset, offset + size)
            occurrence_mask[index, repeat] = True
    total = sum(size * block["repetitions"] for size, block in zip(sizes, blocks))
    return {
        "mof_bb_atom_types": atoms,
        "mof_bb_atom_mask": atom_mask,
        "mof_bb_bond_adj": bonds,
        "mof_bb_occ_atom_index": occurrence,
        "mof_bb_occ_mask": occurrence_mask,
        "mof_bb_mask": torch.ones(count, dtype=torch.bool),
        "mof_bb_repetitions": torch.tensor(
            [block["repetitions"] for block in blocks], dtype=torch.long
        ),
        "mof_bb_is_linker": torch.tensor(
            [block["is_linker"] for block in blocks], dtype=torch.bool
        ),
        "mof_total_num_atoms": torch.tensor(total),
        "mof_num_unique_bbs": torch.tensor(count),
        "mof_has_composition_targets": torch.tensor(True),
        # Already vocabulary ids, so the collate must not shift them again.
        "mof_composition_remapped": True,
    }


def stage_two_batch(grouped, device, vocabulary, *, start_index=0):
    """Collate generated compositions into a batch the CSP can denoise."""
    items = []
    for offset, blocks in enumerate(grouped):
        composition = composition_from_blocks(blocks)
        item = csp_item(composition, start_index + offset)
        item.update(_composition_fields(composition["blocks"], vocabulary))
        items.append(item)
    batch = collate_mofs(items)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def generate(sampler, batch, *, start_index=0, **stage_one):
    """Run both stages and return the CSP result plus the generated graphs."""
    network = sampler.net
    if not getattr(network, "joint_composition", False):
        raise RuntimeError("Two-stage generation needs a joint composition model")
    stage_one_output = network.mof_joint.generate_composition(batch, **stage_one)
    grouped = blocks_from_stage_one(stage_one_output, stage_one_output["packed"])
    stage_two = stage_two_batch(
        grouped,
        batch["token_id"].device,
        AtomVocabulary(network.config.mof_composition_atom_vocab),
        start_index=start_index,
    )
    result = sampler.sample(stage_two)
    return {
        "blocks": grouped,
        "stage_one": stage_one_output,
        "stage_two_batch": stage_two,
        **result,
    }


def generate_unconditional(sampler, metadata, *, start_index=0, **stage_one):
    """Learned metadata -> LG-Flow -> CSP, without a reference MOF."""
    from .metadata_prior import metadata_request, validate_metadata, MetadataConfig

    network = sampler.net
    if not getattr(network, "joint_composition", False):
        raise RuntimeError("Unconditional generation needs a joint composition checkpoint")
    for row in metadata:
        validate_metadata(row, MetadataConfig(max_blocks=32))
    device = next(network.parameters()).device
    request = metadata_request(metadata, device)
    stage_one_output = network.mof_joint.generate_request(request, **stage_one)
    grouped = blocks_from_stage_one(stage_one_output, stage_one_output["packed"])
    for row, blocks in zip(metadata, grouped, strict=True):
        if len(blocks) != row["num_blocks"]:
            raise ValueError("Decoded block count differs from sampled metadata")
    batch = stage_two_batch(grouped, device, AtomVocabulary(network.config.mof_composition_atom_vocab),
                            start_index=start_index)
    result = sampler.sample(batch)
    return {"metadata": metadata, "blocks": grouped, "stage_one": stage_one_output,
            "stage_two_batch": batch, **result}
