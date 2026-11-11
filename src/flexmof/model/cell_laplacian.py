"""Cell-level Laplacian positional encoding in front of the sequence trunk.

This is *not* the autoencoder's spectral input. Those two are easy to confuse
and must not be:

* The autoencoder's ``spectral``/``spectral_mask`` are eigenpairs of a single
  **building block's** bond graph. They are a required encoder input -- the
  codec was trained with them and cannot be evaluated without them -- and are
  never switched off, in any mode.
* What this module produces are eigenvectors of the **whole unit cell's** bond
  graph, added to the CSP node features. It is new in the fused model, it is
  optional, and regression against the pre-fusion checkpoints runs with it
  disabled. Its parameters live under ``cell_laplacian.`` so they cannot
  collide with anything an older checkpoint contains.

Eigenvector signs are arbitrary, so training randomly flips them and inference
does not, the same convention the autoencoder uses for its own spectral input.
"""

from __future__ import annotations

import torch
from torch import nn


def cell_laplacian_features(edge_index, num_atoms, frequencies, *, padded_atoms=None):
    """Lowest non-trivial Laplacian eigenvectors of each MOF's bond graph.

    ``edge_index`` is ``[B, 2, E]`` with ``-1`` padding, matching the batch's
    bond edges. Solved per MOF on the CPU so repeated eigenvalues get a stable
    basis, as in the corpus preparation.
    """
    mofs = edge_index.shape[0]
    atoms = padded_atoms if padded_atoms is not None else int(num_atoms.max())
    device = edge_index.device
    features = torch.zeros(mofs, atoms, frequencies, device=device)
    mask = torch.zeros(mofs, frequencies, dtype=torch.bool, device=device)
    edges_cpu = edge_index.cpu()
    counts = num_atoms.cpu()
    for index in range(mofs):
        count = int(counts[index])
        if count < 2:
            continue
        edges = edges_cpu[index]
        valid = (edges[0] >= 0) & (edges[1] >= 0) & (edges[0] < count) & (edges[1] < count)
        adjacency = torch.zeros(count, count, dtype=torch.float64)
        if bool(valid.any()):
            rows, columns = edges[0][valid], edges[1][valid]
            adjacency[rows, columns] = 1.0
            adjacency[columns, rows] = 1.0
        laplacian = torch.diag(adjacency.sum(-1)) - adjacency
        vectors = torch.linalg.eigh(laplacian).eigenvectors
        # Drop the constant zero mode; it carries no positional information.
        available = min(count - 1, frequencies)
        if available < 1:
            continue
        features[index, :count, :available] = (
            vectors[:, 1 : 1 + available].float().to(device)
        )
        mask[index, :available] = True
    return features, mask


class CellLaplanEncodingError(ValueError):
    pass


class CellLaplacianEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frequencies = config.mof_joint_csp_laplacian_frequencies
        if self.frequencies < 1:
            raise CellLaplanEncodingError(
                "Cell Laplacian encoding needs at least one frequency"
            )
        self.project = nn.Sequential(
            nn.Linear(self.frequencies, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=False),
        )
        # Starts as an exact no-op, so turning the encoding on does not by
        # itself move a trunk that was trained without it.
        nn.init.zeros_(self.project[-1].weight)

    def forward(self, batch, padding_mask):
        atoms = padding_mask.shape[1]
        cached = batch.get("mof_cell_laplacian")
        cached_mask = batch.get("mof_cell_laplacian_mask")
        if cached is not None and cached_mask is not None:
            features = cached[:, :atoms, : self.frequencies]
            mask = cached_mask[:, : self.frequencies]
        else:
            features, mask = cell_laplacian_features(
                batch["edge_index"],
                batch["num_atoms"],
                self.frequencies,
                padded_atoms=atoms,
            )
        features = features.to(padding_mask.device).clone()
        if self.training:
            signs = (
                torch.randint(
                    0, 2, (features.size(0), 1, self.frequencies), device=features.device
                )
                * 2
                - 1
            )
            features = features * signs
        features = features * mask[:, None, :].to(features.dtype)
        projected = self.project(features)
        return projected.masked_fill(padding_mask[..., None], 0)
