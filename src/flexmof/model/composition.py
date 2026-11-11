"""DiGress-style MOF composition diffusion from the original PSM path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GraphDiffusionInput:
    node_labels: torch.Tensor
    node_mask: torch.Tensor
    edge_labels: Optional[torch.Tensor]
    edge_mask: Optional[torch.Tensor]
    time_step: torch.Tensor
    alpha_bar: Optional[torch.Tensor] = None


@dataclass
class GraphDiffusionState:
    node_t: torch.Tensor
    edge_t: Optional[torch.Tensor]
    node_loss_mask: torch.Tensor
    edge_loss_mask: Optional[torch.Tensor]
    time_index: torch.Tensor
    alpha_bar: torch.Tensor


class DiGressUniformGraphDiffusion:
    """Categorical graph diffusion with a uniform atom and no-bond limit."""

    def __init__(
        self,
        num_node_classes: int,
        num_edge_classes: int,
        num_timesteps: int,
        valid_node_classes: list[int],
    ) -> None:
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.num_timesteps = num_timesteps
        self.valid_node_classes = valid_node_classes

    def q_sample(self, graph_input: GraphDiffusionInput) -> GraphDiffusionState:
        time_step = self._as_graph_time(graph_input.time_step)
        alpha_bar = self._alpha_bar(graph_input.alpha_bar, time_step).to(
            graph_input.node_labels.device
        )
        time_index = (time_step * (self.num_timesteps - 1)).round().long()
        time_index = time_index.clamp(min=0, max=self.num_timesteps - 1)

        node_probs = self._marginal_probs(
            graph_input.node_labels,
            self.num_node_classes,
            self._node_prior(graph_input.node_labels.device),
            alpha_bar,
        )
        node_t = self._sample_categorical(node_probs)
        node_t = torch.where(graph_input.node_mask, node_t, graph_input.node_labels)

        edge_t = None
        if graph_input.edge_labels is not None and graph_input.edge_mask is not None:
            edge_probs = self._marginal_probs(
                graph_input.edge_labels,
                self.num_edge_classes,
                self._edge_prior(graph_input.edge_labels.device),
                alpha_bar,
            )
            edge_t = self._sample_categorical(edge_probs)
            edge_t = torch.where(graph_input.edge_mask, edge_t, graph_input.edge_labels)
            edge_t = self._symmetrize_edges(edge_t, graph_input.edge_mask)

        return GraphDiffusionState(
            node_t=node_t,
            edge_t=edge_t,
            node_loss_mask=graph_input.node_mask,
            edge_loss_mask=graph_input.edge_mask,
            time_index=time_index,
            alpha_bar=alpha_bar,
        )

    def sample_prior(self, node_mask, edge_mask=None):
        node_shape = [*node_mask.shape, self.num_node_classes]
        node_prior = self._node_prior(node_mask.device)
        node_probs = node_prior.view(*([1] * node_mask.dim()), -1).expand(node_shape)
        node_t = self._sample_categorical(node_probs).masked_fill(~node_mask, 0)

        edge_t = None
        if edge_mask is not None:
            edge_shape = [*edge_mask.shape, self.num_edge_classes]
            edge_prior = self._edge_prior(edge_mask.device)
            edge_probs = edge_prior.view(*([1] * edge_mask.dim()), -1).expand(
                edge_shape
            )
            edge_t = self._sample_categorical(edge_probs)
            edge_t = torch.where(edge_mask, edge_t, torch.zeros_like(edge_t))
            edge_t = self._symmetrize_edges(edge_t, edge_mask)
        return node_t, edge_t

    def p_sample(
        self,
        node_t,
        edge_t,
        node_logits,
        edge_logits,
        node_mask,
        edge_mask,
        alpha_bar_t,
        alpha_bar_s,
    ):
        node_probs = self.posterior_probs(
            node_t,
            node_logits,
            alpha_bar_t,
            alpha_bar_s,
            self._node_prior(node_t.device),
            self._node_class_mask(node_t.device),
        )
        fallback = torch.full_like(node_probs, 1.0 / node_probs.size(-1))
        node_probs = torch.where(node_mask[..., None], node_probs, fallback)
        node_s = self._sample_categorical(node_probs)
        node_s = torch.where(node_mask, node_s, node_t)

        edge_s, edge_probs = None, None
        if edge_t is not None and edge_logits is not None and edge_mask is not None:
            edge_probs = self.posterior_probs(
                edge_t,
                edge_logits,
                alpha_bar_t,
                alpha_bar_s,
                self._edge_prior(edge_t.device),
            )
            edge_s = self._sample_categorical(edge_probs)
            edge_s = torch.where(edge_mask, edge_s, edge_t)
            edge_s = self._symmetrize_edges(edge_s, edge_mask)
        return node_s, edge_s, node_probs, edge_probs

    def posterior_probs(
        self,
        x_t,
        x0_logits,
        alpha_bar_t,
        alpha_bar_s,
        prior,
        class_mask=None,
    ):
        x0_probs = torch.softmax(x0_logits, dim=-1)
        if class_mask is not None:
            class_mask = class_mask.to(device=x0_probs.device, dtype=x0_probs.dtype)
            x0_probs = x0_probs * class_mask.view(*([1] * x_t.dim()), -1)
            x0_probs = x0_probs / x0_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        alpha_bar_t = self._as_batch_alpha(alpha_bar_t, x_t)
        alpha_bar_s = self._as_batch_alpha(alpha_bar_s, x_t).clamp_min(1e-8)
        alpha_t_given_s = (alpha_bar_t / alpha_bar_s).clamp(0.0, 1.0)

        prior = prior.to(x0_probs.device)
        prior_xt = prior[x_t.long().clamp(min=0, max=prior.numel() - 1)]
        same_class = F.one_hot(x_t.long(), num_classes=x0_probs.size(-1)).to(
            x0_probs.dtype
        )
        q_t_given_s = alpha_t_given_s * same_class + (
            1.0 - alpha_t_given_s
        ) * prior_xt.unsqueeze(-1)
        q_t_given_x0 = alpha_bar_t * same_class + (
            1.0 - alpha_bar_t
        ) * prior_xt.unsqueeze(-1)
        inverse = x0_probs / q_t_given_x0.clamp_min(1e-12)
        factor = alpha_bar_s * inverse + (1.0 - alpha_bar_s) * prior.view(
            *([1] * x_t.dim()), -1
        ) * inverse.sum(dim=-1, keepdim=True)
        posterior = q_t_given_s * factor
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # A recycled sample can contain a bond at alpha_t=0 even though the
        # absorbing no-bond prior assigns it zero mass. The ordinary posterior
        # is then 0/0. Its alpha_t -> 0+ limit preserves that observed class:
        # an absorbing forward process cannot create an out-of-prior class.
        outside_prior = prior_xt.eq(0).unsqueeze(-1) & alpha_bar_t.eq(0)
        if class_mask is not None:
            # Never reintroduce forbidden node classes (including padding).
            outside_prior &= class_mask[x_t.long()].bool().unsqueeze(-1)
        return torch.where(outside_prior, same_class, posterior)

    @staticmethod
    def _as_graph_time(time_step):
        if time_step.dim() == 0:
            return time_step.view(1).clamp(0.0, 1.0)
        if time_step.dim() == 1:
            return time_step.clamp(0.0, 1.0)
        dimensions = tuple(range(1, time_step.dim()))
        return time_step.float().mean(dim=dimensions).clamp(0.0, 1.0)

    @staticmethod
    def _alpha_bar(alpha_bar, time_step):
        if alpha_bar is None:
            return (1.0 - time_step).clamp(0.0, 1.0)
        if alpha_bar.dim() > 1:
            alpha_bar = alpha_bar.float().mean(dim=tuple(range(1, alpha_bar.dim())))
        alpha_bar = alpha_bar.float()
        if alpha_bar.max() > 1.0:
            alpha_bar = alpha_bar.clamp_min(0.0).sqrt()
        return alpha_bar.clamp(0.0, 1.0)

    @staticmethod
    def _as_batch_alpha(alpha_bar, x_t):
        if not torch.is_tensor(alpha_bar):
            alpha_bar = torch.tensor(alpha_bar, device=x_t.device, dtype=torch.float32)
        alpha_bar = alpha_bar.to(device=x_t.device, dtype=torch.float32)
        if alpha_bar.dim() == 0:
            alpha_bar = alpha_bar.view(1)
        if alpha_bar.dim() > 1:
            alpha_bar = alpha_bar.mean(dim=tuple(range(1, alpha_bar.dim())))
        return alpha_bar.view(x_t.size(0), *([1] * x_t.dim()))

    def _node_prior(self, device):
        prior = torch.ones(self.num_node_classes, device=device, dtype=torch.float32)
        prior[0] = 0.0
        prior = prior.masked_fill(~self._node_class_mask(device), 0.0)
        return prior / prior.sum().clamp_min(1.0)

    def _node_class_mask(self, device):
        mask = torch.zeros(self.num_node_classes, device=device, dtype=torch.bool)
        valid = torch.as_tensor(self.valid_node_classes, device=device)
        valid = valid[(valid >= 0) & (valid < self.num_node_classes)]
        mask[valid] = True
        return mask

    def _edge_prior(self, device):
        prior = torch.zeros(self.num_edge_classes, device=device, dtype=torch.float32)
        prior[0] = 1.0
        return prior

    @staticmethod
    def _marginal_probs(labels, num_classes, prior, alpha_bar):
        labels = labels.long().clamp(min=0, max=num_classes - 1)
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        alpha_bar = alpha_bar.view(labels.size(0), *([1] * labels.dim()))
        prior = prior.view(*([1] * labels.dim()), num_classes)
        return alpha_bar * one_hot + (1.0 - alpha_bar) * prior

    @staticmethod
    def _sample_categorical(probs):
        flat = probs.reshape(-1, probs.size(-1)).clamp_min(0.0)
        flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(flat, 1).view(probs.shape[:-1])

    @staticmethod
    def _symmetrize_edges(edge_t, edge_mask):
        if edge_t.dim() < 4:
            return edge_t
        upper = torch.triu(edge_t, diagonal=1)
        return (upper + upper.transpose(-1, -2)).masked_fill(~edge_mask, 0)


class MOFCompositionHead(nn.Module):
    """Predict clean compact building-block atom and bond labels."""

    def __init__(self, config, input_dim: int):
        super().__init__()
        self.config = config
        self.num_atom_classes = config.mof_composition_num_atom_classes
        self.num_bond_classes = config.mof_composition_num_bond_classes
        hidden_dim = config.embedding_dim
        self.context_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.bond_embedding = nn.Embedding(
            max(self.num_bond_classes, config.mof_composition_bond_mask_token + 1),
            hidden_dim,
        )
        self.atom_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.num_atom_classes, bias=False),
        )
        self.bond_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.num_bond_classes, bias=False),
        )

    def _sparse_bond_logits(self, atom_hidden, noised_bonds, pair_mask):
        batch_size, num_bbs, num_atoms, hidden_dim = atom_hidden.shape
        pair_index = pair_mask.nonzero(as_tuple=True)
        first_linear = self.bond_mlp[0]
        source_weight, target_weight, bond_weight = first_linear.weight.split(
            hidden_dim, dim=1
        )
        source = F.linear(atom_hidden, source_weight)
        target = F.linear(atom_hidden, target_weight)
        bond_table = F.linear(self.bond_embedding.weight, bond_weight)
        bond_types = (
            noised_bonds[pair_index].long().clamp(min=0, max=bond_table.size(0) - 1)
        )
        pair_hidden = (
            source[pair_index[0], pair_index[1], pair_index[2]]
            + target[pair_index[0], pair_index[1], pair_index[3]]
            + F.embedding(bond_types, bond_table)
        )
        pair_logits = self.bond_mlp[3](self.bond_mlp[2](self.bond_mlp[1](pair_hidden)))
        logits = atom_hidden.new_zeros(
            batch_size,
            num_bbs,
            num_atoms,
            num_atoms,
            self.num_bond_classes,
        )
        return logits.index_put(pair_index, pair_logits)

    def _dense_bond_logits(self, atom_hidden, noised_bonds):
        num_atoms = atom_hidden.size(2)
        source = atom_hidden.unsqueeze(3).expand(-1, -1, -1, num_atoms, -1)
        target = atom_hidden.unsqueeze(2).expand(-1, -1, num_atoms, -1, -1)
        bond = self.bond_embedding(noised_bonds.clamp_min(0))
        return self.bond_mlp(torch.cat([source, target, bond], dim=-1))

    def forward(
        self,
        context_features,
        padding_mask,
        noised_atom_types,
        noised_bonds,
        diffusion_time,
        batch,
    ):
        if context_features.size(0) != padding_mask.size(0):
            context_features = context_features.transpose(0, 1)
        atom_tokens = (
            torch.arange(padding_mask.size(1), device=padding_mask.device)[None]
            < batch["num_atoms"][:, None]
        )
        context_mask = (~padding_mask) & atom_tokens
        weights = context_mask.to(context_features.dtype).unsqueeze(-1)
        context = (context_features * weights)
        context = self.context_proj(context)

        batch_size, num_bbs, num_atoms = noised_atom_types.shape
        occurrence = batch["mof_bb_occ_atom_index"].long()
        occurrence_mask = batch["mof_bb_occ_mask"].bool()
        atom_mask = batch["mof_bb_atom_mask"].bool()
        bb_mask = batch["mof_bb_mask"].bool()
        num_repetitions = occurrence.shape[2]

        context_by_bb = torch.zeros([batch_size, num_bbs, num_atoms, context.size(-1)], dtype=context.dtype, device=context.device)

        for batch_idx in range(batch_size):
            for bb_idx in range(num_bbs):
                if bb_mask[batch_idx, bb_idx]:
                    for repetition_idx in range(num_repetitions):
                        if occurrence_mask[batch_idx, bb_idx, repetition_idx]:
                            bb_size = atom_mask[batch_idx, bb_idx].long().sum()
                            context_by_bb[batch_idx, bb_idx, :bb_size] = context_by_bb[batch_idx, bb_idx, :bb_size] + context[batch_idx][occurrence[batch_idx, bb_idx, repetition_idx, :bb_size]]

        context_by_unique_bb = context_by_bb / (occurrence_mask[..., None, None].long().sum(dim=2)).clamp(min=1)

        hidden = context_by_unique_bb.view(batch_size, num_bbs, num_atoms, -1)
        atom_logits = self.atom_mlp(hidden)
        if self.config.mof_composition_sparse_bond_head:
            bond_logits = self._sparse_bond_logits(
                hidden, noised_bonds, batch["mof_comp_bond_loss_mask"].bool()
            )
        else:
            bond_logits = self._dense_bond_logits(hidden, noised_bonds)
        return {"atom_logits": atom_logits, "bond_logits": bond_logits}


def masked_classification_loss(logits, labels, mask):
    """Return mean cross entropy, accuracy, and the supervised item count."""
    count = int(mask.sum())
    if count == 0:
        zero = logits.sum() * 0.0
        return zero, zero.detach(), 0
    selected_logits = logits[mask].float()
    selected_labels = labels[mask].long()
    loss = F.cross_entropy(selected_logits, selected_labels)
    accuracy = selected_logits.argmax(dim=-1).eq(selected_labels).float().mean()
    return loss, accuracy, count
