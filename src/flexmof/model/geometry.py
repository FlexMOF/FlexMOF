import math
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from flexmof.model.attention import PrecomputedLogAttentionWeights
from flexmof.model.periodic import iter_periodic_edge_slices, unpack_periodic_edges


def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (a * std)


class GaussianLayer(nn.Module):

    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(*x.shape[:-1], self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-05
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels
        if intermediate_channels is None:
            intermediate_channels = hidden_channels
        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        act_class_mapping = {"silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )
        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)
        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(-2) * vec2
        if self.act is not None:
            x = self.act(x)
        return (x, v)


class EquivariantVectorOutput(nn.Module):

    def __init__(self, hidden_channels=768, activation="silu"):
        super(EquivariantVectorOutput, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze(-1)


class EquivariantLayerNorm(nn.Module):
    """Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """

    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 0.001,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
        use_smooth_norm=False,
        add_vec_bias=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()
        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
        self.use_smooth_norm = use_smooth_norm
        if self.use_smooth_norm and add_vec_bias:
            self.eps = self.eps**2

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.
        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2) @ v.transpose(-2, -1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_smooth_norm:
            input = input.float()
            input_norm = input.pow(2).sum(dim=-2, keepdim=True)
            input_norm = torch.mean(input_norm, dim=-1)
            input_norm = (input_norm + self.eps).pow(-0.5)
            if self.elementwise_linear:
                input = input * self.weight
            else:
                input = input
            output = input * input_norm.unsqueeze(-1)
            return output.to(self.weight.dtype)
        else:
            input = input.to(torch.float64)
            input = self.mean_center(input)
            reg_matrix = (
                torch.diag(torch.tensor([1.0, 2.0, 3.0]))
                .unsqueeze(0)
                .to(input.device)
                .type(input.dtype)
            )
            covar = self.covariance(input) + self.eps * reg_matrix
            covar_sqrtinv = self.symsqrtinv(covar)
            return (covar_sqrtinv @ input).to(self.weight.dtype) * self.weight.reshape(
                1, 1, self.normalized_shape[0]
            )

    def extra_repr(self) -> str:
        return "{normalized_shape}, elementwise_linear={elementwise_linear}".format(
            **self.__dict__
        )


class MemEffInvariantAttention(nn.Module):

    def __init__(
        self,
        hidden_channels,
        head_dim,
        dropout,
        d_tilde=1,
        add_rope=True,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
        normalize_softmax_probs_after_local_attn_weights: bool = False,
        normalize_softmax_probs_after_local_attn_weights_eps: float = 1e-10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.head_dim = head_dim
        self.num_heads = hidden_channels // head_dim
        self.out_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.dropout = dropout
        self.attn_ln = nn.LayerNorm(hidden_channels)
        self.scaling = (self.head_dim / d_tilde) ** 0.5 / self.head_dim
        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
        self.use_no_pre_cutoff_softmax = use_no_pre_cutoff_softmax
        self.normalize_softmax_probs_after_local_attn_weights = (
            normalize_softmax_probs_after_local_attn_weights
        )
        self.normalize_softmax_probs_after_local_attn_weights_eps = (
            normalize_softmax_probs_after_local_attn_weights_eps
        )
        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        q,
        k,
        v,
        attn_bias=None,
        key_padding_mask=None,
        pbc_expand_batched: Optional[Dict] = None,
        softmax_weights: Optional[Tensor] = None,
    ):
        if pbc_expand_batched is not None and softmax_weights is None:
            outcell_index = (
                pbc_expand_batched["outcell_index"]
                .unsqueeze(-1)
                .repeat(1, 1, k.size()[-1])
            )
            local_attention_weight = pbc_expand_batched["local_attention_weight"].to(
                dtype=q.dtype
            )
            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)
            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
            expand_mask = pbc_expand_batched["expand_mask"]
        else:
            local_attention_weight = None
            expand_mask = None
        bsz, tgt_len, src_len = (q.shape[0], q.shape[1], k.shape[1])
        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        if key_padding_mask is not None:
            if pbc_expand_batched is not None and softmax_weights is None:
                assert expand_mask is not None
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=1)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        effective_bias = None if softmax_weights is not None else attn_bias
        attn_mask = (
            effective_bias.contiguous().view(bsz, self.num_heads, tgt_len, src_len)
            if effective_bias is not None
            else None
        )
        if key_padding_mask is not None and key_padding_mask.bool().any():
            if (
                softmax_weights is None
                or not self.normalize_softmax_probs_after_local_attn_weights
            ):
                if attn_mask is None:
                    attn_mask = torch.zeros(
                        bsz, 1, 1, src_len, device=q.device, dtype=q.dtype
                    )
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
        if isinstance(softmax_weights, PrecomputedLogAttentionWeights):
            softmax_log_weights = softmax_weights.value
            softmax_weights = None
        else:
            softmax_log_weights = None
        weighted_attention = softmax_weights
        if weighted_attention is None and softmax_log_weights is None:
            weighted_attention = local_attention_weight
        can_fuse = (
            (weighted_attention is not None or softmax_log_weights is not None)
            and self.normalize_softmax_probs_after_local_attn_weights
            and (not self.use_smooth_softmax)
        )
        if can_fuse:
            if softmax_log_weights is not None:
                log_weight = softmax_log_weights
            else:
                weight = weighted_attention.to(dtype=q.dtype)
                if weight.dim() == 3:
                    weight = weight.unsqueeze(1)
                log_weight = weight.clamp_min(
                    self.normalize_softmax_probs_after_local_attn_weights_eps
                ).log_()
                log_weight.masked_fill_(weight <= 0, float("-inf"))
                if attn_mask is not None:
                    log_weight.add_(attn_mask)
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                attn = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    attn_mask=log_weight,
                    is_causal=False,
                )
        elif (
            local_attention_weight is not None
            or attn_bias is not None
            or softmax_weights is not None
        ):
            attn_weights = (q * self.scaling).matmul(k.transpose(-1, -2))
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            if local_attention_weight is not None:
                local_attention_weight = local_attention_weight.to(dtype=q.dtype)
                if self.use_smooth_softmax:
                    attn_weights = (
                        attn_weights + self.smooth_factor
                    ) * local_attention_weight.unsqueeze(1) - self.smooth_factor
                    if key_padding_mask is not None:
                        attn_weights = attn_weights.masked_fill(
                            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                            float("-inf"),
                        )
                elif not self.use_no_pre_cutoff_softmax:
                    attn_weights = attn_weights.masked_fill(
                        (local_attention_weight <= 1e-05).unsqueeze(1), float("-inf")
                    )
            attn_probs_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            if local_attention_weight is not None:
                attn_probs_float = attn_probs_float * local_attention_weight.unsqueeze(
                    1
                )
                if self.normalize_softmax_probs_after_local_attn_weights:
                    attn_probs_float = attn_probs_float / (
                        attn_probs_float.sum(dim=-1, keepdim=True)
                        + self.normalize_softmax_probs_after_local_attn_weights_eps
                    )
            elif softmax_weights is not None:
                attn_probs_float = attn_probs_float * softmax_weights
                if self.normalize_softmax_probs_after_local_attn_weights:
                    attn_probs_float = attn_probs_float / (
                        attn_probs_float.sum(dim=-1, keepdim=True)
                        + self.normalize_softmax_probs_after_local_attn_weights_eps
                    )
            attn_probs_float = F.dropout(
                attn_probs_float, self.dropout, training=self.training
            )
            attn_probs = attn_probs_float.type_as(v)
            attn = attn_probs.matmul(v)
        else:
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                if attn_mask is not None:
                    attn_mask = attn_mask.to(dtype=q.dtype)
                attn = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    attn_mask=attn_mask,
                    is_causal=False,
                )
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.hidden_channels)
        attn = self.attn_ln(attn)
        attn = self.out_proj(attn)
        return attn


class MemEffEquivariantAttention(nn.Module):

    def __init__(
        self,
        hidden_channels,
        head_dim,
        dropout,
        d_tilde=1,
        use_smooth_softmax=False,
        use_smooth_equviariant_norm=False,
        smooth_factor=0.0,
        add_rope=True,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
        normalize_softmax_probs_after_local_attn_weights: bool = False,
        normalize_softmax_probs_after_local_attn_weights_eps: float = 1e-10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.head_dim = head_dim
        self.num_heads = hidden_channels // head_dim
        self.out_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.dropout = dropout
        self.attn_ln = EquivariantLayerNorm(
            hidden_channels,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )
        self.scaling = (self.head_dim / (d_tilde * 3)) ** 0.5 / self.head_dim
        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
        self.use_no_pre_cutoff_softmax = use_no_pre_cutoff_softmax
        self.normalize_softmax_probs_after_local_attn_weights = (
            normalize_softmax_probs_after_local_attn_weights
        )
        self.normalize_softmax_probs_after_local_attn_weights_eps = (
            normalize_softmax_probs_after_local_attn_weights_eps
        )
        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        q,
        k,
        v,
        attn_bias=None,
        key_padding_mask=None,
        pbc_expand_batched: Optional[Dict] = None,
        softmax_weights: Optional[Tensor] = None,
    ):
        if pbc_expand_batched is not None and softmax_weights is None:
            outcell_index = (
                pbc_expand_batched["outcell_index"]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, k.size()[-2], k.size()[-1])
            )
            local_attention_weight = pbc_expand_batched["local_attention_weight"].to(
                dtype=q.dtype
            )
            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)
            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
            expand_mask = pbc_expand_batched["expand_mask"]
        else:
            local_attention_weight = None
            expand_mask = None
        bsz, tgt_len, src_len = (q.shape[0], q.shape[1], k.shape[1])
        pos_feature_num = q.shape[2]
        q = (
            q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
        )
        k = (
            k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
        )
        v = (
            v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
        )
        attn_mask = None
        if key_padding_mask is not None:
            if pbc_expand_batched is not None and softmax_weights is None:
                expand_mask = pbc_expand_batched["expand_mask"]
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=-1)
            if key_padding_mask.bool().any() and (
                softmax_weights is None
                or not self.normalize_softmax_probs_after_local_attn_weights
            ):
                attn_mask = (
                    attn_bias.contiguous().view(bsz, self.num_heads, tgt_len, src_len)
                    if attn_bias is not None
                    else torch.zeros(bsz, 1, 1, src_len, device=q.device, dtype=q.dtype)
                ).masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
        if attn_mask is None and attn_bias is not None and (softmax_weights is None):
            attn_mask = attn_bias.contiguous().view(
                bsz, self.num_heads, tgt_len, src_len
            )
        if isinstance(softmax_weights, PrecomputedLogAttentionWeights):
            softmax_log_weights = softmax_weights.value
            softmax_weights = None
        else:
            softmax_log_weights = None
        weighted_attention = softmax_weights
        if weighted_attention is None and softmax_log_weights is None:
            weighted_attention = local_attention_weight
        can_fuse = (
            (weighted_attention is not None or softmax_log_weights is not None)
            and self.normalize_softmax_probs_after_local_attn_weights
            and (not self.use_smooth_softmax)
        )
        if can_fuse:
            if softmax_log_weights is not None:
                log_weight = softmax_log_weights
            else:
                weight = weighted_attention.to(dtype=q.dtype)
                if weight.dim() == 3:
                    weight = weight.unsqueeze(1)
                log_weight = weight.clamp_min(
                    self.normalize_softmax_probs_after_local_attn_weights_eps
                ).log_()
                log_weight.masked_fill_(weight <= 0, float("-inf"))
                if attn_mask is not None:
                    log_weight.add_(attn_mask)
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                attn = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    attn_mask=log_weight,
                    is_causal=False,
                )
        elif (
            local_attention_weight is not None
            or attn_bias is not None
            or softmax_weights is not None
        ):
            attn_weights = (q * self.scaling).matmul(k.transpose(-1, -2))
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            if local_attention_weight is not None:
                local_attention_weight = local_attention_weight.to(dtype=q.dtype)
                if self.use_smooth_softmax:
                    attn_weights = (
                        attn_weights + self.smooth_factor
                    ) * local_attention_weight.unsqueeze(1) - self.smooth_factor
                    if key_padding_mask is not None:
                        attn_weights = attn_weights.masked_fill(
                            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                            float("-inf"),
                        )
                elif not self.use_no_pre_cutoff_softmax:
                    attn_weights = attn_weights.masked_fill(
                        (local_attention_weight <= 1e-05).unsqueeze(1), float("-inf")
                    )
            attn_probs_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            if local_attention_weight is not None:
                attn_probs_float = attn_probs_float * local_attention_weight.unsqueeze(
                    1
                )
                if self.normalize_softmax_probs_after_local_attn_weights:
                    attn_probs_float = attn_probs_float / (
                        attn_probs_float.sum(dim=-1, keepdim=True)
                        + self.normalize_softmax_probs_after_local_attn_weights_eps
                    )
            elif softmax_weights is not None:
                attn_probs_float = attn_probs_float * softmax_weights
                if self.normalize_softmax_probs_after_local_attn_weights:
                    attn_probs_float = attn_probs_float / (
                        attn_probs_float.sum(dim=-1, keepdim=True)
                        + self.normalize_softmax_probs_after_local_attn_weights_eps
                    )
            attn_probs_float = F.dropout(
                attn_probs_float, self.dropout, training=self.training
            )
            attn_probs = attn_probs_float.type_as(v)
            attn = attn_probs.matmul(v)
        else:
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                if attn_mask is not None:
                    attn_mask = attn_mask.to(dtype=q.dtype)
                attn = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    attn_mask=attn_mask,
                    is_causal=False,
                )
        attn = (
            attn.transpose(1, 2)
            .reshape(bsz, tgt_len, self.num_heads, pos_feature_num, self.head_dim)
            .transpose(2, 3)
            .reshape(bsz, tgt_len, pos_feature_num, self.hidden_channels)
        )
        attn = self.attn_ln(attn)
        attn = self.out_proj(attn)
        return attn


class InvariantSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_channels,
        head_dim,
        num_heads,
        dropout,
        use_linear_bias=False,
        use_memory_efficient_attention=True,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
        normalize_softmax_probs_after_local_attn_weights: bool = False,
        normalize_softmax_probs_after_local_attn_weights_eps: float = 1e-10,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.use_linear_bias = use_linear_bias
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.invariant_attention = MemEffInvariantAttention(
            hidden_channels,
            head_dim,
            dropout,
            use_smooth_softmax=use_smooth_softmax,
            smooth_factor=smooth_factor,
            use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            normalize_softmax_probs_after_local_attn_weights=normalize_softmax_probs_after_local_attn_weights,
            normalize_softmax_probs_after_local_attn_weights_eps=normalize_softmax_probs_after_local_attn_weights_eps,
        )
        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.v_proj.bias.data.fill_(0)

    def forward(
        self,
        x,
        attn_bias=None,
        mask=None,
        pbc_expand_batched: Optional[Dict] = None,
        softmax_weights: Optional[Tensor] = None,
    ):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = self.invariant_attention(
            q,
            k,
            v,
            attn_bias,
            mask,
            pbc_expand_batched=pbc_expand_batched,
            softmax_weights=softmax_weights,
        )
        return attn


class EquivariantSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_channels,
        head_dim,
        num_heads,
        dropout,
        use_memory_efficient_attention=True,
        use_smooth_equviariant_norm=False,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        add_rope=True,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
        normalize_softmax_probs_after_local_attn_weights: bool = False,
        normalize_softmax_probs_after_local_attn_weights_eps: float = 1e-10,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.equiariant_attention = MemEffEquivariantAttention(
            hidden_channels,
            head_dim,
            dropout,
            use_smooth_equviariant_norm=use_smooth_equviariant_norm,
            use_smooth_softmax=use_smooth_softmax,
            smooth_factor=smooth_factor,
            use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            normalize_softmax_probs_after_local_attn_weights=normalize_softmax_probs_after_local_attn_weights,
            normalize_softmax_probs_after_local_attn_weights_eps=normalize_softmax_probs_after_local_attn_weights_eps,
        )
        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        vec,
        attn_bias=None,
        mask=None,
        pbc_expand_batched: Optional[Dict] = None,
        softmax_weights: Optional[Tensor] = None,
    ):
        q = self.q_proj(vec)
        k = self.k_proj(vec)
        v = self.v_proj(vec)
        attn = self.equiariant_attention(
            q,
            k,
            v,
            attn_bias,
            mask,
            pbc_expand_batched=pbc_expand_batched,
            softmax_weights=softmax_weights,
        )
        return attn


class Invariant2EquivariantAttention(nn.Module):

    def __init__(
        self,
        hidden_channels,
        head_dim,
        num_heads,
        dropout,
        use_linear_bias=False,
        use_memory_efficient_attention=True,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
        normalize_softmax_probs_after_local_attn_weights: bool = False,
        normalize_softmax_probs_after_local_attn_weights_eps: float = 1e-10,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.use_linear_bias = use_linear_bias
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.k1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.invariant_attention = MemEffInvariantAttention(
            hidden_channels,
            head_dim,
            dropout,
            use_smooth_softmax=use_smooth_softmax,
            smooth_factor=smooth_factor,
            use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            normalize_softmax_probs_after_local_attn_weights=normalize_softmax_probs_after_local_attn_weights,
            normalize_softmax_probs_after_local_attn_weights_eps=normalize_softmax_probs_after_local_attn_weights_eps,
        )
        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.k2_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.v1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.v2_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        x,
        vec,
        attn_bias=None,
        mask=None,
        pbc_expand_batched: Optional[Dict] = None,
        softmax_weights: Optional[Tensor] = None,
    ):
        pos_feature_num = vec.shape[2]
        q = self.q_proj(x)
        k1 = self.k1_proj(vec)
        k2 = self.k2_proj(vec)
        k = (k1 * k2).sum(dim=-2) * pos_feature_num ** (-0.5)
        v1 = self.v1_proj(vec)
        v2 = self.v2_proj(vec)
        v = (v1 * v2).sum(dim=-2) * pos_feature_num ** (-0.5)
        attn = self.invariant_attention(
            q,
            k,
            v,
            attn_bias,
            mask,
            pbc_expand_batched=pbc_expand_batched,
            softmax_weights=softmax_weights,
        )
        return attn


class Equivariant2InvariantAttention(nn.Module):

    def __init__(
        self,
        hidden_channels,
        head_dim,
        num_heads,
        dropout,
        eQi_choice,
        gbf_args,
        use_linear_bias=False,
        use_memory_efficient_attention=True,
        use_smooth_equviariant_norm=False,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        add_rope=True,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
        normalize_softmax_probs_after_local_attn_weights: bool = False,
        normalize_softmax_probs_after_local_attn_weights_eps: float = 1e-10,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.eQi_choice = eQi_choice
        self.K = gbf_args[0]
        self.edge_types = gbf_args[1]
        self.use_linear_bias = use_linear_bias
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k1_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.v1_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        if eQi_choice == "original":
            self.k2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.v2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        elif "gbf" in eQi_choice:
            self.gbf = GaussianLayer(self.K, self.edge_types)
            self.gbf_proj = nn.Linear(self.K, hidden_channels, bias=use_linear_bias)
        self.equiariant_attention = MemEffEquivariantAttention(
            hidden_channels,
            head_dim,
            dropout,
            use_smooth_equviariant_norm=use_smooth_equviariant_norm,
            use_smooth_softmax=use_smooth_softmax,
            smooth_factor=smooth_factor,
            use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            add_vec_bias=add_vec_bias,
            normalize_softmax_probs_after_local_attn_weights=normalize_softmax_probs_after_local_attn_weights,
            normalize_softmax_probs_after_local_attn_weights_eps=normalize_softmax_probs_after_local_attn_weights_eps,
        )
        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.k1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.k1_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k2_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.v1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.v1_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v2_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        x,
        vec,
        attn_bias,
        mask,
        pos_unit=None,
        gbf_args=None,
        pbc_expand_batched: Optional[Dict] = None,
        softmax_weights: Optional[Tensor] = None,
    ):
        q = self.q_proj(vec)
        k1 = self.k1_proj(x)
        v1 = self.v1_proj(x)
        pos_mean_centered_unit, pos_relative_unit = pos_unit
        if "gbf" in self.eQi_choice:
            dist, edge_type = gbf_args
            gbf_feature = self.gbf_proj(self.gbf(dist, edge_type))
            edge_feature = gbf_feature.masked_fill(mask.unsqueeze(1).unsqueeze(-1), 0.0)
        if self.eQi_choice == "original":
            k2 = self.k2_proj(vec)
            k = k1.unsqueeze(2) * k2
            v2 = self.v2_proj(vec)
            v = v1.unsqueeze(2) * v2
        elif self.eQi_choice == "meanCentered_vanilla":
            k = pos_mean_centered_unit.unsqueeze(-1) * k1.unsqueeze(-2)
            v = pos_mean_centered_unit.unsqueeze(-1) * v1.unsqueeze(-2)
        elif self.eQi_choice == "sumRelative_vanilla":
            k_edge = pos_relative_unit.unsqueeze(-1) * k1.unsqueeze(2).unsqueeze(-2)
            k_edge = k_edge.masked_fill(
                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            k = k_edge.sum(dim=2)
            k = k.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            v_edge = pos_relative_unit.unsqueeze(-1) * v1.unsqueeze(2).unsqueeze(-2)
            v_edge = v_edge.masked_fill(
                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            v = v_edge.sum(dim=2)
            v = v.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        elif self.eQi_choice == "meanCentered_gbf":
            gbf_sum = pos_mean_centered_unit.unsqueeze(-1) * edge_feature.sum(
                dim=2
            ).unsqueeze(-2)
            gbf_sum = gbf_sum.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            k = k1.unsqueeze(-2) * gbf_sum
            v = v1.unsqueeze(-2) * gbf_sum
        elif self.eQi_choice == "sumRelative_gbf":
            feat_edge = pos_relative_unit.unsqueeze(-1) * edge_feature.unsqueeze(-2)
            feat_edge = feat_edge.masked_fill(
                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            feat_sum = feat_edge.sum(dim=2)
            feat_sum = feat_sum.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            k = k1.unsqueeze(-2) * feat_sum
            v = v1.unsqueeze(-2) * feat_sum
        attn = self.equiariant_attention(
            q,
            k,
            v,
            attn_bias,
            mask,
            pbc_expand_batched=pbc_expand_batched,
            softmax_weights=softmax_weights,
        )
        return attn


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


class EncoderLayer(nn.Module):

    def __init__(
        self,
        hidden_channels,
        num_heads,
        ffn_embedding_dim,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        eQi_choice: str = "original",
        gbf_args=None,
        layer_index=0,
        use_memory_efficient_attention=False,
        use_linear_bias=False,
        use_smooth_equviariant_norm=False,
        use_smooth_softmax=False,
        no_rotary_embedding_for_vector=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
        normalize_softmax_probs_after_local_attn_weights: bool = False,
        normalize_softmax_probs_after_local_attn_weights_eps: float = 1e-10,
    ):
        super().__init__()
        head_dim = hidden_channels // num_heads
        self.layer_index = layer_index
        self.use_linear_bias = use_linear_bias
        if self.layer_index % 2 == 0:
            self.invariant_self_attention = InvariantSelfAttention(
                hidden_channels,
                head_dim,
                num_heads,
                dropout=attention_dropout,
                use_memory_efficient_attention=use_memory_efficient_attention,
                use_linear_bias=use_linear_bias,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                normalize_softmax_probs_after_local_attn_weights=normalize_softmax_probs_after_local_attn_weights,
                normalize_softmax_probs_after_local_attn_weights_eps=normalize_softmax_probs_after_local_attn_weights_eps,
            )
            self.equivariant_self_attention = EquivariantSelfAttention(
                hidden_channels,
                head_dim,
                num_heads,
                dropout=attention_dropout,
                use_memory_efficient_attention=use_memory_efficient_attention,
                use_smooth_equviariant_norm=use_smooth_equviariant_norm,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                add_rope=not no_rotary_embedding_for_vector,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                add_vec_bias=add_vec_bias,
                normalize_softmax_probs_after_local_attn_weights=normalize_softmax_probs_after_local_attn_weights,
                normalize_softmax_probs_after_local_attn_weights_eps=normalize_softmax_probs_after_local_attn_weights_eps,
            )
        else:
            self.invariant2equivariant_attention = Invariant2EquivariantAttention(
                hidden_channels,
                head_dim,
                num_heads,
                dropout=attention_dropout,
                use_memory_efficient_attention=use_memory_efficient_attention,
                use_linear_bias=use_linear_bias,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                normalize_softmax_probs_after_local_attn_weights=normalize_softmax_probs_after_local_attn_weights,
                normalize_softmax_probs_after_local_attn_weights_eps=normalize_softmax_probs_after_local_attn_weights_eps,
            )
            self.equivaiant2invariant_attention = Equivariant2InvariantAttention(
                hidden_channels,
                head_dim,
                num_heads,
                dropout=attention_dropout,
                eQi_choice=eQi_choice,
                gbf_args=gbf_args,
                use_memory_efficient_attention=use_memory_efficient_attention,
                use_linear_bias=use_linear_bias,
                use_smooth_equviariant_norm=use_smooth_equviariant_norm,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                add_rope=not no_rotary_embedding_for_vector,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                add_vec_bias=add_vec_bias,
                normalize_softmax_probs_after_local_attn_weights=normalize_softmax_probs_after_local_attn_weights,
                normalize_softmax_probs_after_local_attn_weights_eps=normalize_softmax_probs_after_local_attn_weights_eps,
            )
        self.invariant_attn_layer_norm = nn.LayerNorm(hidden_channels)
        self.equivariant_attn_layer_norm = EquivariantLayerNorm(
            hidden_channels,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )
        self.activation_fn = gelu
        self.invariant_fc1 = nn.Linear(
            hidden_channels, ffn_embedding_dim, bias=use_linear_bias
        )
        self.invariant_fc2 = nn.Linear(
            ffn_embedding_dim, hidden_channels, bias=use_linear_bias
        )
        self.equivariant_fc1 = nn.Linear(
            hidden_channels, ffn_embedding_dim, bias=use_linear_bias
        )
        self.equivariant_fc2 = nn.Linear(hidden_channels, ffn_embedding_dim, bias=False)
        self.equivariant_fc3 = nn.Linear(ffn_embedding_dim, hidden_channels, bias=False)
        self.invariant_ffn_layer_norm = nn.LayerNorm(hidden_channels)
        self.equivariant_ffn_layer_norm = EquivariantLayerNorm(
            hidden_channels,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )
        self.invariant_ffn_layer_norm_2 = nn.LayerNorm(ffn_embedding_dim)
        self.equivariant_ffn_layer_norm_2 = EquivariantLayerNorm(
            ffn_embedding_dim,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.reset_parameters()

    def reset_parameters(self, d_tilde=1.0):
        self.invariant_attn_layer_norm.reset_parameters()
        self.equivariant_attn_layer_norm.reset_parameters()
        nn.init.xavier_uniform_(
            self.invariant_fc1.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        if self.use_linear_bias:
            self.invariant_fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(
            self.invariant_fc2.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        if self.use_linear_bias:
            self.invariant_fc2.bias.data.fill_(0)
        nn.init.xavier_uniform_(
            self.equivariant_fc1.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        if self.use_linear_bias:
            self.equivariant_fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(
            self.equivariant_fc2.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        nn.init.xavier_uniform_(
            self.equivariant_fc3.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        self.invariant_ffn_layer_norm.reset_parameters()
        self.equivariant_ffn_layer_norm.reset_parameters()
        self.invariant_ffn_layer_norm_2.reset_parameters()
        self.equivariant_ffn_layer_norm_2.reset_parameters()

    def forward(
        self,
        x,
        vec,
        attn_bias_iself,
        attn_bias_i2e,
        attn_bias_eself,
        attn_bias_e2i,
        mask,
        pos_unit=None,
        gbf_args=None,
        pbc_expand_batched: Optional[Dict] = None,
        softmax_weights: Optional[Tensor] = None,
    ):
        dx = self.invariant_attn_layer_norm(x)
        dvec = self.equivariant_attn_layer_norm(vec)
        if self.layer_index % 2 == 0:
            dx_invariant = self.invariant_self_attention(
                dx,
                attn_bias_iself,
                mask,
                pbc_expand_batched=pbc_expand_batched,
                softmax_weights=softmax_weights,
            )
            dx_invariant = F.dropout(
                dx_invariant, p=self.dropout, training=self.training
            )
            dvec_equivariant = self.equivariant_self_attention(
                dvec,
                attn_bias_eself,
                mask,
                pbc_expand_batched=pbc_expand_batched,
                softmax_weights=softmax_weights,
            )
            x = x + dx_invariant
            vec = vec + dvec_equivariant
        else:
            dx_equivariant = self.invariant2equivariant_attention(
                dx,
                dvec,
                attn_bias_i2e,
                mask,
                pbc_expand_batched=pbc_expand_batched,
                softmax_weights=softmax_weights,
            )
            dx_equivariant = F.dropout(
                dx_equivariant, p=self.dropout, training=self.training
            )
            dvec_invariant = self.equivaiant2invariant_attention(
                dx,
                dvec,
                attn_bias_e2i,
                mask,
                pos_unit,
                gbf_args,
                pbc_expand_batched=pbc_expand_batched,
                softmax_weights=softmax_weights,
            )
            x = x + dx_equivariant
            vec = vec + dvec_invariant
        dx = self.invariant_ffn_layer_norm(x)
        dvec = self.equivariant_ffn_layer_norm(vec)
        dx_ffn = self.activation_fn(self.invariant_fc1(dx))
        dx_ffn = F.dropout(dx_ffn, p=self.activation_dropout, training=self.training)
        dx_ffn = self.invariant_ffn_layer_norm_2(dx_ffn)
        dx_ffn = self.invariant_fc2(dx_ffn)
        dx_ffn = F.dropout(dx_ffn, p=self.dropout, training=self.training)
        dvec_ffn = self.activation_fn(self.equivariant_fc1(dx)).unsqueeze(
            -2
        ) * self.equivariant_fc2(dvec)
        dvec_ffn = self.equivariant_ffn_layer_norm_2(dvec_ffn)
        dvec_ffn = self.equivariant_fc3(dvec_ffn)
        x = x + dx_ffn
        vec = vec + dvec_ffn
        return (x, vec)


class MOFGeometry(nn.Module):
    """Checkpoint-compatible equivariant decoder specialized for periodic MOFs."""

    def __init__(self, config):
        super().__init__()
        self.unified_encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.embedding_dim,
                    config.num_attention_heads,
                    config.decoder_ffn_dim,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    activation_dropout=config.activation_dropout,
                    eQi_choice="original",
                    gbf_args=[config.num_3d_bias_kernel, config.num_edges],
                    layer_index=index,
                    use_memory_efficient_attention=True,
                    use_linear_bias=config.equivar_use_linear_bias,
                    use_smooth_equviariant_norm=config.use_smooth_equviariant_norm,
                    use_smooth_softmax=False,
                    no_rotary_embedding_for_vector=True,
                    use_no_pre_cutoff_softmax=True,
                    normalize_softmax_probs_after_local_attn_weights=True,
                    normalize_softmax_probs_after_local_attn_weights_eps=config.normalize_softmax_probs_after_local_attn_weights_eps,
                )
                for index in range(config.num_pred_attn_layer)
            ]
        )
        self.unified_gbf_vec = GaussianLayer(
            config.num_3d_bias_kernel, config.num_edges
        )
        self.unified_vec_proj = nn.Linear(
            config.num_3d_bias_kernel, config.embedding_dim, bias=False
        )
        self.unified_augmented_vec_proj = nn.Linear(
            config.num_3d_bias_kernel,
            config.embedding_dim,
            bias=config.equivar_use_linear_bias,
        )
        self.unified_final_equivariant_ln = EquivariantLayerNorm(
            config.embedding_dim,
            use_smooth_norm=config.use_smooth_equviariant_norm,
            add_vec_bias=False,
        )
        self.unified_final_invariant_ln = nn.LayerNorm(config.embedding_dim)
        self.time_embed_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.num_3d_bias_kernel),
        )

    def _streaming_vector_reduction(
        self, batch, positions, time_projection, periodic, edge_offset, edges
    ):
        batch_size, node_count = positions.shape[:2]
        edge_batch, query, _, source = unpack_periodic_edges(periodic, edges)
        edge_relative = periodic["edge_relative"][
            edge_offset : edge_offset + edges.numel()
        ]
        edge_distance = periodic["edge_distance"][
            edge_offset : edge_offset + edges.numel()
        ]
        edge_direction = edge_relative / (edge_distance[:, None] + 1e-05)
        edge_direction = torch.cat(
            [torch.ones_like(edge_direction[:, :1]), edge_direction], dim=-1
        )
        edge_types = batch["node_type_edge"][edge_batch, query, source]
        edge_radial = self.unified_gbf_vec(edge_distance, edge_types)
        edge_radial += time_projection[edge_batch, query]
        edge_ratio = edge_distance / periodic["cutoff"]
        edge_weight = torch.clamp(
            1 - 6 * edge_ratio**5 + 15 * edge_ratio**4 - 10 * edge_ratio**3,
            min=0.0,
        )
        contribution = (
            edge_direction[..., None]
            * edge_radial[:, None, :]
            * edge_weight[:, None, None]
        )
        target = edge_batch * node_count + query
        return (
            torch.zeros(
                batch_size * node_count,
                4,
                contribution.size(-1),
                dtype=contribution.dtype,
                device=contribution.device,
            )
            .scatter_add(
                0,
                target[:, None, None].expand_as(contribution),
                contribution,
            )
            .view(batch_size, node_count, 4, -1)
        )

    def forward(
        self,
        batch,
        invariant,
        positions,
        attention_bias,
        padding_mask,
        periodic,
        time_embed=None,
        softmax_weights=None,
    ):
        positions = positions.to(dtype=invariant.dtype)
        time_embed = time_embed.to(dtype=invariant.dtype)
        streaming = periodic.get("streaming_per_atom", False)
        if streaming:
            batch_size, node_count = positions.shape[:2]
            time_projection = self.time_embed_proj(time_embed)
            relative = positions[:, :, None] - positions[:, None]
            distances = relative.norm(dim=-1)
            relative = relative / (distances[..., None] + 1e-05)
            radial = self.unified_gbf_vec(distances, batch["node_type_edge"])
            radial += time_projection[:, :, None]
            pair_mask = padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(1)
            radial = radial.masked_fill(pair_mask[..., None], 0.0)
            ratio = distances / periodic["cutoff"]
            local_weight = torch.clamp(
                1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3, min=0.0
            ).masked_fill(pair_mask, 0.0)
            direction = torch.cat(
                [torch.ones_like(relative[..., :1]), relative], dim=-1
            )
            vectors = (
                direction[..., None]
                * radial[..., None, :]
                * local_weight[..., None, None]
            ).sum(dim=2)

            for edge_offset, edges in iter_periodic_edge_slices(
                positions, batch["masked_token_type"], periodic
            ):
                reduction = partial(
                    self._streaming_vector_reduction,
                    batch,
                    positions,
                    time_projection,
                    periodic,
                    edge_offset,
                )
                if (
                    self.training
                    and torch.is_grad_enabled()
                    and periodic["checkpoint_edge_activations"]
                ):
                    addition = checkpoint(reduction, edges, use_reentrant=False)
                else:
                    addition = reduction(edges)
                vectors = vectors + addition
            layer_periodic = None
            layer_edge_types = batch["node_type_edge"]
        elif periodic["outcell_index"].dim() == 3:
            expanded_positions = torch.cat(
                [
                    positions[:, None].expand(-1, positions.size(1), -1, -1),
                    periodic["expand_pos"],
                ],
                dim=2,
            )
            relative = positions[:, :, None] - expanded_positions
            expanded_mask = torch.cat(
                [
                    padding_mask[:, None].expand(-1, positions.size(1), -1),
                    periodic["expand_mask"],
                ],
                dim=-1,
            )
        else:
            expanded_positions = torch.cat([positions, periodic["expand_pos"]], dim=1)
            relative = positions[:, :, None] - expanded_positions[:, None]
            expanded_mask = torch.cat([padding_mask, periodic["expand_mask"]], dim=-1)
        if not streaming:
            distances = relative.norm(dim=-1)
            relative = relative / (distances[..., None] + 1e-05)
            radial = self.unified_gbf_vec(distances, periodic["expand_node_type_edge"])
            radial += self.time_embed_proj(time_embed)[:, :, None]
            radial = radial.masked_fill(padding_mask[:, :, None, None], 0.0)
            if expanded_mask.dim() == 3:
                radial = radial.masked_fill(expanded_mask[..., None], 0.0)
            else:
                radial = radial.masked_fill(expanded_mask[:, None, :, None], 0.0)
            relative = torch.cat([torch.ones_like(relative[..., :1]), relative], dim=-1)
            vector_edges = relative[..., None] * radial[..., None, :]
            local_weights = periodic["local_attention_weight"]
            if local_weights is not None:
                vector_edges = vector_edges * local_weights[..., None, None].to(
                    invariant.dtype
                )
            vectors = vector_edges.sum(dim=2)
            layer_periodic = periodic
            layer_edge_types = periodic["expand_node_type_edge"]
        augmented = self.unified_augmented_vec_proj(vectors[:, :, :1])
        vectors = self.unified_vec_proj(vectors[:, :, 1:])
        vectors = torch.cat([augmented, vectors], dim=2)
        vectors = vectors.masked_fill(padding_mask[:, :, None, None], 0.0)
        output = invariant.transpose(0, 1).contiguous()
        output = output.masked_fill(padding_mask[..., None], 0.0)
        position_unit = positions / (positions.norm(dim=-1, keepdim=True) + 1e-05)
        for layer in self.unified_encoder_layers:
            output, vectors = layer(
                output,
                vectors,
                attention_bias,
                attention_bias,
                attention_bias,
                attention_bias,
                padding_mask,
                [position_unit, relative],
                [distances, layer_edge_types],
                layer_periodic,
                softmax_weights=softmax_weights,
            )
        output = self.unified_final_invariant_ln(output)
        vectors = self.unified_final_equivariant_ln(vectors)[:, :, 1:]
        return (output, vectors)
