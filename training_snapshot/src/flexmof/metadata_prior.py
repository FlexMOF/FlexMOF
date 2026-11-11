"""Learned autoregressive distribution over MOF sizes and multiplicities.

The sequence is N, B, then (role, repetitions, explicit size, core size) for
each unique block slot. A finite-budget grammar makes N = sum(R*S) exact.
No chemical graph, source key, or source MOF is required at inference.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TOTAL, BLOCKS, ROLE, REPEAT, SIZE, CORE, PAD = range(7)


@dataclass
class MetadataConfig:
    max_atoms: int = 768
    max_blocks: int = 9
    max_size: int = 256
    max_repetitions: int = 128
    max_hydrogens: int = 4
    width: int = 128
    layers: int = 2
    embedding_dim: int = 64

    def __post_init__(self):
        if min(self.max_atoms, self.max_size, self.max_repetitions,
               self.width, self.layers, self.embedding_dim) < 1 or self.max_blocks < 2:
            raise ValueError("Invalid metadata dimensions")
        if max(self.max_size, self.max_repetitions, self.max_blocks) > self.max_atoms:
            raise ValueError("Field bounds must not exceed the total-atom vocabulary")
        if self.max_hydrogens < 0:
            raise ValueError("Invalid hydrogen bound")


class MetadataGrammar(nn.Module):
    """Exact lookahead; every allowed prefix has at least one completion.

    Inorganic blocks are single core atoms (as in the training representation).
    At least one inorganic and one linker slot are required, ordered by role.
    Equal metadata tuples remain separate slots: their graphs can differ.
    State: remaining atoms, blocks after this one, slot, previous role,
    current role, current repetition, current explicit size.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        reachable = np.zeros((c.max_blocks + 1, c.max_atoms + 1), dtype=bool)
        reachable[0, 0] = True
        contributions = sorted({r * s for r in range(1, c.max_repetitions + 1)
                                for s in range(1, c.max_size + 1) if r*s <= c.max_atoms})
        for k in range(1, c.max_blocks + 1):
            for n in contributions:
                reachable[k, n:] |= reachable[k-1, :-n]
        # [role, future slots, remaining atoms, repetition]
        repeats = np.zeros((2, c.max_blocks, c.max_atoms + 1, c.max_repetitions + 1), bool)
        for k in range(c.max_blocks):
            for r in range(1, c.max_repetitions + 1):
                if k > 0:
                    repeats[0, k, r:, r] = reachable[k, :-r]
                for s in range(1, min(c.max_size, c.max_atoms // r) + 1):
                    n = r*s
                    repeats[1, k, n:, r] |= reachable[k, :-n]
        root = np.zeros((c.max_atoms + 1, c.max_blocks + 1), bool)
        for b in range(2, c.max_blocks + 1):
            root[:, b] = repeats[0, b-1].any(-1)
        self.register_buffer("reachable", torch.from_numpy(reachable), persistent=False)
        self.register_buffer("repeats", torch.from_numpy(repeats), persistent=False)
        self.register_buffer("root", torch.from_numpy(root), persistent=False)

    def allowed(self, fields, states):
        shape = fields.shape
        fields, states = fields.reshape(-1), states.reshape(-1, 7).long()
        c = self.config
        out = torch.zeros(len(fields), c.max_atoms + 1, dtype=torch.bool, device=fields.device)
        rem, after, slot, previous, role, rep, size = states.T
        idx = fields == TOTAL
        out[idx] = self.root.any(-1)
        idx = fields == BLOCKS
        out[idx, :c.max_blocks + 1] = self.root[rem[idx]]
        idx = fields == ROLE
        for value in (0, 1):
            legal = self.repeats[value, after[idx], rem[idx]].any(-1)
            legal &= (previous[idx] == 0) if value == 0 else (slot[idx] > 0)
            out[idx, value] = legal
        idx = fields == REPEAT
        out[idx, :c.max_repetitions + 1] = self.repeats[role[idx], after[idx], rem[idx]]
        idx = fields == SIZE
        sizes = torch.arange(c.max_size + 1, device=fields.device)
        remainder = rem[idx, None] - rep[idx, None] * sizes
        legal = (sizes > 0) & (remainder >= 0)
        legal &= self.reachable[after[idx, None], remainder.clamp(0, c.max_atoms)]
        legal &= (role[idx, None] == 1) | (sizes == 1)
        out[idx, :c.max_size + 1] = legal
        idx = fields == CORE
        cores = torch.arange(c.max_size + 1, device=fields.device)
        out[idx, :c.max_size + 1] = ((cores > 0) & (cores <= size[idx, None]) &
                                                  (cores * (c.max_hydrogens + 1) >= size[idx, None]))
        out[fields == PAD, 0] = True
        return out.reshape(*shape, -1)

    @staticmethod
    def advance(states, fields, values):
        states = states.clone()
        idx = fields == TOTAL
        states[idx, 0] = values[idx]
        idx = fields == BLOCKS
        states[idx, 1] = values[idx] - 1
        for field, column in ((ROLE, 4), (REPEAT, 5), (SIZE, 6)):
            idx = fields == field
            states[idx, column] = values[idx]
        idx = fields == CORE
        states[idx, 0] -= states[idx, 5] * states[idx, 6]
        states[idx, 1] -= 1
        states[idx, 2] += 1
        states[idx, 3] = states[idx, 4]
        return states


def validate_metadata(row, config):
    blocks = row["blocks"]
    if not 2 <= len(blocks) == row["num_blocks"] <= config.max_blocks:
        raise ValueError("Invalid number of unique block slots")
    roles = [bool(b["is_linker"]) for b in blocks]
    if roles != sorted(roles) or roles[0] or not roles[-1]:
        raise ValueError("Require inorganic slots followed by linker slots")
    for b in blocks:
        s, c, r = b["size"], b["core_size"], b["repetitions"]
        if any(type(x) is not int for x in (s, c, r)):
            raise ValueError("Sizes and repetitions must be integers")
        if not (1 <= c <= s <= config.max_size and s <= (config.max_hydrogens+1)*c
                and 1 <= r <= config.max_repetitions):
            raise ValueError("Block exceeds representation bounds")
        if not b["is_linker"] and (s != 1 or c != 1):
            raise ValueError("Inorganic blocks must be single atoms")
    if not 1 <= row["num_atoms"] <= config.max_atoms:
        raise ValueError("Total exceeds model capacity")
    if sum(b["size"] * b["repetitions"] for b in blocks) != row["num_atoms"]:
        raise ValueError("Explicit atom total differs from block accounting")
    if sum(b["core_size"] * b["repetitions"] for b in blocks) != row["core_num_atoms"]:
        raise ValueError("Core atom total differs from block accounting")


def encode_metadata(row, config):
    validate_metadata(row, config)
    tokens = [row["num_atoms"], row["num_blocks"]]
    for b in row["blocks"]:
        tokens += [int(b["is_linker"]), b["repetitions"], b["size"], b["core_size"]]
    return tokens


def teacher_batch(tokens):
    """Prefix states before each target; -1 padding never enters the model."""
    n, length = tokens.shape
    device = tokens.device
    states = torch.zeros(n, 7, dtype=torch.long, device=device)
    all_states, all_fields = [], []
    for t in range(length):
        field = t if t < 2 else ROLE + (t - 2) % 4
        fields = torch.full((n,), field, device=device, dtype=torch.long)
        fields = fields.masked_fill(tokens[:, t] < 0, PAD)
        all_states.append(states)
        all_fields.append(fields)
        states = MetadataGrammar.advance(states, fields, tokens[:, t])
    return torch.stack(all_fields, 1), torch.stack(all_states, 1)


class MetadataPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grammar = MetadataGrammar(config)
        self.value_embedding = nn.Embedding(config.max_atoms + 2, config.embedding_dim)
        self.field_embedding = nn.Embedding(7, 16)
        self.gru = nn.GRU(config.embedding_dim + 16 + 7, config.width,
                          config.layers, batch_first=True)
        self.output = nn.Linear(config.width, config.max_atoms + 1)
        self.register_buffer("state_scale", torch.tensor([
            config.max_atoms, config.max_blocks, config.max_blocks, 1, 1,
            config.max_repetitions, config.max_size], dtype=torch.float32), persistent=False)

    def forward(self, previous, fields, states, hidden=None):
        features = torch.cat([self.value_embedding(previous), self.field_embedding(fields),
                              states.to(self.state_scale.dtype) / self.state_scale], -1)
        features, hidden = self.gru(features, hidden)
        logits = self.output(features)
        allowed = self.grammar.allowed(fields, states)
        return logits.masked_fill(~allowed, -torch.inf), hidden

    def loss(self, tokens, *, details=False):
        fields, states = teacher_batch(tokens)
        previous = F.pad(tokens[:, :-1].clamp_min(0), (1, 0), value=self.config.max_atoms + 1)
        logits, _ = self(previous, fields, states)
        nll = F.cross_entropy(logits.transpose(1, 2), tokens, ignore_index=-1, reduction="none")
        active = tokens >= 0
        loss = nll.sum() / active.sum()
        if details:
            return loss, {"nll_sum": nll.sum().detach(), "tokens": active.sum(),
                          "structures": len(tokens), "field_nll": torch.stack([
                              nll[fields == f].sum().detach() for f in range(6)]),
                          "field_tokens": torch.stack([(fields == f).sum() for f in range(6)])}
        return loss

    @torch.no_grad()
    def sample(self, count, *, seed=9013, temperature=1.0):
        if count < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Positive count and temperature required")
        device = self.value_embedding.weight.device
        generator = torch.Generator(device=device).manual_seed(seed)
        states = torch.zeros(count, 7, dtype=torch.long, device=device)
        previous = torch.full((count, 1), self.config.max_atoms + 1, device=device, dtype=torch.long)
        rows, hidden = [], None
        lengths = torch.full((count,), 2 + 4*self.config.max_blocks, device=device)
        for t in range(2 + 4*self.config.max_blocks):
            field = t if t < 2 else ROLE + (t-2) % 4
            fields = torch.full((count,), field, device=device, dtype=torch.long)
            fields = fields.masked_fill(t >= lengths, PAD)
            logits, hidden = self(previous, fields[:, None], states[:, None], hidden)
            values = torch.multinomial((logits[:, 0] / temperature).softmax(-1), 1,
                                       generator=generator).flatten()
            if t == 1:
                lengths = 2 + 4*values
            rows.append(values)
            states = self.grammar.advance(states, fields, values)
            previous = values[:, None]
        result = []
        for row in torch.stack(rows, 1).cpu().tolist():
            blocks = [{"is_linker": bool(row[t]), "repetitions": row[t+1],
                       "size": row[t+2], "core_size": row[t+3]}
                      for t in range(2, 2+4*row[1], 4)]
            metadata = {"num_atoms": row[0], "num_blocks": row[1], "blocks": blocks,
                        "core_num_atoms": sum(b["core_size"]*b["repetitions"] for b in blocks)}
            validate_metadata(metadata, self.config)
            result.append(metadata)
        return result


def load_metadata_prior(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "flexmof_metadata_prior_v1":
        raise ValueError("Not a metadata prior checkpoint")
    model = MetadataPrior(MetadataConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def metadata_request(rows, device="cpu"):
    """Supply the same core sizes/repetitions/roles as a known-MOF request."""
    if not rows:
        raise ValueError("Empty metadata request")
    width = max(len(row["blocks"]) for row in rows)
    present = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    blocks = []
    for i, row in enumerate(rows):
        present[i, :len(row["blocks"])] = True
        blocks.extend(row["blocks"])
    tensor = lambda key: torch.tensor([b[key] for b in blocks], device=device)
    return {"shape": tuple(present.shape), "present": present.flatten(),
            "core_sizes": tensor("core_size").long(), "repetitions": tensor("repetitions").long(),
            "is_linker": tensor("is_linker").bool()}
