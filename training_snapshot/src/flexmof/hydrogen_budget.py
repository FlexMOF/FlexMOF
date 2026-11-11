"""Maximum-likelihood codec decoding with an exact explicit-hydrogen budget."""
import numpy as np
import torch


@torch.no_grad()
def decode_hydrogen_budget(atom_logits, hydrogen_logits, mask, totals, hydrogen_label):
    """Joint MAP atom/H labels subject to sum(H)=budget in each graph.

    Hydrogen core atoms cannot themselves carry attached hydrogens. Dynamic
    programming scores the codec probabilities; it never changes the budget.
    """
    if totals.shape != (mask.shape[0],) or atom_logits.shape[:2] != mask.shape:
        raise ValueError("Hydrogen budget shape mismatch")
    if hydrogen_logits.shape[:2] != mask.shape or totals.is_floating_point():
        raise ValueError("Hydrogen budgets must be integer per-block totals")
    atom_logp = atom_logits.log_softmax(-1).cpu().numpy()
    h_logp = hydrogen_logits.log_softmax(-1).cpu().numpy()
    if not np.isfinite(atom_logp).all() or not np.isfinite(h_logp).all():
        raise ValueError("Nonfinite codec logits")
    mask_cpu = mask.cpu().numpy()
    atoms = np.zeros(mask.shape, np.int64)
    counts = np.zeros(mask.shape, np.int64)
    for row, budget in enumerate(totals.cpu().tolist()):
        present = np.flatnonzero(mask_cpu[row])
        n, classes = len(present), h_logp.shape[-1]
        if not 0 <= budget <= n*(classes-1):
            raise ValueError(f"Infeasible hydrogen budget {budget} for {n} core nodes")
        a = atom_logp[row, present]
        best = a.argmax(-1)
        non_h = a.copy()
        non_h[:, hydrogen_label] = -np.inf
        best_non_h = non_h.argmax(-1)
        labels = np.broadcast_to(best_non_h[:, None], (n, classes)).copy()
        labels[:, 0] = best
        scores = h_logp[row, present] + np.take_along_axis(a, labels, axis=1)
        dp = np.full(budget+1, -np.inf)
        dp[0] = 0
        choices = np.zeros((n, budget+1), np.int64)
        for node in range(n):
            candidates = np.full((classes, budget+1), -np.inf)
            for h in range(min(classes, budget+1)):
                candidates[h, h:] = dp[:budget+1-h] + scores[node, h]
            choices[node] = candidates.argmax(0)
            dp = candidates.max(0)
        if not np.isfinite(dp[budget]):
            raise ValueError("Hydrogen budget has no supported assignment")
        remaining = budget
        for node in range(n-1, -1, -1):
            h = choices[node, remaining]
            counts[row, present[node]] = h
            atoms[row, present[node]] = labels[node, h]
            remaining -= h
        assert remaining == 0
    return (torch.as_tensor(atoms, device=atom_logits.device),
            torch.as_tensor(counts, device=atom_logits.device))
