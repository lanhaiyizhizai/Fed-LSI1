"""FABA (Filtered Aggregation with Byzantine Attacker removal) defense."""
import copy
import torch
from collections import OrderedDict


def faba_aggregate(w_locals, f=None, device='cpu'):
    """
    FABA: iteratively remove the client furthest from the mean until
    n-f clients remain, then average.
    """
    n = len(w_locals)
    if f is None:
        f = n // 4
    f = min(f, n - 1)

    def flatten(w):
        return torch.cat([w[k].float().flatten() for k in sorted(w.keys())])

    indices = list(range(n))
    remaining = list(range(n))

    for _ in range(f):
        if len(remaining) <= 2:
            break
        flats = [flatten(w_locals[i]).to(device) for i in remaining]
        mean_flat = torch.mean(torch.stack(flats), dim=0)

        # Find furthest from mean
        max_dist = -1
        max_idx = 0
        for j, flat in enumerate(flats):
            dist = torch.norm(flat - mean_flat).item()
            if dist > max_dist:
                max_dist = dist
                max_idx = j

        remaining.pop(max_idx)

    # Average remaining
    w_avg = copy.deepcopy(w_locals[remaining[0]])
    for k in w_avg.keys():
        total = w_avg[k].float()
        for i in remaining[1:]:
            total += w_locals[i][k].float()
        w_avg[k] = (total / len(remaining)).to(device)

    return w_avg
