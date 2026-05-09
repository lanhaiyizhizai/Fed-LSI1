"""ALIE (A Little Is Enough) attack."""
import torch
import numpy as np
from scipy.stats import norm


def alie_attack(benign_updates, n_attackers, n_total, device='cpu', z_max=None):
    """
    ALIE attack: compute the minimal perturbation that can bypass coordinate-wise
    defenses by pushing updates just beyond the expected range.

    Args:
        benign_updates: list of benign client model state_dicts or flat tensors
        n_attackers: number of attacker clients
        n_total: total number of clients
        device: torch device
        z_max: optional override for z value
    Returns:
        list of malicious state_dicts/tensors (length n_attackers)
    """
    if isinstance(benign_updates[0], dict):
        flat_updates = []
        for w in benign_updates:
            flat = torch.cat([w[k].float().flatten() for k in sorted(w.keys())])
            flat_updates.append(flat)
    else:
        flat_updates = [u.float() for u in benign_updates]

    stacked = torch.stack(flat_updates).to(device)
    mu = torch.mean(stacked, dim=0)
    std = torch.std(stacked, dim=0)

    n_good = len(benign_updates)
    s = np.floor(n_total / 2 + 1) - n_attackers
    cdf_value = (n_good - s) / n_good
    cdf_value = max(cdf_value, 0.001)  # clamp
    computed_z = norm.ppf(cdf_value)
    z = z_max if z_max is not None else max(computed_z, 2.5)

    pert = std * z
    mal_flat = mu - pert

    if isinstance(benign_updates[0], dict):
        results = []
        keys = sorted(benign_updates[0].keys())
        shapes = [benign_updates[0][k].shape for k in keys]
        sizes = [s.numel() for s in shapes]
        for _ in range(n_attackers):
            w_mal = {}
            offset = 0
            for k, sh, sz in zip(keys, shapes, sizes):
                w_mal[k] = mal_flat[offset:offset + sz].view(sh)
                offset += sz
            results.append(w_mal)
        return results
    else:
        return [mal_flat] * n_attackers
