"""Inner Product Manipulation (IPM) attack."""
import torch
import numpy as np


def ipm_attack(benign_updates, n_attackers, device='cpu', epsilon=1.0):
    """
    IPM attack: manipulate the inner product between benign gradient and
    malicious gradient to derail the global model.

    Args:
        benign_updates: list of benign client model state_dicts or flat tensors
        n_attackers: number of attacker clients
        device: torch device
        epsilon: attack strength
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

    # IPM: push in the opposite direction of the mean
    mal_flat = -epsilon * mu

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
