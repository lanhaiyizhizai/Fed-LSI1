"""Min-Max attack from 'Manipulating the Byzantine' (NDSS 2022)."""
import torch


def minmax_attack(benign_updates, n_attackers, device='cpu', dev_type='unit_vec'):
    """
    Min-Max attack: craft malicious updates that maximize the maximum distance
    between any benign update and the malicious one.

    Args:
        benign_updates: list of benign client model state_dicts or flat tensors
        n_attackers: number of attacker clients to generate
        device: torch device
        dev_type: deviation type ('unit_vec', 'sign', 'std')
    Returns:
        list of malicious state_dicts/tensors (length n_attackers)
    """
    # Flatten benign updates to tensors
    if isinstance(benign_updates[0], dict):
        flat_updates = []
        for w in benign_updates:
            flat = torch.cat([w[k].float().flatten() for k in sorted(w.keys())])
            flat_updates.append(flat)
    else:
        flat_updates = [u.float() for u in benign_updates]

    all_updates = torch.stack(flat_updates).to(device)
    mu = torch.mean(all_updates, dim=0).to(device)

    # Compute deviation direction
    if dev_type == 'unit_vec':
        deviation = mu / torch.norm(mu)
    elif dev_type == 'sign':
        deviation = torch.sign(mu)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, dim=0)
    else:
        deviation = mu / torch.norm(mu)

    # Binary search for optimal lambda
    lamda = torch.Tensor([10.0]).float().to(device)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0.0

    distances = []
    for update in all_updates:
        distance = torch.norm(all_updates - update, dim=1) ** 2
        distances.append(distance)
    distances = torch.stack(distances)
    max_distance = torch.max(distances)

    max_iter = 30
    for _ in range(max_iter):
        if torch.abs(lamda_succ - lamda) <= threshold_diff:
            break
        mal_update = (mu - lamda * deviation)
        distance = torch.norm(all_updates - mal_update, dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2
        lamda_fail = lamda_fail / 2

    mal_update = (mu - lamda_succ * deviation)

    # Return n_attackers copies
    if isinstance(benign_updates[0], dict):
        results = []
        keys = sorted(benign_updates[0].keys())
        shapes = [benign_updates[0][k].shape for k in keys]
        sizes = [s.numel() for s in shapes]
        for _ in range(n_attackers):
            w_mal = {}
            offset = 0
            for k, s, sz in zip(keys, shapes, sizes):
                w_mal[k] = mal_update[offset:offset + sz].view(s)
                offset += sz
            results.append(w_mal)
        return results
    else:
        return [mal_update] * n_attackers
