"""FLTrust defense - properly implemented using model updates.

Reference: Cao et al., "FLTrust: Byzantine-robust Federated Learning
via Trust Bootstrapping" (NDSS 2021).

The server trains on a small root dataset to produce a trusted update,
then uses cosine similarity to weight client updates.
"""
import copy
import torch
from collections import OrderedDict


def compute_server_update(model_class, model_args, w_prev, root_loader,
                          device='cpu', lr=0.001, local_ep=1):
    """
    Compute the server's own update from a small root dataset.
    This is the trust anchor for FLTrust.

    Args:
        model_class: model class to instantiate
        model_args: tuple of args for model constructor
        w_prev: previous global model state_dict
        root_loader: DataLoader for server's root dataset
        device: compute device
        lr: learning rate
        local_ep: number of local epochs

    Returns:
        server_update: state_dict of the updated model
    """
    import torch.nn as nn

    server_model = model_class(*model_args).to(device)
    server_model.load_state_dict(copy.deepcopy(w_prev))
    server_model.train()

    optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(local_ep):
        for data, target in root_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(server_model(data), target)
            loss.backward()
            optimizer.step()

    return server_model.state_dict()


def compute_server_update_rec(model, w_prev, root_loader,
                               device='cpu', lr=0.001, local_ep=1):
    """Compute server update for recommendation models (MSE loss)."""
    import torch.nn as nn

    model.load_state_dict(copy.deepcopy(w_prev))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(local_ep):
        for users, items, ratings in root_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def fltrust(w_locals, server_update=None, device='cpu', prev_w=None):
    """
    FLTrust: use server's update as trust anchor.

    Key: operates on model UPDATES (w_local - w_prev), not raw weights.
    The server_update should be the server's own update computed from a
    small root dataset via compute_server_update().

    If server_update is None, falls back to FedAvg (no trust anchor available).
    """
    if prev_w is None and server_update is None:
        return _fedavg_fallback(w_locals, device)

    def flatten(w):
        return torch.cat([w[k].float().flatten() for k in sorted(w.keys())])

    # Compute client updates (delta = w_local - w_prev)
    if prev_w is not None:
        prev_flat = flatten(prev_w).to(device)
        client_updates = [flatten(w).to(device) - prev_flat for w in w_locals]
    else:
        client_updates = [flatten(w).to(device) for w in w_locals]

    # Server update (trust anchor) - must be provided by caller
    if server_update is not None:
        server_update_flat = flatten(server_update).to(device)
        # If server_update is a full state_dict, compute its delta
        if prev_w is not None:
            prev_flat_for_server = flatten(prev_w).to(device)
            server_update_flat = server_update_flat - prev_flat_for_server
    else:
        # No trust anchor - fall back to FedAvg
        return _fedavg_fallback(w_locals, device)

    server_norm = torch.norm(server_update_flat)
    if server_norm < 1e-10:
        return _fedavg_fallback(w_locals, device)

    trust_scores = []
    aligned_updates = []

    for update in client_updates:
        update_norm = torch.norm(update)
        if update_norm < 1e-10:
            trust_scores.append(0.0)
            aligned_updates.append(torch.zeros_like(update))
            continue

        cos_sim = torch.dot(update, server_update_flat) / (update_norm * server_norm)
        trust = max(cos_sim.item(), 0.0)  # ReLU
        trust_scores.append(trust)

        # Normalize client update to have same norm as server update
        aligned = update * (server_norm / update_norm)
        aligned_updates.append(aligned)

    total_trust = sum(trust_scores)
    if total_trust < 1e-10:
        return _fedavg_fallback(w_locals, device)

    # Weighted aggregation of aligned updates
    result_update = torch.zeros_like(client_updates[0])
    for aligned, trust in zip(aligned_updates, trust_scores):
        result_update += aligned * (trust / total_trust)

    # Unflatten: result = w_prev + aggregated_update
    if prev_w is not None:
        result = copy.deepcopy(prev_w)
        prev_flat_check = flatten(prev_w)
        result_flat = prev_flat_check.to(device) + result_update
    else:
        result = copy.deepcopy(w_locals[0])
        result_flat = result_update

    offset = 0
    for k in sorted(result.keys()):
        size = result[k].numel()
        result[k] = result_flat[offset:offset + size].view(result[k].shape).to(result[k].dtype)
        offset += size

    return result


def _fedavg_fallback(w_locals, device):
    w_avg = copy.deepcopy(w_locals[0])
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] = w_avg[k].float() + w_locals[i][k].float()
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg
