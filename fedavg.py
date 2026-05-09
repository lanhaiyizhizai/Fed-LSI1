"""FedAvg - Federated Averaging baseline."""
import copy
import torch


def fedavg(w_locals):
    """Standard federated averaging."""
    w_avg = copy.deepcopy(w_locals[0])
    for k in w_avg.keys():
        if w_avg[k].dtype in (torch.long, torch.int):
            continue  # Skip non-numeric buffers (num_batches_tracked, etc.)
        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k].float()
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg
