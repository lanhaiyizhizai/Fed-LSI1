"""Gaussian noise attack."""
import copy
import numpy as np
import torch


def gaussian_attack(w, sigma=None, sigma_mult=12.0, device='cpu'):
    """Add Gaussian noise to model weight parameters (skip BN buffers)."""
    w_gaussian = copy.deepcopy(w)

    # Estimate std from weight parameters only (skip BN stats, int tensors)
    if sigma is None:
        all_params = []
        for key in w.keys():
            if w[key].dtype in (torch.long, torch.int):
                continue
            if 'running_mean' in key or 'running_var' in key or 'num_batches' in key:
                continue
            all_params.append(w[key].cpu().float().numpy().flatten())
        if all_params:
            all_params = np.concatenate(all_params)
            sigma = np.std(all_params) * sigma_mult

    for k in w.keys():
        if w[k].dtype in (torch.long, torch.int):
            continue
        if 'running_mean' in k or 'running_var' in k or 'num_batches' in k:
            continue
        noise = np.random.normal(0, sigma, w[k].size())
        noise = torch.from_numpy(noise).float().to(device)
        w_gaussian[k] = w[k].float() + noise

    return w_gaussian
