"""MSA (Matrix Singular-value Attack) - The user's own attack method."""
import copy
import random
import torch
import logging


def msa_attack(w, round_idx=0, device='cpu', scale_range=0.02, clamp_std=2.5,
               shuffle=True, blend=1.0):
    """
    MSA Attack: SVD-based singular value perturbation + optional shuffle.
    Uses adaptive clamp based on original per-parameter statistics.

    Args:
        w: model state_dict
        round_idx: current round number
        device: torch device
        scale_range: singular value perturbation half-range (e.g. 0.03 -> 0.97~1.03)
        clamp_std: clamp to ±clamp_std * original_param_std for each parameter
        shuffle: whether to shuffle singular values (default True)
        blend: interpolation strength between original and attacked weights
    """
    w_attacker = copy.deepcopy(w)
    target_layers = [k for k in w.keys() if 'weight' in k and len(w[k].shape) >= 2]

    for layer_name in target_layers:
        param = w_attacker[layer_name]
        original_shape = param.shape
        if len(original_shape) > 2:
            param_2d = param.view(original_shape[0], -1)
        else:
            param_2d = param

        try:
            U, S, Vh = torch.linalg.svd(param_2d.float(), full_matrices=False)
            k = S.size(0)

            # Singular value perturbation: small random scaling
            s_scale = torch.tensor([
                random.uniform(1.0, 1.0 + scale_range) if (i + round_idx) % 2 == 0
                else random.uniform(1.0 - scale_range, 1.0)
                for i in range(k)
            ]).to(device)
            S_attacked = S * s_scale

            # Global shuffle of singular values (optional)
            if shuffle:
                perm = torch.randperm(k).to(device)
                S_attacked = S_attacked[perm]

            # Reconstruct
            param_2d_attacked = U @ torch.diag(S_attacked) @ Vh
            w_attacker[layer_name] = param_2d_attacked.view(original_shape)

            # Bias scaling
            bias_name = layer_name.replace('.weight', '.bias')
            if bias_name in w_attacker:
                avg_scale = s_scale.mean()
                w_attacker[bias_name] = w_attacker[bias_name] * avg_scale

        except Exception as e:
            logging.warning(f"MSA: SVD failed for {layer_name}: {e}")
            continue

    # Adaptive clamp: per-parameter based on ORIGINAL (pre-attack) statistics
    for key in w_attacker.keys():
        if w_attacker[key].dtype in (torch.long, torch.int):
            continue
        if 'running_mean' in key or 'running_var' in key or 'num_batches' in key:
            continue
        param = w_attacker[key].float()
        orig_std = w[key].float().std().item()  # Use original std
        if orig_std > 1e-10:
            w_attacker[key] = torch.clamp(param, -clamp_std * orig_std, clamp_std * orig_std)

    if blend != 1.0:
        for key in w_attacker.keys():
            if not w_attacker[key].dtype.is_floating_point:
                continue
            w_attacker[key] = w[key].float() + blend * (w_attacker[key].float() - w[key].float())

    return w_attacker
