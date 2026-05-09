"""
Unified Federated Learning Experiment Runner
Supports: Image Classification (MNIST, Fashion-MNIST, CIFAR-10)
          and Recommendation Systems (Steam, Yelp, ML-10M, ML-20M, UserBehavior)

Attack methods: MSA, Gaussian, Min-Max, Min-Sum, ALIE, Label-Flip, IPM
Defense/Aggregation methods: FedAvg, Krum, Trimmed-Mean, FLTrust, Median, FABA
"""
import os
import sys
import copy
import random
import json
import time
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import MNISTCNN, FMNISTCNN, CIFAR10CNN
from models.resnet import resnet18_cifar
from attacks.msa import msa_attack
from attacks.gaussian import gaussian_attack
from attacks.minmax import minmax_attack
from attacks.minsum import minsum_attack
from attacks.alie import alie_attack
from attacks.ipm import ipm_attack
from defenses.fedavg import fedavg
from defenses.krum import krum_aggregate
from defenses.trimmed_mean import trimmed_mean
from defenses.fltrust import fltrust
from defenses.median import median_aggregate
from defenses.faba import faba_aggregate
from utils.data_loader import (
    load_image_dataset, dirichlet_partition, iid_partition
)
from utils.metrics import test_classification


def get_model(dataset_name, device='cpu'):
    if dataset_name == 'mnist':
        return MNISTCNN().to(device)
    elif dataset_name == 'fmnist':
        return FMNISTCNN().to(device)
    elif dataset_name == 'cifar10':
        return CIFAR10CNN().to(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def apply_attack(attack_name, w, benign_updates=None, round_idx=0,
                 n_attackers=0, n_total=0, device='cpu', dataset_name='',
                 source_label=1, target_label=0,
                 scale_range=0.02, clamp_std=2.5,
                 sigma_mult=12.0, alie_z=None, ipm_epsilon=1.0,
                 msa_shuffle=True, msa_blend=1.0, dev_type='unit_vec'):
    """Apply the specified attack to model weights."""
    if attack_name == 'no_attack':
        return w

    if attack_name == 'MSA':
        return msa_attack(w, round_idx=round_idx, device=device,
                          scale_range=scale_range, clamp_std=clamp_std,
                          shuffle=msa_shuffle, blend=msa_blend)

    elif attack_name == 'gaussian':
        return gaussian_attack(w, sigma_mult=sigma_mult, device=device)

    elif attack_name == 'label_flip':
        # Label flip is applied during training, not post-hoc
        return w

    # For attacks that need all benign updates (minmax, minsum, alie, ipm)
    elif attack_name in ['minmax', 'minsum', 'alie', 'ipm']:
        if benign_updates is None or len(benign_updates) == 0:
            return w
        if attack_name == 'minmax':
            results = minmax_attack(benign_updates, n_attackers, device=device,
                                    dev_type=dev_type)
            return results[0] if results else w
        elif attack_name == 'minsum':
            results = minsum_attack(benign_updates, n_attackers, device=device,
                                    dev_type=dev_type)
            return results[0] if results else w
        elif attack_name == 'alie':
            results = alie_attack(benign_updates, n_attackers, n_total,
                                  device=device, z_max=alie_z)
            return results[0] if results else w
        elif attack_name == 'ipm':
            results = ipm_attack(benign_updates, n_attackers,
                                 device=device, epsilon=ipm_epsilon)
            return results[0] if results else w

    return w


def apply_defense(defense_name, w_locals, f=None, device='cpu', server_update=None, prev_w=None):
    """Apply the specified defense/aggregation method."""
    if defense_name == 'fedavg':
        return fedavg(w_locals)
    elif defense_name == 'krum':
        return krum_aggregate(w_locals, f=f, device=device)
    elif defense_name == 'trimmed_mean':
        return trimmed_mean(w_locals, device=device)
    elif defense_name == 'fltrust':
        return fltrust(w_locals, server_update=server_update, device=device, prev_w=prev_w)
    elif defense_name == 'median':
        return median_aggregate(w_locals, device=device)
    elif defense_name == 'faba':
        return faba_aggregate(w_locals, f=f, device=device)
    else:
        return fedavg(w_locals)


def state_delta(local_w, global_w):
    """Return update delta = local - global for floating tensors."""
    delta = {}
    for key, value in local_w.items():
        if value.dtype.is_floating_point:
            delta[key] = value.detach().clone() - global_w[key].detach().clone()
        else:
            delta[key] = value.detach().clone()
    return delta


def apply_delta(global_w, delta_w):
    """Convert a malicious update delta back to model weights."""
    out = {}
    for key, value in global_w.items():
        if value.dtype.is_floating_point:
            out[key] = value.detach().clone() + delta_w[key].to(value.device).type_as(value)
        else:
            out[key] = value.detach().clone()
    return out


def scale_delta(delta_w, strength):
    """Scale floating-point update deltas for controlled attack-strength sweeps."""
    if strength == 1.0:
        return delta_w
    out = {}
    for key, value in delta_w.items():
        if value.dtype.is_floating_point:
            out[key] = value * strength
        else:
            out[key] = value
    return out


def local_train(model, train_loader, args, is_attacker=False):
    """Local training on a client."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = []
    for _ in range(args.local_ep):
        batch_loss = []
        for data, target in train_loader:
            data, target = data.to(args.device), target.to(args.device)

            # Label flip attack
            if is_attacker and args.attack == 'label_flip':
                target = args.target_label * torch.ones_like(target)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        if batch_loss:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / max(len(epoch_loss), 1)


def run_image_experiment(args):
    """Run federated learning image classification experiment."""
    logging.info(f"=" * 60)
    logging.info(f"Dataset: {args.dataset}, Attack: {args.attack}, "
                 f"Defense: {args.defense}, Compromised: {args.compromised_rate}")
    logging.info(f"=" * 60)

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    train_set, test_set = load_image_dataset(args.dataset, data_dir='./data')
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # Partition data
    if args.partition == 'iid':
        client_indices = iid_partition(train_set, args.num_clients, seed=args.seed)
    else:
        client_indices = dirichlet_partition(train_set, args.num_clients,
                                             alpha=args.alpha, seed=args.seed)

    # Initialize global model
    net_glob = get_model(args.dataset, device)
    w_glob = net_glob.state_dict()

    # Generate compromised clients
    n_compromised = int(args.compromised_rate * args.num_clients)
    if args.attack != 'no_attack' and n_compromised > 0:
        compromised = random.sample(range(args.num_clients), n_compromised)
    else:
        compromised = []
        n_compromised = 0

    logging.info(f"Clients: {args.num_clients}, Compromised: {n_compromised} ({compromised})")

    results = {'rounds': [], 'test_acc': [], 'test_loss': []}

    for round_idx in range(args.rounds):
        logging.info(f"\n--- Round {round_idx + 1}/{args.rounds} ---")

        w_locals = []
        benign_updates = []
        benign_deltas = []

        # Select clients for this round
        m = max(int(args.frac * args.num_clients), 1)
        selected = np.random.choice(range(args.num_clients), m, replace=False)

        attackers_selected = [i for i in selected if i in compromised]
        non_attackers = [i for i in selected if i not in compromised]

        # First collect benign updates
        for idx in non_attackers:
            if idx not in client_indices or len(client_indices[idx]) < args.batch_size:
                continue
            subset = Subset(train_set, client_indices[idx])
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            local_model = copy.deepcopy(net_glob)
            w, loss = local_train(local_model, loader, args, is_attacker=False)
            w_locals.append(w)
            benign_updates.append(copy.deepcopy(w))
            benign_deltas.append(state_delta(w, w_glob))

        # Then collect attacker updates
        if len(attackers_selected) > 0:
            # For omniscient attacks (minmax, minsum, alie, ipm), compute malicious updates
            if args.attack in ['minmax', 'minsum', 'alie', 'ipm']:
                n_atk = len(attackers_selected)
                if len(benign_deltas) == 0:
                    continue
                if args.attack == 'minmax':
                    mal_updates = minmax_attack(benign_deltas, n_atk, device=device,
                                                dev_type=args.dev_type)
                elif args.attack == 'minsum':
                    mal_updates = minsum_attack(benign_deltas, n_atk, device=device,
                                                dev_type=args.dev_type)
                elif args.attack == 'alie':
                    mal_updates = alie_attack(benign_deltas, n_atk,
                                              n_total=len(selected), device=device,
                                              z_max=args.alie_z)
                elif args.attack == 'ipm':
                    mal_updates = ipm_attack(benign_deltas, n_atk, device=device,
                                             epsilon=args.ipm_epsilon)
                else:
                    mal_updates = [state_delta(copy.deepcopy(w_glob), w_glob)] * n_atk

                for mal_w in mal_updates:
                    mal_w = scale_delta(mal_w, args.attack_strength)
                    w_locals.append(apply_delta(w_glob, mal_w))
            else:
                # Per-client attacks (MSA, gaussian, label_flip)
                for idx in attackers_selected:
                    if idx not in client_indices or len(client_indices[idx]) < args.batch_size:
                        continue
                    subset = Subset(train_set, client_indices[idx])
                    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
                    local_model = copy.deepcopy(net_glob)
                    w, loss = local_train(local_model, loader, args,
                                         is_attacker=(args.attack == 'label_flip'))

                    if args.attack == 'label_flip':
                        w_locals.append(w)
                    else:
                        w_attacked = apply_attack(
                            args.attack, w, benign_updates=None,
                            round_idx=round_idx, device=device,
                            scale_range=args.scale_range,
                            clamp_std=args.clamp_std,
                            sigma_mult=args.sigma_mult,
                            alie_z=args.alie_z,
                            ipm_epsilon=args.ipm_epsilon,
                            msa_shuffle=args.msa_shuffle,
                            msa_blend=args.msa_blend
                        )
                        w_locals.append(w_attacked)

        if len(w_locals) == 0:
            continue

        # Aggregate with defense
        # Count actual attackers in this round's participants
        n_atk_this_round = len([i for i in selected if i in compromised and i in client_indices
                                and len(client_indices[i]) >= args.batch_size])
        f = n_atk_this_round
        prev_w = copy.deepcopy(w_glob)  # Save previous global weights

        # For FLTrust: compute server update from root data
        server_update_w = None
        if args.defense == 'fltrust':
            from defenses.fltrust import compute_server_update
            root_loader = DataLoader(
                Subset(train_set, list(range(min(500, len(train_set))))),
                batch_size=args.batch_size, shuffle=True
            )
            model_fn = get_model(args.dataset, 'cpu')
            server_update_w = compute_server_update(
                type(model_fn), (), prev_w, root_loader,
                device=device, lr=args.lr, local_ep=1
            )

        w_glob = apply_defense(args.defense, w_locals, f=f, device=device,
                               server_update=server_update_w, prev_w=prev_w)
        net_glob.load_state_dict(w_glob)

        # Evaluate
        test_acc, test_loss = test_classification(net_glob, test_loader, device)
        results['rounds'].append(round_idx + 1)
        results['test_acc'].append(test_acc)
        results['test_loss'].append(test_loss)

        if (round_idx + 1) % args.eval_freq == 0 or round_idx == args.rounds - 1:
            logging.info(f"Round {round_idx + 1}: Test Acc={test_acc:.4f}, Loss={test_loss:.4f}")

    # Final results
    final_acc = results['test_acc'][-1]
    final_loss = results['test_loss'][-1]
    best_acc = max(results['test_acc'])
    best_idx = results['test_acc'].index(best_acc)

    logging.info(f"\n{'=' * 40}")
    logging.info(f"Final: Acc={final_acc:.4f}, Loss={final_loss:.4f}")
    logging.info(f"Best:  Acc={best_acc:.4f} at Round {results['rounds'][best_idx]}")
    logging.info(f"{'=' * 40}")

    return {
        'dataset': args.dataset,
        'attack': args.attack,
        'defense': args.defense,
        'compromised_rate': args.compromised_rate,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'best_acc': best_acc,
        'best_round': results['rounds'][best_idx],
        'all_accs': results['test_acc'],
        'all_losses': results['test_loss'],
        'partition': args.partition,
        'alpha': args.alpha,
        'num_clients': args.num_clients,
        'frac': args.frac,
        'rounds': args.rounds,
        'local_ep': args.local_ep,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'attack_params': {
            'scale_range': args.scale_range,
            'clamp_std': args.clamp_std,
            'msa_shuffle': args.msa_shuffle,
            'msa_blend': args.msa_blend,
            'sigma_mult': args.sigma_mult,
            'alie_z': args.alie_z,
            'ipm_epsilon': args.ipm_epsilon,
            'dev_type': args.dev_type,
            'attack_strength': args.attack_strength,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Attack/Defense Experiment')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'cifar10'])
    parser.add_argument('--attack', type=str, default='no_attack',
                        choices=['no_attack', 'MSA', 'gaussian', 'minmax', 'minsum',
                                 'alie', 'label_flip', 'ipm'])
    parser.add_argument('--defense', type=str, default='fedavg',
                        choices=['fedavg', 'krum', 'trimmed_mean', 'fltrust', 'median', 'faba'])
    parser.add_argument('--compromised_rate', type=float, default=0.2)
    parser.add_argument('--num_clients', type=int, default=50)
    parser.add_argument('--frac', type=float, default=0.2)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--partition', type=str, default='noniid', choices=['iid', 'noniid'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--scale_range', type=float, default=0.02)
    parser.add_argument('--clamp_std', type=float, default=2.5)
    parser.add_argument('--sigma_mult', type=float, default=12.0)
    parser.add_argument('--alie_z', type=float, default=None)
    parser.add_argument('--ipm_epsilon', type=float, default=1.0)
    parser.add_argument('--msa_shuffle', type=int, default=1,
                        help='MSA shuffle: 1=yes, 0=no')
    parser.add_argument('--msa_blend', type=float, default=1.0,
                        help='Interpolate MSA perturbation: w + blend * (msa(w)-w)')
    parser.add_argument('--attack_strength', type=float, default=1.0,
                        help='Multiplier applied to malicious update deltas for omniscient attacks')
    parser.add_argument('--dev_type', type=str, default='unit_vec',
                        choices=['unit_vec', 'sign', 'std'],
                        help='Deviation type for minmax/minsum attacks')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, default='results/image_results.json')

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.msa_shuffle = bool(args.msa_shuffle)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    result = run_image_experiment(args)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(f"\nResult: {args.dataset} | {args.attack} vs {args.defense} | "
          f"Acc={result['final_acc']:.4f} | Loss={result['final_loss']:.4f}")


if __name__ == '__main__':
    main()
