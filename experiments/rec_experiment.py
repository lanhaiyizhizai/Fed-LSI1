"""
Federated Recommendation System Experiment Runner
Supports: Steam, Yelp, ML-10M, ML-20M, UserBehavior
Metrics: HR@K, NDCG@K
"""
import os
import sys
import copy
import random
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rec_model import FedRecModel
from attacks.msa import msa_attack
from attacks.rec_targeted_msa import rec_targeted_msa, select_target_competitor_items, compute_promotion_metrics
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
    load_ml_dataset, load_steam_dataset, load_yelp_dataset,
    load_user_behavior_dataset, rec_partition, rec_dirichlet_partition,
    build_rec_test_data, RecDataset
)
from utils.metrics import (
    compute_rec_metrics, compute_rec_metrics_at_ks, generate_fixed_negatives
)


class RecTrainDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def state_delta(local_w, global_w):
    """Return model update delta = local - global for floating tensors."""
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


def local_train_rec(model, train_loader, args, is_attacker=False):
    """Local training for recommendation model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    epoch_loss = []
    for _ in range(args.local_ep):
        batch_loss = []
        for users, items, ratings in train_loader:
            users = users.to(args.device)
            items = items.to(args.device)
            ratings = ratings.to(args.device)

            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        if batch_loss:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return model.state_dict(), sum(epoch_loss) / max(len(epoch_loss), 1)


def run_rec_experiment(args):
    """Run federated recommendation experiment."""
    logging.info(f"=" * 60)
    logging.info(f"Rec Experiment: {args.dataset}, Attack: {args.attack}, "
                 f"Defense: {args.defense}, Compromised: {args.compromised_rate}")
    logging.info(f"=" * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device

    # Load dataset from pre-processed CSV files
    csv_map = {
        'steam': 'data/rec/steam.csv',
        'yelp': 'data/rec/yelp.csv',
        'ml-100k': 'data/rec/ml-100k.csv',
        'ml-1m': 'data/rec/ml-1m.csv',
        'ml-10m': 'data/rec/ml-10m.csv',
        'ml-20m': 'data/rec/ml-20m.csv',
        'user_behavior': 'data/rec/user_behavior.csv',
    }
    csv_path = csv_map.get(args.dataset)
    if csv_path and os.path.exists(csv_path) and not getattr(args, 'prefer_raw_loader', False):
        logging.info(f"Loading {args.dataset} from {csv_path}")
        df = pd.read_csv(csv_path)
        df = df[['user', 'item', 'rating']].dropna()
        num_users = df['user'].nunique()
        num_items = df['item'].nunique()
    else:
        # Fallback to original loaders
        if args.dataset == 'steam':
            df, num_users, num_items = load_steam_dataset(args.data_dir)
        elif args.dataset == 'yelp':
            df, num_users, num_items = load_yelp_dataset(args.data_dir)
        elif args.dataset.startswith('ml-'):
            df, num_users, num_items = load_ml_dataset(args.data_dir, args.dataset)
        elif args.dataset == 'user_behavior':
            df, num_users, num_items = load_user_behavior_dataset(args.data_dir)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    logging.info(f"Data: {len(df)} interactions, {num_users} users, {num_items} items")

    # Clip to manageable size for large datasets
    max_interactions = getattr(args, 'max_interactions', 500000)
    if len(df) > max_interactions:
        df = df.sample(max_interactions, random_state=args.seed).reset_index(drop=True)

    # Re-encode after sampling
    le_user = LabelEncoder()
    le_item = LabelEncoder()
    df['user'] = le_user.fit_transform(df['user'])
    df['item'] = le_item.fit_transform(df['item'])
    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    # Train/test split
    train_idx, test_idx = build_rec_test_data(df, test_ratio=0.2, seed=args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Client partition
    if args.partition == 'user_dirichlet':
        client_indices = rec_dirichlet_partition(
            train_df, args.num_clients, alpha=args.alpha, seed=args.seed
        )
    else:
        client_indices = rec_partition(train_df, args.num_clients, seed=args.seed)

    non_empty_clients = sum(1 for rows in client_indices.values() if len(rows) > 0)
    logging.info(f"Partition={args.partition}, alpha={args.alpha}, "
                 f"non-empty clients={non_empty_clients}/{args.num_clients}")

    # Test dataset
    test_dataset = RecDataset(
        test_df['user'].values, test_df['item'].values, test_df['rating'].values
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_ks = sorted({int(k) for k in getattr(args, 'eval_ks', [args.K])})
    primary_k = args.K

    # Generate fixed negatives for reproducible evaluation
    fixed_negs = generate_fixed_negatives(
        test_loader, num_items, n_neg=max(99, max(eval_ks) - 1), seed=args.seed
    )

    # Initialize model
    net_glob = FedRecModel(num_users, num_items, embed_dim=args.embed_dim).to(device)
    w_glob = net_glob.state_dict()

    # Compromised clients
    n_compromised = int(args.compromised_rate * args.num_clients)
    if args.attack != 'no_attack' and n_compromised > 0:
        compromised = random.sample(range(args.num_clients), n_compromised)
    else:
        compromised = []
        n_compromised = 0

    results = {'rounds': [], 'hr_k': [], 'ndcg_k': [], 'hr_at_k': {}, 'ndcg_at_k': {}}
    for k in eval_ks:
        results['hr_at_k'][str(k)] = []
        results['ndcg_at_k'][str(k)] = []

    # For targeted rec attack: select target/competitor items
    target_items = None
    competitor_items = None
    if args.attack == 'targeted_msa':
        target_items, competitor_items = select_target_competitor_items(
            num_items, n_target=5, n_competitor=10, seed=args.seed
        )
        logging.info(f"Target items: {target_items}, Competitor items: {competitor_items}")

    for round_idx in range(args.rounds):
        logging.info(f"\n--- Round {round_idx + 1}/{args.rounds} ---")

        w_locals = []
        benign_updates = []
        benign_deltas = []

        m = max(int(args.frac * args.num_clients), 1)
        selected = np.random.choice(range(args.num_clients), m, replace=False)
        attackers_selected = [i for i in selected if i in compromised]
        non_attackers = [i for i in selected if i not in compromised]

        # Benign clients
        for idx in non_attackers:
            if idx not in client_indices or len(client_indices[idx]) == 0:
                continue
            subset = RecDataset(
                train_df.iloc[client_indices[idx]]['user'].values,
                train_df.iloc[client_indices[idx]]['item'].values,
                train_df.iloc[client_indices[idx]]['rating'].values
            )
            if len(subset) < 2:
                continue
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
            local_model = FedRecModel(num_users, num_items, embed_dim=args.embed_dim).to(device)
            local_model.load_state_dict(copy.deepcopy(w_glob))
            w, loss = local_train_rec(local_model, loader, args)
            w_locals.append(w)
            benign_updates.append(copy.deepcopy(w))
            benign_deltas.append(state_delta(w, w_glob))

        # Attacker clients
        if len(attackers_selected) > 0:
            if args.attack in ['minmax', 'minsum', 'alie', 'ipm']:
                n_atk = len(attackers_selected)
                if args.attack == 'minmax':
                    mal_updates = minmax_attack(benign_deltas, n_atk, device=device,
                                                dev_type=getattr(args, 'dev_type', 'unit_vec'))
                elif args.attack == 'minsum':
                    mal_updates = minsum_attack(benign_deltas, n_atk, device=device,
                                                dev_type=getattr(args, 'dev_type', 'unit_vec'))
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
            elif args.attack in ['MSA', 'targeted_msa']:
                for idx in attackers_selected:
                    if idx not in client_indices or len(client_indices[idx]) == 0:
                        continue
                    subset = RecDataset(
                        train_df.iloc[client_indices[idx]]['user'].values,
                        train_df.iloc[client_indices[idx]]['item'].values,
                        train_df.iloc[client_indices[idx]]['rating'].values
                    )
                    if len(subset) < 2:
                        continue
                    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
                    local_model = FedRecModel(num_users, num_items, embed_dim=args.embed_dim).to(device)
                    local_model.load_state_dict(copy.deepcopy(w_glob))
                    w, loss = local_train_rec(local_model, loader, args)
                    if args.attack == 'targeted_msa':
                        w_attacked = rec_targeted_msa(
                            w, target_items=target_items,
                            competitor_items=competitor_items,
                            num_items=num_items,
                            round_idx=round_idx, device=device,
                            ablation_mode=getattr(args, 'ablation_mode', None)
                        )
                    else:
                        w_attacked = msa_attack(w, round_idx=round_idx, device=device,
                                                scale_range=args.scale_range,
                                                clamp_std=args.clamp_std,
                                                shuffle=args.msa_shuffle,
                                                blend=args.msa_blend)
                    w_locals.append(w_attacked)
            elif args.attack == 'gaussian':
                for idx in attackers_selected:
                    if idx not in client_indices or len(client_indices[idx]) == 0:
                        continue
                    subset = RecDataset(
                        train_df.iloc[client_indices[idx]]['user'].values,
                        train_df.iloc[client_indices[idx]]['item'].values,
                        train_df.iloc[client_indices[idx]]['rating'].values
                    )
                    if len(subset) < 2:
                        continue
                    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
                    local_model = FedRecModel(num_users, num_items, embed_dim=args.embed_dim).to(device)
                    local_model.load_state_dict(copy.deepcopy(w_glob))
                    w, loss = local_train_rec(local_model, loader, args)
                    w_attacked = gaussian_attack(w, device=device,
                                                 sigma_mult=args.sigma_mult)
                    w_locals.append(w_attacked)

        if len(w_locals) == 0:
            continue

        # Aggregate
        # Count actual attackers in this round's participants
        n_atk_this_round = len(attackers_selected)
        f = n_atk_this_round
        prev_w = copy.deepcopy(w_glob)

        if args.defense == 'fedavg':
            w_glob = fedavg(w_locals)
        elif args.defense == 'krum':
            w_glob = krum_aggregate(w_locals, f=f, device=device)
        elif args.defense == 'trimmed_mean':
            w_glob = trimmed_mean(w_locals, device=device)
        elif args.defense == 'median':
            w_glob = median_aggregate(w_locals, device=device)
        elif args.defense == 'faba':
            w_glob = faba_aggregate(w_locals, f=f, device=device)
        elif args.defense == 'fltrust':
            # Compute server update from root data
            from defenses.fltrust import compute_server_update_rec
            root_size = min(args.fltrust_root_size, len(train_df))
            root_seed = args.seed + round_idx + args.fltrust_root_seed_offset
            if args.fltrust_root_random:
                root_indices = np.random.RandomState(root_seed).choice(
                    len(train_df), size=root_size, replace=False
                )
                root_df = train_df.iloc[root_indices]
            else:
                root_df = train_df.iloc[:root_size]
            root_subset = RecDataset(
                root_df['user'].values,
                root_df['item'].values,
                root_df['rating'].values
            )
            root_loader = DataLoader(root_subset, batch_size=args.batch_size, shuffle=True)
            root_model = FedRecModel(num_users, num_items, embed_dim=args.embed_dim).to(device)
            server_update_w = compute_server_update_rec(
                root_model, prev_w, root_loader, device=device,
                lr=args.fltrust_server_lr, local_ep=args.fltrust_server_ep
            )
            w_glob = fltrust(w_locals, server_update=server_update_w,
                             device=device, prev_w=prev_w)
        else:
            w_glob = fedavg(w_locals)

        net_glob.load_state_dict(w_glob)

        # Evaluate
        if (round_idx + 1) % args.eval_freq == 0 or round_idx == args.rounds - 1:
            if len(eval_ks) == 1:
                hr, ndcg = compute_rec_metrics(net_glob, test_loader, K=primary_k,
                                               device=device, num_items=num_items,
                                               fixed_negs=fixed_negs)
                metrics_at_k = {primary_k: {'hr': hr, 'ndcg': ndcg}}
            else:
                metrics_at_k = compute_rec_metrics_at_ks(
                    net_glob, test_loader, Ks=eval_ks, device=device,
                    num_items=num_items, fixed_negs=fixed_negs
                )
                hr = metrics_at_k[primary_k]['hr']
                ndcg = metrics_at_k[primary_k]['ndcg']

            results['rounds'].append(round_idx + 1)
            results['hr_k'].append(hr)
            results['ndcg_k'].append(ndcg)
            for k in eval_ks:
                results['hr_at_k'][str(k)].append(metrics_at_k[k]['hr'])
                results['ndcg_at_k'][str(k)].append(metrics_at_k[k]['ndcg'])

            # Promotion metrics for targeted_msa
            if args.attack == 'targeted_msa' and target_items is not None:
                prom = compute_promotion_metrics(
                    net_glob, target_items, competitor_items,
                    test_loader, num_items, K=args.K, device=device
                )
                results.setdefault('avg_target_rank', []).append(prom['avg_target_rank'])
                results.setdefault('avg_competitor_rank', []).append(prom['avg_competitor_rank'])
                results.setdefault('rank_promotion', []).append(prom['rank_promotion_ratio'])
                results.setdefault('avg_target_score', []).append(prom['avg_target_score'])
                results.setdefault('score_uplift', []).append(prom['score_uplift_ratio'])
                logging.info(f"Round {round_idx + 1}: HR@{primary_k}={hr:.4f}, NDCG@{primary_k}={ndcg:.4f}, "
                           f"TargetRank={prom['avg_target_rank']:.1f}, CompRank={prom['avg_competitor_rank']:.1f}, "
                           f"RankPromo={prom['rank_promotion_ratio']:.2f}x")
            else:
                msg = f"Round {round_idx + 1}: HR@{primary_k}={hr:.4f}, NDCG@{primary_k}={ndcg:.4f}"
                if len(eval_ks) > 1:
                    extras = [
                        f"K={k}:HR={metrics_at_k[k]['hr']:.4f},NDCG={metrics_at_k[k]['ndcg']:.4f}"
                        for k in eval_ks
                    ]
                    msg += " | " + "; ".join(extras)
                logging.info(msg)

    # Final results
    final_result = {
        'dataset': args.dataset,
        'attack': args.attack,
        'defense': args.defense,
        'compromised_rate': args.compromised_rate,
        'K': primary_k,
        'eval_ks': eval_ks,
        'partition': args.partition,
        'alpha': args.alpha,
        'num_clients': args.num_clients,
        'frac': args.frac,
        'rounds': args.rounds,
        'local_ep': args.local_ep,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'embed_dim': args.embed_dim,
        'max_interactions': args.max_interactions,
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
            'fltrust_root_size': args.fltrust_root_size,
            'fltrust_root_random': bool(args.fltrust_root_random),
            'fltrust_root_seed_offset': args.fltrust_root_seed_offset,
            'fltrust_server_lr': args.fltrust_server_lr,
            'fltrust_server_ep': args.fltrust_server_ep,
        },
        'ablation_mode': getattr(args, 'ablation_mode', None),
        'final_hr': results['hr_k'][-1] if results['hr_k'] else 0,
        'final_ndcg': results['ndcg_k'][-1] if results['ndcg_k'] else 0,
        'best_hr': max(results['hr_k']) if results['hr_k'] else 0,
        'best_ndcg': max(results['ndcg_k']) if results['ndcg_k'] else 0,
        'all_hrs': results['hr_k'],
        'all_ndcgs': results['ndcg_k'],
        'final_hr_at_k': {
            k: vals[-1] if vals else 0 for k, vals in results['hr_at_k'].items()
        },
        'final_ndcg_at_k': {
            k: vals[-1] if vals else 0 for k, vals in results['ndcg_at_k'].items()
        },
        'all_hr_at_k': results['hr_at_k'],
        'all_ndcg_at_k': results['ndcg_at_k'],
    }

    # Add promotion metrics if targeted
    if args.attack == 'targeted_msa' and 'rank_promotion' in results:
        final_result['final_avg_target_rank'] = results['avg_target_rank'][-1]
        final_result['final_avg_competitor_rank'] = results['avg_competitor_rank'][-1]
        final_result['final_rank_promotion'] = results['rank_promotion'][-1]
        final_result['final_avg_target_score'] = results['avg_target_score'][-1]
        final_result['final_score_uplift'] = results['score_uplift'][-1]
        logging.info(f"\nFinal: HR@{primary_k}={final_result['final_hr']:.4f}, "
                     f"NDCG@{primary_k}={final_result['final_ndcg']:.4f}, "
                     f"TargetRank={final_result['final_avg_target_rank']:.1f}, "
                     f"CompRank={final_result['final_avg_competitor_rank']:.1f}, "
                     f"RankPromo={final_result['final_rank_promotion']:.2f}x")
    else:
        logging.info(f"\nFinal: HR@{primary_k}={final_result['final_hr']:.4f}, "
                     f"NDCG@{primary_k}={final_result['final_ndcg']:.4f}")

    return final_result


def main():
    parser = argparse.ArgumentParser(description='Federated Recommendation Experiment')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        choices=['steam', 'yelp', 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'user_behavior'])
    parser.add_argument('--attack', type=str, default='no_attack',
                        choices=['no_attack', 'MSA', 'targeted_msa', 'gaussian', 'minmax', 'minsum',
                                 'alie', 'ipm'])
    parser.add_argument('--defense', type=str, default='fedavg',
                        choices=['fedavg', 'krum', 'trimmed_mean', 'fltrust', 'median', 'faba'])
    parser.add_argument('--compromised_rate', type=float, default=0.2)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--frac', type=float, default=0.5)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--local_ep', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--Ks', type=str, default=None,
                        help='Comma-separated K values for HR/NDCG, e.g. 5,10,20')
    parser.add_argument('--partition', type=str, default='user_iid',
                        choices=['user_iid', 'user_dirichlet'],
                        help='Recommendation client partitioning protocol')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for user_dirichlet partition')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='results/rec_results.json')
    parser.add_argument('--max_interactions', type=int, default=500000)
    parser.add_argument('--prefer_raw_loader', type=int, default=0,
                        help='Prefer dataset-specific raw loader instead of cached CSV when available')
    parser.add_argument('--ablation_mode', type=str, default=None,
                        choices=[None, 'embedding_only', 'fc_svd_only'])
    parser.add_argument('--scale_range', type=float, default=0.02,
                        help='MSA scale range for singular value perturbation')
    parser.add_argument('--clamp_std', type=float, default=2.5,
                        help='MSA adaptive clamp multiplier')
    parser.add_argument('--msa_shuffle', type=int, default=0,
                        help='MSA shuffle: 1=yes, 0=no')
    parser.add_argument('--msa_blend', type=float, default=1.0,
                        help='Interpolate MSA perturbation: w + blend * (msa(w)-w)')
    parser.add_argument('--sigma_mult', type=float, default=12.0,
                        help='Gaussian noise multiplier')
    parser.add_argument('--alie_z', type=float, default=None,
                        help='ALIE z_max parameter')
    parser.add_argument('--ipm_epsilon', type=float, default=1.0,
                        help='IPM epsilon parameter')
    parser.add_argument('--dev_type', type=str, default='unit_vec',
                        choices=['unit_vec', 'sign', 'std'],
                        help='Deviation type for minmax/minsum attacks')
    parser.add_argument('--attack_strength', type=float, default=1.0,
                        help='Multiplier applied to malicious update deltas for non-MSA omniscient attacks')
    parser.add_argument('--fltrust_root_size', type=int, default=500,
                        help='Number of root interactions for FLTrust server update')
    parser.add_argument('--fltrust_root_random', type=int, default=1,
                        help='Whether to randomly sample FLTrust root interactions each round')
    parser.add_argument('--fltrust_root_seed_offset', type=int, default=1000,
                        help='Seed offset used for FLTrust root sampling')
    parser.add_argument('--fltrust_server_lr', type=float, default=0.001,
                        help='Server learning rate for FLTrust root update')
    parser.add_argument('--fltrust_server_ep', type=int, default=1,
                        help='Server local epochs for FLTrust root update')

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.msa_shuffle = bool(args.msa_shuffle)
    args.prefer_raw_loader = bool(args.prefer_raw_loader)
    args.fltrust_root_random = bool(args.fltrust_root_random)
    if args.Ks:
        args.eval_ks = [int(x.strip()) for x in args.Ks.split(',') if x.strip()]
        if args.K not in args.eval_ks:
            args.eval_ks.append(args.K)
    else:
        args.eval_ks = [args.K]

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    result = run_rec_experiment(args)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'a') as f:
        f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
