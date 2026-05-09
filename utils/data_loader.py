"""Data loading and partitioning utilities for all datasets."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


# ==============================================================================
# Image Classification Datasets
# ==============================================================================

def get_image_transforms(dataset_name):
    if dataset_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'fmnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    elif dataset_name == 'cifar10':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    return transforms.ToTensor()


def get_test_transforms(dataset_name):
    if dataset_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'fmnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    elif dataset_name == 'cifar10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    return transforms.ToTensor()


def load_image_dataset(dataset_name, data_dir='./data'):
    """Load image classification datasets."""
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == 'mnist':
        train_transform = get_image_transforms('mnist')
        test_transform = get_test_transforms('mnist')
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)
    elif dataset_name == 'fmnist':
        train_transform = get_image_transforms('fmnist')
        test_transform = get_test_transforms('fmnist')
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_transform)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transform)
    elif dataset_name == 'cifar10':
        train_transform = get_image_transforms('cifar10')
        test_transform = get_test_transforms('cifar10')
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unknown image dataset: {dataset_name}")

    return train_set, test_set


def dirichlet_partition(train_set, num_clients, alpha=0.5, seed=42):
    """Non-IID partition using Dirichlet distribution."""
    np.random.seed(seed)
    targets = np.array([train_set[i][1] for i in range(len(train_set))])
    num_classes = len(np.unique(targets))

    client_indices = {i: [] for i in range(num_clients)}

    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        splits = (proportions * len(idx_k)).astype(int)
        # Adjust last split to include all remaining
        splits[-1] = len(idx_k) - splits[:-1].sum()

        start = 0
        for i in range(num_clients):
            client_indices[i].extend(idx_k[start:start + splits[i]].tolist())
            start += splits[i]

    return client_indices


def iid_partition(train_set, num_clients, seed=42):
    """IID partition: randomly split data among clients."""
    np.random.seed(seed)
    num_items = len(train_set)
    indices = np.random.permutation(num_items)
    split_size = num_items // num_clients
    client_indices = {}
    for i in range(num_clients):
        start = i * split_size
        if i == num_clients - 1:
            client_indices[i] = indices[start:].tolist()
        else:
            client_indices[i] = indices[start:start + split_size].tolist()
    return client_indices


# ==============================================================================
# Recommendation System Datasets
# ==============================================================================

class RecDataset(Dataset):
    """Generic recommendation dataset (user, item, rating)."""

    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class InterRecDataset(Dataset):
    """Interaction-based recommendation dataset for implicit feedback."""

    def __init__(self, users, items):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]


def load_ml_dataset(data_dir, dataset_name='ml-1m'):
    """Load MovieLens dataset (ml-100k, ml-1m, ml-10m, ml-20m)."""
    file_map = {
        'ml-100k': 'u.data',
        'ml-1m': 'ratings.dat',
        'ml-10m': 'ratings.dat',
        'ml-20m': 'ratings.csv',
    }

    path = os.path.join(data_dir, dataset_name, file_map.get(dataset_name, 'ratings.dat'))

    if not os.path.exists(path):
        print(f"Dataset not found at {path}. Attempting to download...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _download_ml_dataset(data_dir, dataset_name)

    if dataset_name == 'ml-100k':
        df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    elif dataset_name == 'ml-20m':
        df = pd.read_csv(path)
        df.columns = ['user', 'item', 'rating', 'timestamp']
    else:
        df = pd.read_csv(path, sep='::', names=['user', 'item', 'rating', 'timestamp'],
                         engine='python', encoding='latin-1')

    return _process_rec_df(df)


def load_steam_dataset(data_dir):
    """Load Steam game dataset."""
    path = os.path.join(data_dir, 'steam', 'steam.csv')
    if not os.path.exists(path):
        # Try alternative paths
        alt = os.path.join(data_dir, 'steam.csv')
        if os.path.exists(alt):
            path = alt
        else:
            print(f"Steam dataset not found. Creating synthetic data...")
            return _create_synthetic_rec_data(10000, 5000, 500000)

    df = pd.read_csv(path)
    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if 'user' in cl:
            col_map[col] = 'user'
        elif 'item' in cl or 'game' in cl or 'appid' in cl or 'product' in cl:
            col_map[col] = 'item'
        elif 'rating' in cl or 'score' in cl or 'play' in cl or 'recommend' in cl:
            col_map[col] = 'rating'
    df = df.rename(columns=col_map)

    if 'rating' not in df.columns:
        df['rating'] = 1.0  # implicit feedback

    return _process_rec_df(df[['user', 'item', 'rating']].dropna())


def load_yelp_dataset(data_dir):
    """Load Yelp review dataset."""
    path = os.path.join(data_dir, 'yelp', 'yelp_academic_dataset_review.json')
    if not os.path.exists(path):
        alt = os.path.join(data_dir, 'yelp_review.csv')
        if os.path.exists(alt):
            df = pd.read_csv(alt)
        else:
            print(f"Yelp dataset not found. Creating synthetic data...")
            return _create_synthetic_rec_data(50000, 20000, 2000000)
    else:
        import json
        records = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 500000:  # limit for memory
                    break
                obj = json.loads(line)
                records.append({'user': obj['user_id'], 'item': obj['business_id'], 'rating': obj['stars']})
        df = pd.DataFrame(records)

    return _process_rec_df(df[['user', 'item', 'rating']].dropna())


def load_user_behavior_dataset(csv_path='UserBehavior.csv'):
    """Load Taobao UserBehavior dataset."""
    if not os.path.exists(csv_path):
        print(f"UserBehavior.csv not found at {csv_path}")
        return _create_synthetic_rec_data(100000, 50000, 5000000)

    df = pd.read_csv(csv_path, header=None,
                     names=['user_id', 'item_id', 'cat_id', 'behavior_type', 'timestamp'])

    # Encode behavior types to implicit ratings
    beh_map = {'pv': 1, 'cart': 2, 'fav': 3, 'buy': 4}
    if df['behavior_type'].dtype == object:
        df['rating'] = df['behavior_type'].map(beh_map).fillna(1)
    else:
        df['rating'] = df['behavior_type']

    df = df[['user_id', 'item_id', 'rating']].copy()
    df.columns = ['user', 'item', 'rating']
    return _process_rec_df(df.dropna().sample(min(len(df), 2000000), random_state=42))


def _process_rec_df(df):
    """Process recommendation DataFrame: encode IDs, filter, return stats."""
    le_user = LabelEncoder()
    le_item = LabelEncoder()

    df['user'] = le_user.fit_transform(df['user'].astype(str))
    df['item'] = le_item.fit_transform(df['item'].astype(str))

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    return df, num_users, num_items


def _create_synthetic_rec_data(num_users, num_items, num_interactions):
    """Create synthetic recommendation data for testing."""
    np.random.seed(42)
    users = np.random.randint(0, num_users, num_interactions)
    items = np.random.randint(0, num_items, num_interactions)
    ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], num_interactions)
    df = pd.DataFrame({'user': users, 'item': items, 'rating': ratings})
    return df, num_users, num_items


def _download_ml_dataset(data_dir, dataset_name):
    """Download MovieLens datasets."""
    import urllib.request
    import zipfile

    urls = {
        'ml-100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
        'ml-20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip',
    }

    if dataset_name not in urls:
        raise ValueError(f"Cannot download {dataset_name}")

    url = urls[dataset_name]
    zip_path = os.path.join(data_dir, f'{dataset_name}.zip')

    print(f"Downloading {dataset_name}...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    os.remove(zip_path)
    print(f"Downloaded and extracted {dataset_name}")


def rec_partition(df, num_clients, min_interactions=5, seed=42):
    """Partition recommendation data by assigning user groups across all clients."""
    np.random.seed(seed)

    user_counts = df['user'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index.values

    if len(valid_users) < num_clients:
        # Allow users with fewer interactions
        valid_users = user_counts.index.values

    user_groups = df.groupby('user').groups
    shuffled_users = np.array(valid_users, copy=True)
    np.random.shuffle(shuffled_users)

    client_indices = {i: [] for i in range(num_clients)}
    client_loads = np.zeros(num_clients, dtype=np.int64)

    # Greedy load balancing keeps user histories intact while avoiding the old
    # "one selected user per client" protocol.
    for uid in sorted(shuffled_users, key=lambda u: len(user_groups[u]), reverse=True):
        cid = int(np.argmin(client_loads))
        rows = list(user_groups[uid])
        client_indices[cid].extend(rows)
        client_loads[cid] += len(rows)

    return client_indices


def rec_dirichlet_partition(df, num_clients, alpha=0.5, min_interactions=5,
                            seed=42):
    """
    Non-IID recommendation partition using Dirichlet allocation over user groups.

    Users are first binned by their dominant/highest-frequency item. Whole user
    histories are then assigned to clients, so train/test leakage through split
    user histories is avoided while client item distributions become non-IID.
    """
    rng = np.random.RandomState(seed)
    user_counts = df['user'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index.values

    if len(valid_users) < num_clients:
        valid_users = user_counts.index.values

    user_groups = df.groupby('user').groups
    user_item_bins = {}
    for uid in valid_users:
        items = df.loc[user_groups[uid], 'item'].values
        if len(items) == 0:
            continue
        vals, counts = np.unique(items, return_counts=True)
        user_item_bins.setdefault(vals[np.argmax(counts)], []).append(uid)

    client_indices = {i: [] for i in range(num_clients)}
    client_loads = np.zeros(num_clients, dtype=np.int64)

    for users in user_item_bins.values():
        users = np.array(users, copy=True)
        rng.shuffle(users)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        raw_splits = np.floor(proportions * len(users)).astype(int)
        raw_splits[-1] += len(users) - raw_splits.sum()
        rng.shuffle(raw_splits)

        start = 0
        for cid, count in enumerate(raw_splits):
            if count <= 0:
                continue
            assigned_users = users[start:start + count]
            start += count
            for uid in assigned_users:
                rows = list(user_groups[uid])
                client_indices[cid].extend(rows)
                client_loads[cid] += len(rows)

    empty_clients = [cid for cid, rows in client_indices.items() if len(rows) == 0]
    for cid in empty_clients:
        donor = int(np.argmax(client_loads))
        donor_rows = client_indices[donor]
        if len(donor_rows) <= 1:
            break
        split = max(1, len(donor_rows) // 2)
        client_indices[cid] = donor_rows[:split]
        client_indices[donor] = donor_rows[split:]
        client_loads[cid] = len(client_indices[cid])
        client_loads[donor] = len(client_indices[donor])

    return client_indices


def build_rec_test_data(df, test_ratio=0.2, seed=42):
    """Split recommendation data into train/test by interactions."""
    np.random.seed(seed)

    # For each user, hold out some items for testing
    train_indices = []
    test_indices = []

    for uid, group in df.groupby('user'):
        idx = group.index.values.copy()
        if len(idx) < 2:
            train_indices.extend(idx)
            continue
        np.random.shuffle(idx)
        split = max(1, int(len(idx) * test_ratio))
        test_indices.extend(idx[:split])
        train_indices.extend(idx[split:])

    return train_indices, test_indices
