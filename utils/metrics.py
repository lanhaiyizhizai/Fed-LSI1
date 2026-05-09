"""Evaluation metrics for image classification and recommendation systems."""
import torch
import numpy as np


def test_classification(model, test_loader, device='cpu'):
    """Evaluate classification model: returns accuracy and loss."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target, reduction='sum')
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    avg_loss = test_loss / total
    return accuracy, avg_loss


def generate_fixed_negatives(test_loader, num_items, n_neg=99, seed=42):
    """
    Generate fixed negative samples for each test instance.
    Returns a dict mapping (user_id, pos_item) -> tensor of negative items.
    """
    rng = np.random.RandomState(seed)
    fixed_negs = {}
    for user_ids, item_ids, ratings in test_loader:
        for i in range(user_ids.size(0)):
            u = user_ids[i].item()
            pos = item_ids[i].item()
            negs = rng.choice(num_items, size=n_neg, replace=False).tolist()
            # Ensure positive item not in negatives
            negs = [n for n in negs if n != pos]
            while len(negs) < n_neg:
                extra = rng.randint(0, num_items)
                if extra != pos and extra not in negs:
                    negs.append(extra)
            fixed_negs[(u, pos)] = torch.LongTensor(negs)
    return fixed_negs


def compute_hr_k(model, test_loader, K=10, device='cpu', num_items=None,
                 fixed_negs=None):
    """
    Compute Hit Ratio at K (HR@K) for recommendation.
    Uses fixed negative samples if provided, otherwise random (legacy).
    """
    model.eval()
    hits = 0
    total = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)

            batch_size = user_ids.size(0)

            for i in range(batch_size):
                u = user_ids[i:i + 1]
                pos_item = item_ids[i:i + 1]

                if fixed_negs is not None:
                    key = (user_ids[i].item(), item_ids[i].item())
                    neg_items = fixed_negs.get(key, torch.randint(0, num_items, (99,))).to(device)
                else:
                    n_neg = max(K - 1, 99)
                    neg_items = torch.randint(0, num_items, (n_neg,)).to(device)

                all_items = torch.cat([pos_item, neg_items])
                all_users = u.expand(len(all_items))

                scores = model(all_users, all_items).squeeze()

                _, topk_indices = torch.topk(scores, k=min(K, len(all_items)))
                topk_items = all_items[topk_indices]

                if pos_item.item() in topk_items.tolist():
                    hits += 1
                total += 1

    hr = hits / max(total, 1)
    return hr


def compute_ndcg_k(model, test_loader, K=10, device='cpu', num_items=None,
                   fixed_negs=None):
    """
    Compute NDCG@K for recommendation.
    Uses fixed negative samples if provided, otherwise random (legacy).
    """
    model.eval()
    ndcg_sum = 0.0
    total = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)

            batch_size = user_ids.size(0)

            for i in range(batch_size):
                u = user_ids[i:i + 1]
                pos_item = item_ids[i:i + 1]

                if fixed_negs is not None:
                    key = (user_ids[i].item(), item_ids[i].item())
                    neg_items = fixed_negs.get(key, torch.randint(0, num_items, (99,))).to(device)
                else:
                    n_neg = max(K - 1, 99)
                    neg_items = torch.randint(0, num_items, (n_neg,)).to(device)

                all_items = torch.cat([pos_item, neg_items])
                all_users = u.expand(len(all_items))

                scores = model(all_users, all_items).squeeze()

                _, topk_indices = torch.topk(scores, k=min(K, len(all_items)))
                topk_items = all_items[topk_indices]

                dcg = 0.0
                for rank, item in enumerate(topk_items):
                    if item.item() == pos_item.item():
                        dcg += 1.0 / np.log2(rank + 2)

                idcg = 1.0 / np.log2(2)

                ndcg_sum += dcg / idcg
                total += 1

    ndcg = ndcg_sum / max(total, 1)
    return ndcg


def compute_rec_metrics(model, test_loader, K=10, device='cpu', num_items=None,
                        fixed_negs=None):
    """Compute both HR@K and NDCG@K with fixed negatives."""
    hr = compute_hr_k(model, test_loader, K=K, device=device, num_items=num_items,
                      fixed_negs=fixed_negs)
    ndcg = compute_ndcg_k(model, test_loader, K=K, device=device, num_items=num_items,
                          fixed_negs=fixed_negs)
    return hr, ndcg


def compute_rec_metrics_at_ks(model, test_loader, Ks=(10,), device='cpu', num_items=None,
                              fixed_negs=None):
    """
    Compute HR@K and NDCG@K for multiple K values in one scoring pass.

    Returns:
        dict: {K: {'hr': float, 'ndcg': float}}
    """
    model.eval()
    ks = sorted({int(k) for k in Ks})
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    total = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)

            for i in range(user_ids.size(0)):
                u = user_ids[i:i + 1]
                pos_item = item_ids[i:i + 1]

                if fixed_negs is not None:
                    key = (user_ids[i].item(), item_ids[i].item())
                    neg_items = fixed_negs.get(key, torch.randint(0, num_items, (99,))).to(device)
                else:
                    neg_items = torch.randint(0, num_items, (max(max_k - 1, 99),)).to(device)

                all_items = torch.cat([pos_item, neg_items])
                all_users = u.expand(len(all_items))
                scores = model(all_users, all_items).squeeze()

                _, top_indices = torch.topk(scores, k=min(max_k, len(all_items)))
                top_items = all_items[top_indices].tolist()
                pos = pos_item.item()

                try:
                    pos_rank = top_items.index(pos) + 1
                except ValueError:
                    pos_rank = None

                for k in ks:
                    if pos_rank is not None and pos_rank <= k:
                        hits[k] += 1
                        ndcg_sums[k] += 1.0 / np.log2(pos_rank + 1)

                total += 1

    return {
        k: {
            'hr': hits[k] / max(total, 1),
            'ndcg': ndcg_sums[k] / max(total, 1),
        }
        for k in ks
    }
