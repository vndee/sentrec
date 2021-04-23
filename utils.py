import math
import torch
import random
from tqdm import tqdm
from torch import Tensor
from torch_geometric.utils import negative_sampling


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


def split_graph(data, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Train, val, test split for graph
    :param data:
    :param val_ratio:
    :param test_ratio:
    :param random_state:
    :return:
    """
    assert 'batch' not in data

    num_nodes = data.num_nodes
    row, col = data.edge_index
    target = data.y

    # return upper triangular portion
    mask = row < col
    row, col, target = row[mask], col[mask], target[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))
    row, col, target = row[perm], col[perm], target[perm]

    # create validation set
    r, c, y = row[: n_v], col[: n_v], target[: n_v]
    data.val_edge_index = torch.stack([r, c], dim=0)
    data.val_target_index, dist = y, torch.bincount(y)
    val_neg = int(math.floor((1.0 * dist[dist.nonzero()]).mean().item()))
    data.val_neg_index = negative_sampling(edge_index=data.edge_index, num_nodes=num_nodes, num_neg_samples=val_neg)

    # create test set
    r, c, y = row[n_v: n_v + n_t], col[n_v: n_v + n_t], target[n_v: n_v + n_t]
    data.test_edge_index = torch.stack([r, c], dim=0)
    data.test_target_index, dist = y, torch.bincount(y)
    test_neg = int(math.floor((1.0 * dist[dist.nonzero()]).mean().item()))
    data.test_neg_index = negative_sampling(edge_index=data.edge_index, num_nodes=num_nodes, num_neg_samples=test_neg)

    # create train set
    r, c, y = row[n_v + n_t:], col[n_v + n_t:], target[n_v + n_t:]
    data.train_edge_index = torch.stack([r, c], dim=0)
    data.train_target_index, dist = y, torch.bincount(y)
    train_neg = int(math.floor((1.0 * dist[dist.nonzero()]).mean().item()))
    data.train_neg_index = negative_sampling(edge_index=data.edge_index, num_nodes=num_nodes, num_neg_samples=train_neg)

    return data


def verify_negative_edge(data):
    print('Verifying data..')
    # verify negative train edge
    for i in tqdm(range(data.train_neg_index.shape[1])):
        for j in range(data.train_edge_index.shape[1]):
            assert data.train_neg_index[0][i] != data.train_edge_index[0][j] or data.train_neg_index[1][i] != \
                   data.train_edge_index[1][j], IndexError('Found train_neg_edge in train_edge')
        for j in range(data.val_edge_index.shape[1]):
            assert data.train_neg_index[0][i] != data.val_edge_index[0][j] or data.train_neg_index[1][i] != \
                   data.val_edge_index[1][j], IndexError('Found train_neg_edge in val_edge')
        for j in range(data.test_edge_index.shape[1]):
            assert data.train_neg_index[0][i] != data.test_edge_index[0][j] or data.train_neg_index[1][i] != \
                   data.test_edge_index[1][j], IndexError('Found train_neg_edge in test_edge')

    # verify negative val edge
    for i in tqdm(range(data.val_neg_index.shape[1])):
        for j in range(data.train_edge_index.shape[1]):
            assert data.val_neg_index[0][i] != data.train_edge_index[0][j] or data.val_neg_index[1][i] != \
                   data.train_edge_index[1][j], IndexError('Found train_neg_edge in train_edge')
        for j in range(data.val_edge_index.shape[1]):
            assert data.val_neg_index[0][i] != data.val_edge_index[0][j] or data.val_neg_index[1][i] != \
                   data.val_edge_index[1][j], IndexError('Found train_neg_edge in val_edge')
        for j in range(data.test_edge_index.shape[1]):
            assert data.val_neg_index[0][i] != data.test_edge_index[0][j] or data.val_neg_index[1][i] != \
                   data.test_edge_index[1][j], IndexError('Found train_neg_edge in test_edge')

    # verify negative test edge
    for i in tqdm(range(data.test_neg_index.shape[1])):
        for j in range(data.train_edge_index.shape[1]):
            assert data.test_neg_index[0][i] != data.train_edge_index[0][j] or data.test_neg_index[1][i] != \
                   data.train_edge_index[1][j], IndexError('Found train_neg_edge in train_edge')
        for j in range(data.val_edge_index.shape[1]):
            assert data.test_neg_index[0][i] != data.val_edge_index[0][j] or data.test_neg_index[1][i] != \
                   data.val_edge_index[1][j], IndexError('Found train_neg_edge in val_edge')
        for j in range(data.test_edge_index.shape[1]):
            assert data.test_neg_index[0][i] != data.test_edge_index[0][j] or data.test_neg_index[1][i] != \
                   data.test_edge_index[1][j], IndexError('Found train_neg_edge in test_edge')

    print('Test passed! Found no conflicts in negative edges.')
