import math
import torch
import random
from tqdm import tqdm
from torch import Tensor


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


def split_graph(data, pivot=135000):
    """
    Train, val, test split for graph
    :param data:
    :param val_ratio:
    :param edge_attr:
    :return:
    """
    assert 'batch' not in data

    # num_nodes = data.num_nodes
    row, col = data.edge_index
    target = data.y

    # return upper triangular portion
    mask = row < col
    row, col, target = row[mask], col[mask], target[mask]

    # n_v = int(math.floor(val_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))
    row, col, target = row[perm], col[perm], target[perm]

    # create validation set
    r, c, y = row[pivot:], col[pivot:], target[pivot:]
    data.val_edge_index = torch.stack([r, c], dim=0)
    data.val_target_index = y

    # create train set
    r, c, y = row[:pivot], col[:pivot], target[:pivot]
    data.train_edge_index = torch.stack([r, c], dim=0)
    data.train_target_index = y

    return data
