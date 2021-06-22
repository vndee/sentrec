import math
import torch
from torch_sparse import transpose
from torch_sparse.storage import SparseStorage
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.num_nodes import maybe_num_nodes


def coalesce(index, value, m, n, op="mean"):
    storage = SparseStorage(row=index[0], col=index[1], value=value,
                            sparse_sizes=(m, n), is_sorted=False)
    storage = storage.coalesce(reduce=op)
    return torch.stack([storage.row(), storage.col()], dim=0), storage.value()


def is_undirected(edge_index, edge_attr=None, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    if edge_attr is None:
        undirected_edge_index = to_undirected_edge_index(edge_index, num_nodes=num_nodes)
        return edge_index.size(1) == undirected_edge_index.size(1)
    else:
        edge_index_t, edge_attr_t = transpose(edge_index, edge_attr, num_nodes,
                                              num_nodes, coalesced=True)
        index_symmetric = torch.all(edge_index == edge_index_t)
        attr_symmetric = torch.all(edge_attr == edge_attr_t)
        return index_symmetric and attr_symmetric


def to_undirected_edge_index(edge_index, target_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)

    if target_index is not None:
        tgt = torch.cat([target_index, target_index], dim=0)
    else:
        tgt = None

    edge_index = torch.stack([row, col], dim=0)
    edge_index, target_index = coalesce(edge_index, tgt, num_nodes, num_nodes)

    return edge_index, target_index


def to_undirected(data):
    data.edge_index, data.y = to_undirected_edge_index(edge_index=data.edge_index, target_index=data.y)

    data.train_edge_index, data.train_target_index = to_undirected_edge_index(edge_index=data.train_edge_index,
                                                                              target_index=data.train_target_index)
    # dist = torch.bincount(data.train_target_index)
    # tr_n = int(math.floor((1.0 * dist[dist.nonzero()]).mean().item()))
    # data.train_neg_index = negative_sampling(edge_index=data.edge_index, num_neg_samples=tr_n)
    # data.train_neg_index, _ = to_undirected_edge_index(edge_index=data.train_neg_index, target_index=None)

    data.val_edge_index, data.val_target_index = to_undirected_edge_index(edge_index=data.val_edge_index,
                                                                          target_index=data.val_target_index)
    # dist = torch.bincount(data.val_target_index)
    # v_n = int(math.floor((1.0 * dist[dist.nonzero()]).mean().item()))
    # data.val_neg_index = negative_sampling(edge_index=data.edge_index, num_neg_samples=v_n)
    # data.val_neg_index, _ = to_undirected_edge_index(edge_index=data.val_neg_index, target_index=None)
    #
    # data.test_edge_index, data.test_target_index = to_undirected_edge_index(edge_index=data.test_edge_index,
    #                                                                         target_index=data.test_target_index)
    # dist = torch.bincount(data.test_target_index)
    # t_n = int(math.floor((1.0 * dist[dist.nonzero()]).mean().item()))
    # data.test_neg_index = negative_sampling(edge_index=data.edge_index, num_neg_samples=t_n)
    # data.test_neg_index, _ = to_undirected_edge_index(edge_index=data.test_neg_index, target_index=None)
    return data
