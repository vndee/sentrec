import torch
from torch_sparse import coalesce, transpose
from torch_sparse.storage import SparseStorage
from torch_geometric.utils.num_nodes import maybe_num_nodes


def coalesce(index, value, m, n, op="add"):
    """Row-wise sorts :obj:`value` and removes duplicate entries. Duplicate
    entries are removed by scattering them together. For scattering, any
    operation of `"torch_scatter"<https://github.com/rusty1s/pytorch_scatter>`_
    can be used.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of corresponding dense matrix.
        n (int): The second dimension of corresponding dense matrix.
        op (string, optional): The scatter operation to use. (default:
            :obj:`"add"`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

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
    row, col, tgt = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0), torch.cat([target_index, target_index],
                                                                                          dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, target_index = coalesce(edge_index, tgt, num_nodes, num_nodes)

    return edge_index, target_index


def to_undirected(data):
    data.train_edge_index, data.train_target_index = to_undirected_edge_index(edge_index=data.train_edge_index,
                                                                              target_index=data.train_target_index)
    data.val_edge_index, data.val_target_index = to_undirected_edge_index(edge_index=data.val_edge_index,
                                                                          target_index=data.val_target_index)
    data.test_edge_index, data.test_target_index = to_undirected_edge_index(edge_index=data.test_edge_index,
                                                                            target_index=data.test_target_index)
    return data
