import time
from transformers import AutoTokenizer


class EdgeHashMap(object):
    def __init__(self, edge_index, edge_attr):
        self._dict = dict()

        for it in range(edge_index.shape[1]):
            u, v = edge_index[0][it].item(), edge_index[1][it].item()
            if u > v:
                u, v = v, u

            if (u, v) not in self._dict:
                self._dict[(u, v)] = edge_attr[it]

        self.dim = 768

    def get(self, u, v):
        if u > v:
            u, v = v, u

        return self._dict[(u, v)]

    def get_dim(self):
        return self.dim

