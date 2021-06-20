import time
import torch
import pickle
import numpy as np
import pandas as pd
from typing import List
from torch_geometric.data import Data
from datetime import timedelta
from transformers import AutoTokenizer, AutoModel


def chunks(lst, n, device):
    for i in range(0, lst["input_ids"].shape[0], n):
        yield {
            "input_ids": lst["input_ids"][i: i + n].to(device),
            "attention_mask": lst["attention_mask"][i: i + n].to(device)
        }


class AmazonFineFoodsReviews(object):
    def __init__(self, database_path: str):
        super(AmazonFineFoodsReviews, self).__init__()
        self.df = pd.read_csv(database_path)
        print(f'Data:\n{self.df.describe()}')
        print(self.df.columns)
        # self.df = self.df[:100]

    @staticmethod
    def compress(x: List) -> List:
        cnt, dc = 0, {}
        for i in range(len(x)):
            if x[i] not in dc:
                dc[x[i]] = cnt
                x[i] = cnt
                cnt = cnt + 1
            else:
                x[i] = dc[x[i]]

        return x

    def build_graph(self, text_feature=True, language_model_name='bert-base-cased', max_length=512, batch_size=16,
                    device='cpu'):
        """
        Build graph from reviews
        :param text_feature:
        :param language_model_name:
        :param max_length:
        :param batch_size:
        :param device:
        :return:
        """
        user_ids = AmazonFineFoodsReviews.compress(self.df.UserId.tolist())
        product_ids = AmazonFineFoodsReviews.compress(self.df.ProductId.tolist())
        max_user_ids = max(user_ids)
        user_ids = np.array(user_ids)
        product_ids = np.array(product_ids) + max_user_ids + 1
        user_ids, product_ids = user_ids.reshape(1, user_ids.shape[0]), \
                                product_ids.reshape(1, product_ids.shape[0])

        edge_index = torch.tensor(np.concatenate((user_ids, product_ids), 0), dtype=torch.long)

        # self.df.Score = self.df.Score.apply(lambda x: 0 if x < 3 else 1 if x == 3 else 2)
        scores = self.df.Score.tolist()

        if text_feature is True:
            text = self.df.Text.tolist()
            tokenizer = AutoTokenizer.from_pretrained(language_model_name)
            tokenized_text = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt',
                                       verbose=True)

            print("Text embedding...")
            model = AutoModel.from_pretrained(language_model_name).to(device)
            # le, edge_attr, cnt, t0 = tokenized_text["input_ids"].shape[0], None, 0, time.time()

            # for inp in chunks(tokenized_text, batch_size, device):
            #     cnt = cnt + 1
            #     attr = model(**inp)
            #
            #     if device == 'cuda':
            #         attr = attr.pooler_output.cpu().detach().numpy()
            #     else:
            #         attr = attr.pooler_output.detach().numpy()
            #
            #     edge_attr = (
            #         np.atleast_1d(attr)
            #         if edge_attr is None
            #         else np.concatenate([edge_attr, attr])
            #     )
            #
            #     els = time.time() - t0
            #     est = (els / cnt) * ((le / batch_size) - cnt)
            #     print(
            #         f"Embedded {cnt}/{le // batch_size} els: {timedelta(seconds=els)} - est: {timedelta(seconds=est)}")
            #
            # print("Edge attr:", edge_attr.shape)

            # edge_attr = torch.cat(
            #     [tokenized_text['input_ids'].unsqueeze(1), tokenized_text['token_type_ids'].unsqueeze(1),
            #      tokenized_text['attention_mask'].unsqueeze(1)], 1)

            return Data(x=torch.ones(1 + np.max(product_ids), 1), edge_index=edge_index,
                        y=torch.tensor(scores, dtype=torch.long)), None, tokenized_text

        return Data(x=torch.ones(1 + np.max(product_ids), 1), edge_index=edge_index,
                    y=torch.tensor(scores, dtype=torch.long)), None


def dump(file_path, obj):
    with open(file_path, "wb") as stream:
        pickle.dump(obj, stream)


if __name__ == '__main__':
    data = AmazonFineFoodsReviews("data/Reviews.csv")
    graph, edge_attr, tokenized_text = data.build_graph(text_feature=True, device='cuda', batch_size=16)

    dump("data/package/graph.pkl", graph)
    # dump("data/package/edge_attr.pkl", edge_attr)
    dump("data/package/tokenized_text.pkl", tokenized_text)

