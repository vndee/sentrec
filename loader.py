import torch
import numpy as np
import pandas as pd
from typing import List
from torch_geometric.data import Data

from transformers import AutoTokenizer


class AmazonFineFoodsReviews(object):
    def __init__(self, database_path: str):
        super(AmazonFineFoodsReviews, self).__init__()
        self.df = pd.read_csv(database_path)
        print(f'Data:\n{self.df.describe()}')
        print(self.df.columns)
        self.df = self.df[:20]

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

    def build_graph(self, language_model_name='bert-base-cased'):
        """
        Build graph from reviews
        :param language_model_name:
        :return:
        """
        user_ids = AmazonFineFoodsReviews.compress(self.df.UserId.tolist())
        product_ids = AmazonFineFoodsReviews.compress(self.df.ProductId.tolist())
        max_user_ids = max(user_ids)
        user_ids = np.array(user_ids)
        product_ids = np.array(product_ids) + max_user_ids + 1
        user_ids, product_ids = user_ids.reshape(1, user_ids.shape[0]), \
                                product_ids.reshape(1, product_ids.shape[0])

        self.df.Score = self.df.Score.apply(lambda x: 0 if x < 3 else 1 if x == 3 else 2)
        scores = self.df.Score.tolist()
        # summaries = self.df.Summary.tolist()
        text = self.df.Text.tolist()

        tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        tokenized_text = tokenizer(text, padding=True, return_tensors='pt', verbose=True)

        edge_attr = torch.cat(
            [tokenized_text['input_ids'].unsqueeze(1), tokenized_text['token_type_ids'].unsqueeze(1),
             tokenized_text['attention_mask'].unsqueeze(1)], 1)

        edge_index = torch.tensor(np.concatenate((user_ids, product_ids), 0), dtype=torch.long)
        print('-' * 100)

        return Data(x=torch.ones(1 + np.max(product_ids), 1), edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor(scores, dtype=torch.long))
