import os
import time
import pickle
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from loader import AmazonFineFoodsReviews
from transformers import AdamW
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

TRANSFORMER_ALIASES = [
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'xlm-roberta-base',
    'xlm-mlm-xnli15-1024',
    'xlm-mlm-tlm-xnli15-1024',
    'roberta-large-mnli',
    'facebook/bart-large-mnli',
]


class SentimentAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, text, labels, index, max_length=256):
        self.tokenizer = tokenizer
        self.text = text[index]
        self.labels = labels[index]
        self.max_length = max_length

        assert self.text.shape[0] == self.labels.shape[0], IndexError

    def __getitem__(self, idx):
        item = self.tokenizer(text[idx], truncation=True, padding="max_length", max_length=self.max_length)
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return self.labels.shape[0]


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return classification_report(labels, preds, output_dict=True)


def concat(x, y):
    return np.atleast_1d(y) if x is None else np.concatenate([x, y])


if __name__ == '__main__':
    # argument parsing
    argument_parser = argparse.ArgumentParser(description='Fine-tune transformer models for '
                                                          'Vietnamese Sentiment Analysis.')
    argument_parser.add_argument('--model', type=str, default='bert-base-cased',
                                 help='Model shortcut from pretrained hub.')
    argument_parser.add_argument('--freeze_encoder', type=bool, default=True,
                                 help='Whether BERT base encoder is freeze or not.')
    argument_parser.add_argument('--epoch', type=int, default=1, help='Number of training epochs.')
    argument_parser.add_argument('--learning_rate', type=int, default=1e-5, help='Model learning rate.')
    argument_parser.add_argument('--accumulation_steps', type=int, default=50, help='Gradient accumulation steps.')
    argument_parser.add_argument('--device', type=str, default='cpu', help='Training device.')
    argument_parser.add_argument('--root', type=str, default='data', help='Directory to dataset.')
    argument_parser.add_argument('--data', type=str, default='VLSP2016',
                                 help='Which dataset use to train a.k.a (VLSP2016, UIT-VSFC, AIVIVN).')
    argument_parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    argument_parser.add_argument('--test_size', type=float, default=0.2, help='Test size.')
    argument_parser.add_argument('--max_length', type=int, default=256, help='Maximum length of BERT tokenizer.')
    argument_parser.add_argument('--num_labels', type=int, default=5,
                                 help='Number of classification labels (a.k.a sentiment polarities)')
    argument_parser.add_argument('--warmup_steps', type=int, default=300, help='Learning rate warming up step.')
    argument_parser.add_argument('--weight_decay', type=float, default=0.01, help='Training weight decay.')
    argument_parser.add_argument('--save_steps', type=int, default=500, help='Number of step to save model.')
    argument_parser.add_argument('--eval_steps', type=int, default=100, help='Number of step to evaluate model.')
    argument_parser.add_argument('--logging_steps', type=int, default=10, help='Number of step to write log.')
    args = argument_parser.parse_args()
    print(args)

    # load sentiment data
    data = AmazonFineFoodsReviews("data/Reviews.csv")
    graph, text = data.build_graph(text_feature=True)
    label = graph.y - 1

    # init model
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = args.num_labels

    net = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_index = np.arange(label.shape[0])
    np.random.shuffle(data_index)
    val_index = data_index[: int(args.test_size * data_index.shape[0])]
    train_index = data_index[int(args.test_size * data_index.shape[0]):]

    text = np.array(text)
    train_dataset = SentimentAnalysisDataset(tokenizer, text, label, train_index, args.max_length)
    val_dataset = SentimentAnalysisDataset(tokenizer, text, label, val_index, args.max_length)

    # freeze encoder
    if args.freeze_encoder is True:
        for param in net.base_model.parameters():
            param.requires_grad = True
    print(f'Loaded model architecture: {args.model}')

    # init optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir='./results',
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_steps=args.eval_steps,
        logging_dir='./logs',
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.accumulation_steps,
        save_steps=args.save_steps,
        # no_cuda=False if args.device == 'cuda' else True
    )

    trainer = Trainer(
        model=net,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    train_output = trainer.train()
    evaluate_output = trainer.evaluate(eval_dataset=val_dataset)

    print(train_output)
    print(evaluate_output)
