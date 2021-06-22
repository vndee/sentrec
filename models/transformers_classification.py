import os
import time
import pickle
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import AdamW, get_scheduler
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


class SentimentAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, tokenize, text, labels, index, max_length=256):
        self.tokenize = tokenize
        self.text = text[index]
        self.labels = labels[index]
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.tokenize(self.text[idx], truncation=True, padding="max_length", max_length=self.max_length)
        return torch.tensor(item["input_ids"]), torch.tensor(item["attention_mask"]), self.labels[idx]

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


def load_data(file):
    reviews = pd.read_csv(f"{file}.csv")
    return np.array(reviews.Text), np.array(reviews.Score)


if __name__ == '__main__':
    # argument parsing
    argument_parser = argparse.ArgumentParser(description='Fine-tune transformer models for '
                                                          'Vietnamese Sentiment Analysis.')
    argument_parser.add_argument('-m', '--model', type=str, default='bert-base-cased',
                                 help='Model shortcut from pretrained hub.')
    argument_parser.add_argument('-f', '--freeze_encoder', type=bool, default=True,
                                 help='Whether BERT base encoder is freeze or not.')
    argument_parser.add_argument('-e', '--epoch', type=int, default=3, help='Number of training epochs.')
    argument_parser.add_argument('-l', '--learning_rate', type=int, default=5e-5, help='Model learning rate.')
    argument_parser.add_argument('-a', '--accumulation_steps', type=int, default=50,
                                 help='Gradient accumulation steps.')
    argument_parser.add_argument('-d', '--device', type=str, default='cpu', help='Training device.')
    argument_parser.add_argument('-r', '--root', type=str, default='data', help='Directory to dataset.')
    argument_parser.add_argument('-g', '--data', type=str, default='VLSP2016',
                                 help='Which dataset use to train a.k.a (VLSP2016, UIT-VSFC, AIVIVN).')
    argument_parser.add_argument('-b', '--batch_size', type=int, default=16, help='Training batch size.')
    argument_parser.add_argument('-t', '--test_size', type=float, default=0.2, help='Test size.')
    argument_parser.add_argument('-x', '--max_length', type=int, default=256, help='Maximum length of BERT tokenizer.')
    argument_parser.add_argument('-n', '--num_labels', type=int, default=5,
                                 help='Number of classification labels (a.k.a sentiment polarities)')
    argument_parser.add_argument('-w', '--warmup_steps', type=int, default=300, help='Learning rate warming up step.')
    argument_parser.add_argument('-y', '--weight_decay', type=float, default=0.01, help='Training weight decay.')
    argument_parser.add_argument('-v', '--save_steps', type=int, default=500, help='Number of step to save model.')
    argument_parser.add_argument('-p', '--eval_steps', type=int, default=100, help='Number of step to evaluate model.')
    argument_parser.add_argument('-i', '--logging_steps', type=int, default=10, help='Number of step to write log.')
    args = argument_parser.parse_args()
    print(args)

    # load sentiment data
    train_x, train_y = load_data("data/mini/train")
    test_x, test_y = load_data("data/mini/test")
    print(f"Train X: {train_x.shape} - Train Y: {train_y.shape}")
    print(f"Test X: {test_x.shape} - Test Y: {test_y.shape}")

    # init model
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = args.num_labels

    net = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_index = np.arange(train_y.shape[0])
    np.random.shuffle(data_index)
    val_index = data_index[: int(args.test_size * data_index.shape[0])]
    train_index = data_index[int(args.test_size * data_index.shape[0]):]

    train_dataset = SentimentAnalysisDataset(tokenizer, train_x, train_y, train_index, args.max_length)
    val_dataset = SentimentAnalysisDataset(tokenizer, train_x, train_y, val_index, args.max_length)
    test_dataset = SentimentAnalysisDataset(tokenizer, test_x, test_y, np.arange(test_y.shape[0]), args.max_length)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    optimizer = AdamW(net.parameters(), lr=args.learning_rate)
    num_training_steps = args.epoch * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    net.to(args.device).train()

    for epoch in range(args.epoch):
        net.train()
        out, prd = None, None
        for inp, attn, lb in tqdm(train_loader, desc=f"Training {epoch}/{args.epoch}"):
            inp, attn, lb = inp.to(args.device), attn.to(args.device), lb.to(args.device)
            outputs = net(input_ids=inp, attention_mask=attn, labels=lb)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            predictions = torch.argmax(outputs.logits, dim=-1)

            if args.device == "cuda":
                lb = lb.detach().cpu().numpy()
                predictions = predictions.detach().cpu().numpy()

            prd = np.atleast_1d(lb) if prd is None else np.concatenate([prd, lb])
            out = np.atleast_1d(predictions) if out is None else np.concatenate([out, predictions])

        train_acc, train_f1 = accuracy_score(prd, out), f1_score(prd, out, average="macro")

        net.eval()
        out, prd = None, None
        with torch.no_grad():
            for inp, attn, lb in tqdm(val_loader, desc=f"Validating {epoch}/{args.epoch}"):
                inp, attn, lb = inp.to(args.device), attn.to(args.device), lb.to(args.device)
                outputs = net(input_ids=inp, attention_mask=attn, labels=lb)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                if args.device == "cuda":
                    lb = lb.detach().cpu().numpy()
                    predictions = predictions.detach().cpu().numpy()

                prd = np.atleast_1d(lb) if prd is None else np.concatenate([prd, lb])
                out = np.atleast_1d(predictions) if out is None else np.concatenate([out, predictions])

        val_acc, val_f1 = accuracy_score(prd, out), f1_score(prd, out, average="macro")

        print(
            f"Epoch {epoch + 1}/{args.epoch} - train_acc: {train_acc} - train_f1: {train_f1} - val_acc: {val_acc} - val_f1: {val_f1}")

    out, prd = None, None
    with torch.no_grad():
        for inp, attn, lb in test_loader:
            inp, attn, lb = inp.to(args.device), attn.to(args.device), lb.to(args.device)
            outputs = net(input_ids=inp, attention_mask=attn, labels=lb)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            if args.device == "cuda":
                lb = lb.detach().cpu().numpy()
                predictions = predictions.detach().cpu().numpy()

            prd = np.atleast_1d(lb) if prd is None else np.concatenate([prd, lb])
            out = np.atleast_1d(predictions) if out is None else np.concatenate([out, predictions])

        test_acc, test_f1 = accuracy_score(prd, out), f1_score(prd, out, average="macro")

    print(f"Final test acc: {test_acc} - test f1: {test_f1}")
