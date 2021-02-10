#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import Dataset
import csv


class DictDataset(Dataset):
    def __init__(self, bert_tokenizer, file, max_length):
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_length
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.seqs[idx],
                "attention_mask": self.seq_masks[idx],
                "token_type_ids": self.seq_segments[idx],
                "labels": self.labels[idx]}

    # 获取文本与标签
    def get_input(self, file):
        df = pd.read_csv(file,
                         sep='\t',
                         header=None,
                         names=['question1', 'question2', 'label'],
                         quoting=csv.QUOTE_NONE)

        labels = df['label'].values
        self.num_labels = len(df['label'].unique())
        tokens = map(lambda x1, x2: str(x1) + "[SEP]" + str(x2), df["question1"].values, df["question2"].values)
        # max_length = max(map(lambda x: len(x), tokens))
        text_dict = self.bert_tokenizer.batch_encode_plus(tokens,
                                                          max_length=self.max_length,
                                                          padding="max_length",
                                                          truncation=True,
                                                          add_special_tokens=True,
                                                          return_attention_mask=True,
                                                          return_tensors="pt")
        return text_dict["input_ids"], \
               text_dict["attention_mask"], \
               text_dict["token_type_ids"], \
               torch.Tensor(labels).type(torch.long)


if __name__ == '__main__':
    import os
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    tokenizer = BertTokenizer.from_pretrained(root_path + "/model/wobert")
    # text = tokenizer.encode_plus([["我的世界", "我的世界"], ["平凡的世界", "平凡的世界"]])
    # print(text["input_ids"])
    train_dataset = DictDataset(tokenizer, root_path + "/atec_nlp_sim_train_all.csv", max_length=100)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(next(iter(train_dataloader)))
