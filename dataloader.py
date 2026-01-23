import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

class MyDataset(Dataset):
    def __init__(self, path, pretrain_model, max_len, label2id):
        self.max_len = max_len
        self.label2id = label2id
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_model)

        with open(path, 'r') as f:
            lines = f.readlines()
        self.data = [json.loads(line) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['text'],
                                max_length=self.max_len,
                                padding='max_length',
                                truncation=True)

        ll_raw = item['ll_tokens_list']        
        ll_raw = [row[:self.max_len] for row in ll_raw]
        max_len_in_sample = max(len(r) for r in ll_raw)
        ll_padded = []

        for r in ll_raw:
            pad_len = max_len_in_sample - len(r)
            ll_padded.append(r + [0] * pad_len)
        ll_tensor = torch.tensor(ll_padded, dtype=torch.long)

        if max_len_in_sample < self.max_len:
            pad = torch.zeros(4, self.max_len - max_len_in_sample).long()
            ll_tensor = torch.cat([ll_tensor, pad], dim=1)

        ll_tensor = ll_tensor.float() 

        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'll_tokens_list': ll_tensor,
            'label_family': torch.tensor(self.label2id[item['label_family']]),
            'label_binary': torch.tensor(self.label2id[item['label_binary']]),
            'text': item['text'] 
        }

def get_dataloader(data_path, pretrain_model, batch_size, max_len, label2id, shuffle=True):
    dataset = MyDataset(data_path, pretrain_model, max_len, label2id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16)
    return dataloader
