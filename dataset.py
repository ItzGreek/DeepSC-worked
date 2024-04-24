# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: EurDataset.py
@Time: 2021/3/31 23:20
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = 'C:/Users/39392/Desktop/Thesis - Semcom/DeepSC/DeepSC-master/DeepSC-master/'
        #data_dir = '/import/antennas/Datasets/hx301/'
        with open(data_dir + 'europarl/{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)

    #retrive the corresponding item from the dataset
    def __getitem__(self, index):
        sents = self.data[index]
        return  sents
    
    def __len__(self):
        return len(self.data)

#used to collate a batch of data samples into a single tensor.
def collate_data(batch):

    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))   # get the max length of sentence in current batch
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True) #sort in descending order

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # padding the questions

    return  torch.from_numpy(sents)  #array to tensor conversion