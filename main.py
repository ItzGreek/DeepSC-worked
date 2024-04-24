# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.io as sp

parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path',
                    default='checkpoints/deepsc-Rayleigh', type=str)
#parser.add_argument('--channel', default='Rayleigh', type=str,
 #                   help='Please choose AWGN, Rayleigh, Rician or CDL-B')
#parser.add_argument('--channel', default='Rician', type=str,
 #                   help='Please choose AWGN, Rayleigh, Rician or CDL-B')
parser.add_argument('--channel', default='CDL_MMSE', type=str,
                    help='Please choose AWGN, Rayleigh, Rician, CDL_ZF or CDL_MMSE')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)
#addditional arguments for MIMO
parser.add_argument('--n_rx', default = 2, type = int)
parser.add_argument('--n_tx', default = 32, type = int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                            criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator)


def train(epoch, args, net, ch_mat, mi_net=None):
    # load the training set
    train_eur = EurDataset('train')

    # Data loader combines a dataset and a sampler, and provides an iterable over the given dataset
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    # progress bar
    pbar = tqdm(train_iterator)

    # retrieve the noise standard deviation
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            # mutual information
            mi = train_mi(net, mi_net, sents, 0.1,
                          pad_idx, mi_opt, args.channel, ch_mat, args.n_tx, args.n_rx)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net, ch_mat, args.n_tx, args.n_rx)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            #net = deepsc, pad_idx = <PAD>
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel, ch_mat, args.n_tx, args.n_rx)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )


if __name__ == '__main__':
    # setup_seed(10)
    # load files
    args = parser.parse_args()
    # Load the the channel matrix if the channel is not AWGN
    Htot = None
    if "CDL" in args.channel:
        mat_file = sp.loadmat('C:/Users/39392/Desktop/Thesis - Semcom/DeepSC/DeepSC-master/Hmat/H_1.mat')
        Htot = mat_file['Htot']
    args.vocab_file = 'C:/Users/39392/Desktop/Thesis - Semcom/DeepSC/DeepSC-master/DeepSC-master/' + args.vocab_file

    "args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file"
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    # (self, num_layers, src_vocab_size, trg_vocab_size, src_max_len, trg_max_len, d_model, num_heads, dff, dropout = 0.1):
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    # moves the mi_net model parameters and buffer to the CPU or the GPU
    mi_net = Mine().to(device)

    # Loss criterion definition
    criterion = nn.CrossEntropyLoss(reduction='none')

    # optimizers initialization
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    #opt = NoamOpt(args.d_model, 1, 4000, optimizer)

    # parameters initialization
    initNetParams(deepsc)

    # Training and validation for each epoch
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        #train(epoch, args, deepsc)
        train(epoch, args, deepsc, Htot)
        avg_acc = validate(epoch, args, deepsc)

        # Save the best accuracy score as a "checkpoint"
        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc
    record_loss = []
