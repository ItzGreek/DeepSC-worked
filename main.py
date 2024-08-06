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
from utils import SNR_to_noise, noise_to_SNR, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.io as sp
import datetime

prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='../data/europarl/vocab.json', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str,
                    help='Please choose AWGN, Rayleigh, Rician, CDL_ZF or CDL_MMSE')
#parser.add_argument('--channel', default='Rayleigh', type=str,
 #                   help='Please choose AWGN, Rayleigh, Rician or CDL-B')
#parser.add_argument('--channel', default='Rician', type=str,
 #                   help='Please choose AWGN, Rayleigh, Rician or CDL-B')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int) #default is 80
#addditional arguments for MIMO
parser.add_argument('--n_rx', default = 1, type = int)
parser.add_argument('--n_tx', default = 1, type = int)
checkpoint_dir = f"checkpoints/{prefix}{parser.parse_args().channel}_{parser.parse_args().n_tx}x{parser.parse_args().n_rx}_{parser.parse_args().epochs}ep" 
parser.add_argument('--checkpoint-path',
                    default=f'{checkpoint_dir}/deepsc', type=str)
parser.add_argument('--mi-checkpoint-path',
                    default=f'{checkpoint_dir}/minet', type=str)
# added argument to set lambda parameter in the loss computation
parser.add_argument('--lambda_loss', default = 0.0009, type = int)# default was 0.0009 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def check_nan_in_model(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Il parametro {name} contiene NaN")
            return True
    return False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def indices_to_text(indices, idx_to_token,token_to_idx):
    return ' '.join([idx_to_token[idx] for idx in indices if idx in idx_to_token and idx != token_to_idx['<PAD>']])

def validate(epoch, args, net, ch_mat):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
## TODO check se va bene usare 0.1 come standard deviation del noise o se non sarebbe meglio usare un valore diverso? 
    n0_std = 0.1 #  

    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, n0_std, pad_idx,
                            criterion, args.channel, ch_mat, args.n_tx, args.n_rx)
            
            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/simresults.log', 'a') as f:
        f.write('Epoch: {};  Type: VAL; Average Loss: {:.5f}; SNR {:.2f}\n'.format(
                    epoch + 1, total/len(test_iterator), noise_to_SNR(n0_std)))
    print('Epoch: {};  Type: VAL;  Total Average Loss: {:.5f}; SNR {:.2f}\n'.format(
                    epoch + 1, total/len(test_iterator), noise_to_SNR(n0_std)))


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

    #noise_std[0] = 0.1 #DEBUG DELETEME!!!   
    count=0
    for sents in pbar:
        count+=1

        sents = sents.to(device)

        if mi_net is not None:
            # mutual information
            mi = train_mi(net, mi_net, sents, noise_std[0],
                          pad_idx, mi_opt, args.channel, ch_mat, args.n_tx, args.n_rx)

            loss, loss_ce, loss_mine, src_firstsent, pred_firstsent, SNR_inst = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel, ch_mat, args.n_tx, args.n_rx, mi_net=mi_net, lambda_loss=args.lambda_loss)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; Loss_CE {:.5f}; Loss_MI {:.5f}; MI {:.5f}; SNR {:.2f}; SNR_inst {:.2f}'.format(
                    epoch + 1, loss, loss_ce, loss_mine, mi, noise_to_SNR(noise_std[0]), SNR_inst
                )
            )
            
        else:
            #net = deepsc, pad_idx = <PAD>
            mi = 10
            loss, loss_ce, loss_mine, src_firstsent, pred_firstsent = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel, ch_mat, args.n_tx, args.n_rx)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; SNR {:.2f}'.format(
                    epoch + 1, loss, noise_to_SNR(noise_std[0])
                )
            )
    # Prendi la prima frase del batch di input
    first_input_sentence = src_firstsent.tolist()
    input_sentence_text = indices_to_text(first_input_sentence, idx_to_token, token_to_idx)

    # Prendi la prima frase del batch di previsioni e convertila in indici (prendi l'indice con la massima probabilità)
    first_pred_sentence = pred_firstsent.argmax(dim=-1).tolist()
    pred_sentence_text = indices_to_text(first_pred_sentence, idx_to_token, token_to_idx)
    # Stampa le due frasi
    print("Input sentence:", input_sentence_text)
    print("Predicted sentence:", pred_sentence_text)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/simresults.log', 'a') as f:
        f.write('Epoch: {};  Type: Train; Loss: {:.5f}; Loss_CE {:.5f}; Loss_MI {:.5f}; MI {:.5f}; SNR {:.2f}; SNR_inst {:.2f}\n'.format(
                    epoch + 1, loss, loss_ce, loss_mine, mi, noise_to_SNR(noise_std[0]), SNR_inst ))
        f.write(f"Input sentence: {input_sentence_text}\n")
        f.write(f"Predicted sentence:{pred_sentence_text}\n")

    return mi
            
            


if __name__ == '__main__':
    # setup_seed(10)
    # load files
    args = parser.parse_args()
    # Load the the channel matrix if the channel is not AWGN
    Htot = None
    if "CDL" in args.channel:
        try:
            mat_file = sp.loadmat('/home/man2mob/PythonStuff/DeepSC/Hmat/H_1.mat')
            Htot = mat_file['Htot']
            args.vocab_file = '/home/man2mob/PythonStuff/DeepSC/data/' + args.vocab_file
        except FileNotFoundError:
            try:
                mat_file = sp.loadmat('C:/Users/39392/Desktop/Thesis/DeepSC/DeepSC-master/Hmat/H_1.mat')
                Htot = mat_file['Htot']
                args.vocab_file = 'C:/Users/39392/Desktop/Thesis/DeepSC/DeepSC-master/DeepSC-master/' + args.vocab_file
            except FileNotFoundError:
                raise EnvironmentError('Missing Path')

    "args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file"
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = {v: k for k, v in token_to_idx.items()}  # Dizionario inverso
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
        record_mi = 10
        mi = record_mi 

        #train(epoch, args, deepsc)
        mi = train(epoch, args, deepsc, Htot, mi_net) # training with mi_net
       # train(epoch, args, deepsc, Htot) # training without mi_net  
        
        avg_acc = validate(epoch, args, deepsc, Htot)

        # Save the best accuracy score as a "checkpoint"
        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc
        
        # Save the best accuracy score as a "checkpoint"
        if mi < record_mi:
            if not os.path.exists(args.mi_checkpoint_path):
                os.makedirs(args.mi_checkpoint_path)
            with open(args.mi_checkpoint_path + '/mi_checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(mi_net.state_dict(), f)
            record_mi = mi
            
    record_loss = []
    
