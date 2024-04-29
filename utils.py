# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information
#additional libraries
import scipy.io as sp
import random
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 

#turns complex matrices into real values ones
def compl_to_real(mat, n_tx, n_rx):
    H = []
    for i in range(n_rx):
        row = []
        for j in range(n_tx):
            row.append(np.real(mat[i][j]))
            row.append(np.imag(mat[i][j]))
        H.append(row)
    
    return H

#additional function used to compute the channel matrix of the defined CDL channel
def get_mat(ch_mat, n_tx, n_rx):
    
    #draw random values for the first 3 dimensions
    dim1 = random.randint(0, 19999)
    dim2 = random.randint(0, 29)
    dim3 = random.randint(0,2)
    
    #SISO case
    if n_tx == 1 and n_rx == 1:
        dim4 = random.randint(0,1)
        dim5 = random.randint(0, 31)
        
        values = ch_mat[dim1, dim2, dim3, dim4, dim5]
        print(values)
        norm_fact = values*np.conj(values)
        H_norm = values/norm_fact
        H = [[np.real(H_norm), -np.imag(H_norm)], [np.imag(H_norm), np.real(H_norm)]]
        H_inv = np.linalg.inv(H)

        return  torch.Tensor(H).to(device), torch.Tensor(H_inv).to(device)        
    #MIMo case
    else:
        values = ch_mat[dim1, dim2, dim3, :, :n_tx]
        norm_fact = np.linalg.norm(values, 'fro')
        H_norm = values/norm_fact
        #Inverse = np.linalg.inv(H_norm)
        
        U, S, Vh = np.linalg.svd(H_norm, full_matrices = False)

        return  U, S, Vh, H_norm
    
    
    

class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation  
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        
        return Rx_sig
        
        
    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        print(Rx_sig)

        return Rx_sig


    def CDL_ZF(self, Tx_sig, n_var, ch_mat, n_tx, n_rx):
        
        shape = Tx_sig.shape
        
        #Get channel matrix + svd
        U, S, Vh, H = get_mat(ch_mat, n_tx, n_rx)
        #reshape tx_signal and define real and imaginary parts
        Tx_sig = Tx_sig.view(shape[0], -1, 2)
        Tx_sig = Tx_sig.detach().numpy()
        Tx_sig = np.vectorize(complex)(Tx_sig[:,:,0], Tx_sig[:,:,1])
        
        Tx_sig_flat = Tx_sig.flatten() 
        
        #duplicate the signal to simulate the 2 datastream
        Tx_sig_exp = np.repeat(Tx_sig_flat[:, np.newaxis], 2, axis=1)
        
        #matrix mult
        x_tx = np.conj(Vh).T @ Tx_sig_exp.T
        n0 = (np.random.normal(0, n_var, (n_tx, len(Tx_sig_flat))) + np.random.normal(0, n_var, (n_tx, len(Tx_sig_flat)))*1j)/np.sqrt(2)
        #check the introduced noise power 
        y = H @ x_tx + n0
        x_hat = np.conj(U).T @ y 
        S = np.diag(S)
        S_inv = np.linalg.pinv(S)
        x_hat = S_inv @ x_hat
        Rx_array = np.round((np.sum(x_hat, axis=0) / 2), decimals = 7)
        ratio = Rx_array/Tx_sig_flat
        
        #plot to visualize the signal constellation before and after the transmission
        plt.scatter(np.real(Tx_sig_flat), np.imag(Tx_sig_flat), label = "Tx_sig")
        plt.scatter(np.real(Rx_array), np.imag(Rx_array), label = "Rx_sig")
        plt.grid(True)
        plt.show()
   
        
        #Reshape the received vector
        Rx_array = Rx_array.reshape((shape[0],248))
        real_part = np.real(Rx_array)
        imaginary_part = np.imag(Rx_array)
        
        
        Rx_array = np.stack((real_part, imaginary_part), axis=-1)
        #turn the received array into a tensor
        Rx_sig = torch.tensor(Rx_array).view(shape).to(torch.float32)
        
        return Rx_sig
    
    def CDL_MMSE(self, Tx_sig, n_var, ch_mat, n_tx, n_rx):
        
        shape = Tx_sig.shape
        n_var = 0.001
        #Get channel matrix + svd
        _, _, Vh, H = get_mat(ch_mat, n_tx, n_rx)
        #reshape tx_signal and define real and imaginary parts
        Tx_sig = Tx_sig.view(shape[0], -1, 2)
        Tx_sig = Tx_sig.detach().numpy()
        Tx_sig = np.vectorize(complex)(Tx_sig[:,:,0], Tx_sig[:,:,1])
        
        Tx_sig_flat = Tx_sig.flatten() 
        
        Tx_sig_exp = np.repeat(Tx_sig_flat[:, np.newaxis], 2, axis=1) #dim: 31744x2
        
        V = np.conj(Vh).T
        x_tx = V @ Tx_sig_exp.T 
        H_eq = H @ V
        H_herm = np.conj(H_eq).T
        Second_part = np.linalg.inv(H_eq @ H_herm + (n_var**2)*np.eye(n_rx))
        W_MMSE = H_herm @ Second_part
        n0 = (np.random.normal(0, n_var, (n_rx, len(Tx_sig_flat))) + np.random.normal(0, n_var, (n_rx, len(Tx_sig_flat)))*1j)/np.sqrt(2)

        y = H @ x_tx + n0 
        x_hat = W_MMSE @ y
        Rx_array = np.round((np.sum(x_hat, axis=0) / 2), decimals = 7)
        #ratio = Rx_array/Tx_sig_flat
        
        #Reshape the received vector
        #we use half of the multiplication between the second and the third shape values since the original number of symbols was converted to complex values
        Rx_array = Rx_array.reshape((shape[0],int(shape[1]*shape[2]/2))) 
        real_part = np.real(Rx_array)
        imaginary_part = np.imag(Rx_array)
        
        
        Rx_array = np.stack((real_part, imaginary_part), axis=-1)
        #turn the received array into a tensor
        Rx_sig = torch.tensor(Rx_array).view(shape).to(torch.float32)
        
        return Rx_sig

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            #Xavier uniform initialization sets the values of weights in the matrix randomly from a uniform distribution with bounds that depend on the number of input and output units
            nn.init.xavier_uniform_(p)
    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size) #size = 30
    #creates an upper triangular matrix of dimensions 1x30x30
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

#generate masks ensuring that the model attends to relevant information while ignoring padding tokens and future tokens during decoding   
def create_masks(src, trg, padding_idx):

    #(src == padding_idx) checks the elements in the batch equal to the padding index, .unsqueeze(-2) add an additional dimension
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data) #trg.size(-1) = 30, look_ahead_mask size 1x30x30
    #takes the maximum value for each considered position
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.to(device), combined_mask.to(device) #src_mask size 128x1x31, combined_mask size 128x30x30

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    #This creates a mask where non-padding elements are set to 1 and padding elements are set to 0. It ensures that the loss contribution of padding tokens is ignored.
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x


#def SNR_to_noise(snr):
 #   snr = 10 ** (snr / 10)
  #  noise_std = 1 / np.sqrt(2 * snr)

   # return noise_std

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(snr)

    return noise_std

def train_step(model, src, trg, n_var, pad, opt, criterion, channel, ch_mat, n_tx, n_rx, mi_net=None):
    
    #removes the last element from each batch line
    trg_inp = trg[:, :-1]
    #removes the first element from each batch line
    trg_real = trg[:, 1:]

    channels = Channels()
    #Clears the gradients of all optimized tensors.
    opt.zero_grad()
    
    #src = batch (sents), trg_inp = src - last element, pad = <PAD>
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    source = src
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'CDL_ZF':
        Rx_sig = channels.CDL_ZF(Tx_sig, n_var, ch_mat, n_tx, n_rx)
    elif channel == 'CDL_MMSE':
        Rx_sig = channels.CDL_MMSE(Tx_sig, n_var, ch_mat, n_tx, n_rx)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    #calls the decoder forward step
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    
    #y_est = x +  torch.matmul(n, torch.inverse(H))
    #loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)

    #mutual information evaluation
    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss + 0.0009 * loss_mine
    # loss = loss_function(pred, trg_real, pad)

    loss.backward()
    opt.step()

    return loss.item()


def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel, ch_mat, n_tx, n_rx):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    #Creates masks for the source sequence and the target input sequence
    #check if there is a value equal to the padding index in the source
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'CDL_ZF':
        Rx_sig = channels.CDL_ZF(Tx_sig, n_var, ch_mat, n_tx, n_rx)
    elif channel == 'CDL_MMSE':
        Rx_sig = channels.CDL_MMSE(Tx_sig, n_var, ch_mat, n_tx, n_rx)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, Rician, CDL-ZF and CDL-MMSE")

    #joint and marginal distribution computation
    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    
    #mutual information lower bound
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    #computes the gradient of current tensor wrt graph leaves
    loss_mine.backward()
    #the norm is computed over all gradients together, as if they were concatenated into a single vector
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    #updated network parameters through the optimizer
    opt.step()

    #returns mutual information loss
    return loss_mine.item()

def val_step(model, src, trg, n_var, pad, criterion, channel, ch_mat, n_tx, n_rx):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'CDL_ZF':
        Rx_sig = channels.CDL_ZF(Tx_sig, n_var, ch_mat, n_tx, n_rx)
    elif channel == 'CDL_MMSE':
        Rx_sig = channels.CDL_MMSE(Tx_sig, n_var, ch_mat, n_tx, n_rx)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, Rician, CDL-ZF and CDL-MMSE")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    # loss = loss_function(pred, trg_real, pad)
    
    return loss.item()
    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel, ch_mat, n_tx, n_rx):
    """ 
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    elif channel == 'CDL_ZF':
        Rx_sig = channels.CDL_ZF(Tx_sig, n_var, ch_mat, n_tx, n_rx)
    elif channel == 'CDL_MMSE':
        Rx_sig = channels.CDL_MMSE(Tx_sig, n_var, ch_mat, n_tx, n_rx)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, Rician, CDL-ZF and CDL-MMSE")       
    #channel_enc_output = model.blind_csi(channel_enc_output)
          
    memory = model.channel_decoder(Rx_sig)
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
#        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        
        # predict the word
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)
        
        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs



