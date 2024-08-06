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
import pandas as pd
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
#def get_mat(ch_mat, n_tx, n_rx):
    

        
 #    dim1 = random.randint(0, 19999)
  #   dim2 = random.randint(0, 29)
  #   dim3 = random.randint(0,2)
   #  values = ch_mat[dim1, dim2, dim3, :n_rx, :n_tx]
  #   norm_fact = np.linalg.norm(values, 'fro')
  #   H_norm = values/norm_fact
     
     # Compute the Gram matrix H^H H
     #gram_mat = H_norm.conj().T @ H_norm
    
    # Compute the determinant of the Gram matrix
    # det = np.linalg.det(H_norm)
        
   #  U, S, Vh = np.linalg.svd(H_norm, full_matrices = False)
        

   #  return  U, S, Vh, H_norm
 
def get_mat(ch_mat, n_tx, n_rx, channel):
    if channel == "CDL_MMSE" or channel == "CDL_ZF":
        dim1 = random.randint(0, 19999)
        dim2 = random.randint(0, 29)
        dim3 = random.randint(0, 2)
        
        # Extract the submatrix from the channel matrix
        values = ch_mat[dim1, dim2, dim3, :n_rx, :n_tx]
        # normalize channel coefficients in the channel matrix to have average power = 1
        #norm_fact = np.mean(np.abs(values)**2)
        #values = values / np.sqrt(norm_fact)
        correction_fact = 1/n_tx
        values = values / np.sqrt(correction_fact)
        
    elif channel == "Rician":
        K = 1
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))

        H_real = torch.normal(mean, std, size=[n_rx, n_tx]).to(device)
        H_imag = torch.normal(mean, std, size=[n_rx, n_tx]).to(device)
        values = H_real + 1j * H_imag
    elif channel == "Rayleigh":
        H_real = torch.normal(0, 1, size=[n_rx, n_tx]).to(device)
        H_imag = torch.normal(0, 1, size=[n_rx, n_tx]).to(device)
        values = (H_real + 1j * H_imag)/torch.sqrt(torch.tensor(2))
    
    # Convert to PyTorch tensor and move to the specified device
    #values = torch.tensor(values, dtype=torch.complex64).to(device)
    H_norm = torch.tensor(values, dtype=torch.complex64).to(device)
    ## Compute the Frobenius norm
    #norm_fact = torch.linalg.norm(values, 'fro')
    
    # Normalize the matrix
    #H_norm = torch.sqrt(torch.tensor(n_rx).to(device))*values / norm_fact

    # Perform Singular Value Decomposition (SVD)
    U, S, Vh = torch.linalg.svd(H_norm, full_matrices=False)
    
    return U, S, Vh, H_norm
     
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Function to compare tensors and save differences to a file
def save_differences(Tx_array, Rx_sig):
    # Open a file to append the output in the specified directory
    output_file = os.path.join(output_dir, 'differences.txt')
    with open(output_file, 'a') as f:
        # Iterate over the elements and their indices
        for index in np.ndindex(Tx_array.shape):
            # Compare the elements rounded to four decimal places
            if round(Tx_array[index], 4) != round(Rx_sig[index], 4):
                # Create the output string
                output_str = f"Shape {Tx_array.shape}: Rx_sig element{index}: {Rx_sig[index]:.4f}, Tx_array element{index}: {Tx_array[index]:.4f}\n"
                # Write the output string to the file
                f.write(output_str)
class Channels():

    def AWGN(self, Tx_sig, n_var, n_tx, n_rx):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig
    
    def Rayleigh(self, Tx_sig, n_var, n_rx, n_tx):
        shape = Tx_sig.shape
        power_txsig = torch.mean(torch.abs(Tx_sig) ** 2)
        print(f"La potenza del segnale trasmesso Tx_sig è: {power_txsig.item()}")
        Tx_sig = Tx_sig.view(shape[0], -1, 2)
        Tx_sig = torch.view_as_complex(Tx_sig)/(torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device))
        power_txsig = torch.mean(torch.abs(Tx_sig) ** 2)
        print(f"La potenza del segnale trasmesso Tx_sig_norm è: {power_txsig.item()}")
 
        Tx_sig_flat = Tx_sig.view(-1)
        
        #H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        #H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        #H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        H_real = torch.normal(0, math.sqrt(1/2), size=[n_rx, n_tx]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[n_rx, n_tx]).to(device)
        H = H_real + 1j * H_imag
        
        norm_fact = torch.linalg.norm(H, 'fro')
        
        # Normalize the matrix
        H = torch.sqrt(torch.tensor(n_rx).to(device))*H / norm_fact
        
        n0_real = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
        n0_imag = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
        n0 = (n0_real + 1j * n0_imag) / torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device)
        
        
        if n_tx != 1:
           # Tx_sig_exp = Tx_sig_flat.unsqueeze(1).repeat(1, min(n_tx,n_rx)).to(device) # Change to unsqueeze and repeat for each layer for correct dimensions
          #  x_tx = torch.matmul(V, Tx_sig_exp.T)
           # convert V in a column vector
            x_tx = torch.matmul(torch.ones((2,1), dtype=torch.complex64).to(device), Tx_sig_flat.view(1,-1))
            power_xtx = torch.mean(torch.abs(x_tx) ** 2)
            print(f"La potenza del segnale x_tx è: {power_xtx.item()}")
            y = torch.matmul(H, x_tx)# + n0 
            power_sigutil = torch.mean(torch.abs(torch.matmul(H, x_tx)) ** 2)
            print(f"La potenza del segnale utile  (H*x_tx) è: {power_sigutil.item()}")
        else:
            x_tx = Tx_sig_flat.view(1,-1)
            y = H*x_tx + n0 
            
        #Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        #Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_array = torch.matmul(torch.inverse(H), y)
        power_rxsig = torch.mean(torch.abs(Rx_array) ** 2)
        print(f"La potenza del segnale trasmesso Rx_array è: {power_rxsig.item()}")
        Rx_array = torch.sum(Rx_array, dim = 0) / torch.tensor(n_rx).to(device)
        power_rxsig = torch.mean(torch.abs(Rx_array) ** 2)
        print(f"La potenza del segnale trasmesso Rx_array post divisione è: {power_rxsig.item()}")

        Rx_array = Rx_array.view(shape[0], int(shape[1] * shape[2] / 2))
        real_part = Rx_array.real*torch.sqrt(torch.tensor(2.0, dtype=torch.float32))  
        imaginary_part = Rx_array.imag*torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    
        Rx_array = torch.stack((real_part, imaginary_part), dim=-1)
        Rx_sig = Rx_array.view(shape)
        power_rxsig = torch.mean(torch.abs(Rx_sig) ** 2)
        print(f"La potenza del segnale Rx_sig è: {power_rxsig.item()}")
        
        #Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig
        
    '''   
    def Rician(self, Tx_sig, n_var, n_tx, n_rx, K=1):
        shape = Tx_sig.shape
        
        Tx_sig = Tx_sig.view(shape[0], -1, 2)
        Tx_sig = torch.view_as_complex(Tx_sig)/(torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device))
 
        Tx_sig_flat = Tx_sig.view(-1)
        
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        #H_real = torch.normal(mean, std, size=[1]).to(device)
        #H_imag = torch.normal(mean, std, size=[1]).to(device)
        #H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        H_real = torch.normal(mean, std, size=[n_rx, n_tx]).to(device)
        H_imag = torch.normal(mean, std, size=[n_rx, n_tx]).to(device)
        H = H_real + 1j * H_imag
        
        norm_fact = torch.linalg.norm(H, 'fro')
        
        # Normalize the matrix
        H = torch.sqrt(torch.tensor(n_rx).to(device))*H / norm_fact
        
        n0_real = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
        n0_imag = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
        n0 = (n0_real + 1j * n0_imag) / torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device)
        
        
        if n_tx != 1:
           # Tx_sig_exp = Tx_sig_flat.unsqueeze(1).repeat(1, min(n_tx,n_rx)).to(device) # Change to unsqueeze and repeat for each layer for correct dimensions
          #  x_tx = torch.matmul(V, Tx_sig_exp.T)
           # convert V in a column vector
            x_tx = torch.matmul(torch.ones((2,1), dtype=torch.complex64).to(device), Tx_sig_flat.view(1,-1))
            y = torch.matmul(H, x_tx) #+ n0 
            power_sigutil = torch.mean(torch.abs(torch.matmul(H, x_tx)) ** 2)
            print(f"La potenza del segnale utile  (H*x_tx) è: {power_sigutil.item()}")

        else:
            x_tx = Tx_sig_flat.view(1,-1)
            y = H*x_tx + n0 
            
        #Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        #Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_array = torch.matmul(torch.inverse(H), y)
        Rx_array = torch.sum(Rx_array, dim = 0) / torch.tensor(n_rx).to(device)

        Rx_array = Rx_array.view(shape[0], int(shape[1] * shape[2] / 2))
        real_part = Rx_array.real*torch.sqrt(torch.tensor(2.0, dtype=torch.float32))  
        imaginary_part = Rx_array.imag*torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    
        Rx_array = torch.stack((real_part, imaginary_part), dim=-1)
        Rx_sig = Rx_array.view(shape)
        
        #Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

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
    
    def CDL_MMSE(self, Tx_sig, n_var, ch_mat, n_tx, n_rx, channel = "CDL_MMSE"):
        
        shape = Tx_sig.shape
        #power_tx_sig = torch.mean(torch.abs(Tx_sig ** 2))
        ### DEBUG
       # print(f"La potenza di Tx_sig è: {power_tx_sig.item()}")
        
        #Get channel matrix + svd
        #invertible = False
        
       # while not invertible:
        _, S, Vh, H = get_mat(ch_mat, n_tx, n_rx, channel)
        V = torch.conj(Vh[0,:]).T.to(device)
        #reshape tx_signal and define real and imaginary parts
        Tx_sig = Tx_sig.view(shape[0], -1, 2)
        Tx_sig = torch.view_as_complex(Tx_sig)/(torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device))

        
        Tx_sig_flat = Tx_sig.view(-1)
        ### DEBUG
       # power_tx_sig_flat = torch.mean(torch.abs(Tx_sig_flat ** 2))
       # print(f"La potenza di Tx_sig_flat (segnale senza precoding) è: {power_tx_sig_flat.item()}")    
        
        n0_real = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
        n0_imag = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
        n0 = (n0_real + 1j * n0_imag) / torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device)
               
        
        if n_tx != 1:
           # Tx_sig_exp = Tx_sig_flat.unsqueeze(1).repeat(1, min(n_tx,n_rx)).to(device) # Change to unsqueeze and repeat for each layer for correct dimensions
          #  x_tx = torch.matmul(V, Tx_sig_exp.T)
           # convert V in a column vector
            V = V.view(-1,1) 
            x_tx = torch.matmul(V, Tx_sig_flat.view(1,-1))
            H_eq = torch.matmul(H, V)
            H_herm = torch.conj(H_eq).T
            eye = torch.eye(H_eq.shape[1] , dtype=torch.complex64).to(device)
            second_part = torch.inverse(torch.matmul(H_herm, H_eq) + (n_var**2) * eye)
            W_MMSE = torch.matmul(second_part, H_herm)
            y = torch.matmul(H, x_tx) + n0 
        else:
            x_tx = V * Tx_sig_flat
            H_eq = torch.matmul(H, V)
            H_herm = torch.conj(H_eq).T
            eye = torch.tensor(1, dtype=torch.complex64).to(device)
            second_part = 1 / (H_herm * H_eq + (n_var**2) * eye)
            W_MMSE = second_part*H_herm
            y = H*x_tx + n0 


        #second_part = torch.inverse(torch.matmul(H_eq, H_herm) + (n_var**2) * eye)
        #W_MMSE = torch.matmul(H_herm, second_part)
        
            
            
            
            # if torch.linalg.det(W_MMSE).abs() != 0:
            #     invertible = True

        ### DEBUG
       # power_x_tx = torch.mean(torch.abs(x_tx) ** 2)
       # print(f"La potenza del segnale utile con precodifica (x_tx) è: {power_x_tx.item()}")        


       # n0_real = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
       # n0_imag = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
       # n0 = (n0_real + 1j * n0_imag) / torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device)

        ### DEBUG
       # power_n0 = torch.mean(torch.abs(n0) ** 2)
       # print(f"La potenza del segnale del rumore (n0) è: {power_n0.item()}")

        ### DEBUG
       # power_sigutil = torch.mean(torch.abs(torch.matmul(H, x_tx)) ** 2)
       # print(f"La potenza del segnale utile  (H*x_tx) è: {power_sigutil.item()}")


        #y = torch.matmul(H, x_tx) + n0 
        x_hat = torch.matmul(W_MMSE, y)
        Rx_array = x_hat
        #Rx_array = torch.round(torch.sum(x_hat, dim=0) / n_rx, decimals=7)
        
        #Torch non supporta il rounding dei numeri complessi, quindi è necessario computarlo separatamente per la parte immaginaria e reale
      #  Rx_array = torch.sum(x_hat, dim=0) / n_rx
       # real_part = torch.round(Rx_array.real, decimals=7)
       # imag_part = torch.round(Rx_array.imag, decimals=7)
       # Rx_array = torch.complex(real_part, imag_part)
        
        #ratio = Rx_array/Tx_sig_flat

    #     #DEBUG Robi
        # tempH = H.cpu().numpy()
        # dfH= pd.DataFrame(tempH)
        # dfH.to_csv("matH.csv", index=False)

        # tempV = V.cpu().numpy()
        # dfV= pd.DataFrame(tempV)
        # dfV.to_csv("matV.csv", index=False)

        # tempX = Tx_sig_flat.view(1,-1).cpu().detach().numpy()
        # dfX= pd.DataFrame(tempX)
        # dfX.to_csv("vetX.csv", index=False)

        # tempN0 = n0.cpu().detach().numpy() 
        # dfN0 = pd.DataFrame(tempN0)      
        # dfN0.to_csv("vetN0.csv", index=False)

        # tempWMSSE = W_MMSE.cpu().detach().numpy()
        # dfMMSE = pd.DataFrame(tempWMSSE)
        # dfMMSE.to_csv("matWMMSE.csv", index = False) 

        # tempY = y.cpu().detach().numpy()
        # dfY = pd.DataFrame(tempY)
        # dfY.to_csv("vetY.csv", index = False) 

        # tempXhat = x_hat .cpu().detach().numpy()
        # dfXhat = pd.DataFrame(tempXhat)
        # dfXhat.to_csv("vetXhat.csv", index = False) 


    #    # END DEBUG 
 
        
        #Reshape the received vector
        #we use half of the multiplication between the second and the third shape values since the original number of symbols was converted to complex values
        Rx_array = Rx_array.view(shape[0], int(shape[1] * shape[2] / 2))
        real_part = Rx_array.real*torch.sqrt(torch.tensor(2.0, dtype=torch.float32))  
        imaginary_part = Rx_array.imag*torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    
        Rx_array = torch.stack((real_part, imaginary_part), dim=-1)
        Rx_sig = Rx_array.view(shape)
        #.to(torch.float32)
        
        
        #save_differences(TX_array, Rx_sig_ar)
        
        return Rx_sig
    '''
    def ch(self, Tx_sig, n_var, ch_mat, n_tx, n_rx, channel):
        
        shape = Tx_sig.shape
        #power_tx_sig = torch.mean(torch.abs(Tx_sig ** 2))
        ### DEBUG
       # print(f"La potenza di Tx_sig è: {power_tx_sig.item()}")
        
        #Get channel matrix + svd
        #invertible = False
        
       # while not invertible:
        U, S, Vh, H = get_mat(ch_mat, n_tx, n_rx, channel)
        V = torch.conj(Vh[0,:]).T.to(device)
        #reshape tx_signal and define real and imaginary parts
        Tx_sig = Tx_sig.view(shape[0], -1, 2)
        Tx_sig = torch.view_as_complex(Tx_sig)/(torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device))

        
        Tx_sig_flat = Tx_sig.view(-1)
        ### DEBUG
       # power_tx_sig_flat = torch.mean(torch.abs(Tx_sig_flat ** 2))
       # print(f"La potenza di Tx_sig_flat (segnale senza precoding) è: {power_tx_sig_flat.item()}")    
        
        n0_real = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
        n0_imag = torch.normal(0, n_var, size=(n_rx, Tx_sig_flat.size(0)), dtype=torch.float32).to(device)
        n0 = (n0_real + 1j * n0_imag) / torch.sqrt(torch.tensor(2.0, dtype=torch.float32)).to(device)
               
        
        if n_tx != 1:
           # Tx_sig_exp = Tx_sig_flat.unsqueeze(1).repeat(1, min(n_tx,n_rx)).to(device) # Change to unsqueeze and repeat for each layer for correct dimensions
          #  x_tx = torch.matmul(V, Tx_sig_exp.T)
           # convert V in a column vector
            V = V.view(-1,1) 
            x_tx = torch.matmul(V, Tx_sig_flat.view(1,-1))
            if channel == "CDL_MMSE" or channel=="Rayleigh":
                H_eq = torch.matmul(H, V)
                H_herm = torch.conj(H_eq).T
                eye = torch.eye(H_eq.shape[1] , dtype=torch.complex64).to(device)
                second_part = torch.inverse(torch.matmul(H_herm, H_eq) + (n_var**2) * eye)
                W_MMSE = torch.matmul(second_part, H_herm)
            y = torch.matmul(H, x_tx) + n0 
        else:
            x_tx = V * Tx_sig_flat
            if channel == "CDL_MMSE" or channel=="Rayleigh":
                H_eq = torch.matmul(H, V)
                H_herm = torch.conj(H_eq).T
                eye = torch.tensor(1, dtype=torch.complex64).to(device)
                second_part = 1 / (H_herm * H_eq + (n_var**2) * eye)
                W_MMSE = second_part*H_herm
            y = H*x_tx + n0 


        if channel == "CDL_MMSE" or channel=="Rayleigh":
            x_hat = torch.matmul(W_MMSE, y)
            Rx_array = x_hat
            SNR_inst = 10*torch.log10(torch.abs(W_MMSE*H*x_tx)**2/torch.abs(W_MMSE*n0)**2).cpu().detach().numpy().mean()
            
        else:
            x_hat = torch.conj(U).T @ y 
            #S = torch.diag(S)
            #S_inv = torch.linalg.pinv(S)
            #S_inv = torch.diag(1.0/S)
            S_inv = torch.tensor(1.0/S, dtype = torch.complex64).to(device)
            if n_tx != 1: 
                x_hat = S_inv@ x_hat
               # Rx_array = np.round((np.sum(x_hat, axis=0) / 2), decimals = 7)
                Rx_array = x_hat
            else:
                Rx_array = S_inv * x_hat
        ratio = Rx_array/Tx_sig_flat
 
        
        #Reshape the received vector
        #we use half of the multiplication between the second and the third shape values since the original number of symbols was converted to complex values
        Rx_array = Rx_array.view(shape[0], int(shape[1] * shape[2] / 2))
        real_part = Rx_array.real*torch.sqrt(torch.tensor(2.0, dtype=torch.float32))  
        imaginary_part = Rx_array.imag*torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    
        Rx_array = torch.stack((real_part, imaginary_part), dim=-1)
        Rx_sig = Rx_array.view(shape)
        #.to(torch.float32)
        
        
        #save_differences(TX_array, Rx_sig_ar)
        
        return Rx_sig, SNR_inst

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
   # if power > 1:
   #     x = torch.div(x, power)
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

def noise_to_SNR(noise_std):
    snr = 1 / (noise_std**2)
    snr = 10*np.log10(snr)
    return snr

def train_step(model, src, trg, n_var, pad, opt, criterion, channel, ch_mat, n_tx, n_rx, mi_net=None, lambda_loss=0.0009 ):
    
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
        SNR_inst = 10*np.log10(1/n_var**2)
    elif channel == "Rician" or channel == "Rayleigh" or channel == "CDL_MMSE":
        Rx_sig, SNR_inst = channels.ch(Tx_sig, n_var, ch_mat, n_tx, n_rx, channel)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    
    # ### DEBUG 
    # diff_sig = Tx_sig - Rx_sig
    # plt.plot(diff_sig.flatten().to("cpu").detach().numpy())
    # plt.show()
    # plt.grid()
    # power_diff_sig = torch.mean(torch.abs(diff_sig)**2)
    # print(f"La potenza di diff_sig è: {power_diff_sig.item()}")           
    # ### END DEBUG


    channel_dec_output = model.channel_decoder(Rx_sig)
    #calls the decoder forward step
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    
    #y_est = x +  torch.matmul(n, torch.inverse(H))
    #loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss_ce = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)

    #mutual information evaluation
    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss_ce + lambda_loss* loss_mine
    # loss = loss_function(pred, trg_real, pad)
    else:
        loss = loss_ce
        loss_mine = 0
        

    loss.backward()
    opt.step()

    return loss.item(), loss_ce.item(), lambda_loss*loss_mine, src[0], pred[0], SNR_inst


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
    elif channel == "Rician" or channel == "Rayleigh" or channel == "CDL_MMSE":
        Rx_sig, _ = channels.ch(Tx_sig, n_var, ch_mat, n_tx, n_rx, channel)
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
    if math.isinf(loss_mine.item()):
        print(f"Mutual information is infinite:{loss_mine.item()} ")
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
    elif channel == "Rician"  or channel == "CDL_MMSE":
        Rx_sig, _ = channels.ch(Tx_sig, n_var, ch_mat, n_tx, n_rx, channel)
    elif channel == "Rayleigh":
        Rx_sig, _ = channels.Rayleigh(Tx_sig, n_var, ch_mat, n_tx, n_rx, channel)
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
    elif channel == "Rician" or channel == "Rayleigh" or channel == "CDL_MMSE":
        Rx_sig, _ = channels.ch(Tx_sig, n_var, ch_mat, n_tx, n_rx, channel)
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



