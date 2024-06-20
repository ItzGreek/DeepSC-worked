# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
# from bert4keras.backend import keras
# from bert4keras.models import build_bert_model
# from bert4keras.tokenizers import Tokenizer
from w3lib.html import remove_tags
import scipy.io as sp
import spacy


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/20240620-deepsc-CDL_MMSE2x2_singlelayer_80epoch/', type=str)
parser.add_argument('--channel', default='CDL_MMSE', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)
parser.add_argument('--n_rx', default = 2, type = int)
parser.add_argument('--n_tx', default = 2, type = int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# using pre-trained model to compute the sentence similarity
# class Similarity():
#     def __init__(self, config_path, checkpoint_path, dict_path):
#         self.model1 = build_bert_model(config_path, checkpoint_path, with_pool=True)
#         self.model = keras.Model(inputs=self.model1.input,
#                                  outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output)
#         # build tokenizer
#         self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
#
#     def compute_similarity(self, real, predicted):
#         token_ids1, segment_ids1 = [], []
#         token_ids2, segment_ids2 = [], []
#         score = []
#
#         for (sent1, sent2) in zip(real, predicted):
#             sent1 = remove_tags(sent1)
#              sent2 = remove_tags(sent2)
#
#             ids1, sids1 = self.tokenizer.encode(sent1)
#             ids2, sids2 = self.tokenizer.encode(sent2)
#
#             token_ids1.append(ids1)
#             token_ids2.append(ids2)
#             segment_ids1.append(sids1)
#             segment_ids2.append(sids2)
#
#         token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
#         token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')
#
#         segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
#         segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')
#
#         vector1 = self.model.predict([token_ids1, segment_ids1])
#         vector2 = self.model.predict([token_ids2, segment_ids2])
#
#         vector1 = np.sum(vector1, axis=1)
#         vector2 = np.sum(vector2, axis=1)
#
#         vector1 = normalize(vector1, axis=0, norm='max')
#         vector2 = normalize(vector2, axis=0, norm='max')
#
#         dot = np.diag(np.matmul(vector1, vector2.T))  # a*b
#         a = np.diag(np.matmul(vector1, vector1.T))  # a*a
#         b = np.diag(np.matmul(vector2, vector2.T))
#
#         a = np.sqrt(a)
#         b = np.sqrt(b)
#
#         output = dot / (a * b)
#         score = output.tolist()
#
#         return score

# define similarity using spacy library instead of bertmodel

class Similarity():
    def __init__(self):
        spacy.prefer_gpu()
       # self.nlp = spacy.load("en_core_web_sm")
        self.nlp = spacy.load("en_core_web_lg")

    def compute_similarity(self, real_list, predicted_list):
        
        similarity_score = []  
        for real, predicted in zip(real_list, predicted_list):
            doc1 = self.nlp(real)
            doc2 = self.nlp(predicted)

            # Calculate similarity
            similarity_score.append(doc1.similarity(doc2))
        return similarity_score

#Perofrmance evaluation
def performance(args, SNR, net, file):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    similarity = Similarity()
    #set the weight on 1-grams to 1
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    
    mat_file = sp.loadmat('/home/man2mob/PythonStuff/DeepSC/Hmat/H_1.mat')
    Htot = mat_file['Htot']

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    
    #turn the indices into words
    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel, Htot, args.n_tx, args.n_rx)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(target_word)
                Rx_word.append(word)

            bleu_score = []
            sim_score = []
            count = 0
            for sent1, sent2 in tqdm(zip(Tx_word, Rx_word), total=len(Tx_word)):
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
                sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
                #Write TX and RX message in the output file
                file.write(f"**************  SNR: {SNR[count]} dB  **************\n")
                for tx, rx in zip(sent1, sent2):
                    file.write(f"Transmitted: {tx}\n")
                    file.write(f"Received: {rx}\n\n")
                file.write("\n\n")
                count += 1
                    
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)


            sim_score = np.array(sim_score)
            sim_score = np.mean(sim_score, axis=1)
            score2.append(sim_score)

    bleu_scores = np.mean(np.array(score), axis=0)
    sim_scores = np.mean(np.array(score2), axis=0)

    return bleu_scores, sim_scores

if __name__ == '__main__':
    args = parser.parse_args()
    #Tested SNRs
    SNR = [0,3,6,9,12,15,18]
    args.vocab_file = '/home/man2mob/PythonStuff/DeepSC/data/' + args.vocab_file
   
    "args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file"
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    model_paths = []
    #iterates through checkpoint files
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx)) #store the path

    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

    model_path, _ = model_paths[-1] #selects the last model
    checkpoint = torch.load(model_path) 
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    #bleu computation
    with open('performance_results.txt', 'w') as file:
       # BLEU computation
       bleu_score, similarity_score = performance(args, SNR, deepsc, file)
       file.write(f"BLEU scores: {bleu_score}\n")
       file.write(f"Similarity scores: {similarity_score}\n")

    print(f"bleu_score: {bleu_score} \nsimilarity_score: {similarity_score}   ")
   
   # similarity.compute_similarity(sent1, real)
