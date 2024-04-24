# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: text_preprocess.py
@Time: 2021/3/31 22:14
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:44:08 2020

@author: hx301
"""
import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='europarl/en', type=str)
parser.add_argument('--output-train-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--output-test-dir', default='europarl/test_data.pkl', type=str)
parser.add_argument('--output-vocab', default='europarl/vocab.json', type=str)

SPECIAL_TOKENS = {
  '<PAD>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}

def unicode_to_ascii(s):
    #filter out diacritic marks from unicode text
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    # normalize unicode characters
    s = unicode_to_ascii(s)
    # remove the XML-tags
    s = remove_tags(s)
    # add white space before !.?
    s = re.sub(r'([!.?])', r' \1', s) #place a whitespace before [!.?]
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s) #replace every character not mentioned in [^a-zA-Z.!?] with a whitespace
    s = re.sub(r'\s+', r' ', s) #replace every space sequence with a single whitespace
    # change to lower letter
    s = s.lower()
    return s

def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    cutted_lines = list()
    #check if each line is longer than 4 words and shorter than 30 words
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines

def save_clean_sentences(sentence, save_path):
    pickle.dump(sentence, open(save_path, 'wb'))
    print('Saved: %s' % save_path)

def process(text_path):
    global sentences
    global raw_data
    #open the selected file
    fop = open(text_path, 'r', encoding='utf8')
    raw_data = fop.read()

    #remove leading and trailing spaces from the sentences and split the file in a list of sentences based on the newline character
    sentences = raw_data.strip().split('\n')
    #call the normalize_string function previously defined for each one of the sentences
    raw_data_input = [normalize_string(data) for data in sentences]
    #filter out lines longer than 30 words and shorter than 4
    raw_data_input = cutted_data(raw_data_input)
    fop.close()

    return raw_data_input


def tokenize(s, delim=' ',  add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, token_to_idx = { }, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, ):
    token_to_count = {}

    for seq in sequences:
        #tokenize the sequence removing or keeping some punctuation marks
      seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                      punct_to_remove=punct_to_remove,
                      add_start_token=False, add_end_token=False)
      for token in seq_tokens:
         #count the number of appearences of each token
        if token not in token_to_count:
          token_to_count[token] = 0
        token_to_count[token] += 1

    for token, count in sorted(token_to_count.items()):
        #exclude tokens that appear just once and assign idx to each token
      if count >= min_token_count:
        token_to_idx[token] = len(token_to_idx)

    return token_to_idx

#not used
def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
      if token not in token_to_idx:
        if allow_unk:
          token = '<UNK>'
        else:
          raise KeyError('Token "%s" not in vocab' % token)
      seq_idx.append(token_to_idx[token])
    return seq_idx

#not used
def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
      tokens.append(idx_to_token[idx])
      if stop_at_end and tokens[-1] == '<END>':
        break
    if delim is None:
      return tokens
    else:
      return delim.join(tokens)


def main(args):
    #setting up input and output directories
    data_dir = 'C:/Users/39392/Desktop/Thesis - Semcom/DeepSC/DeepSC-master/DeepSC-master/'
    "data_dir = '/import/antennas/Datasets/hx301/'"
    args.input_data_dir = data_dir + args.input_data_dir
    args.output_train_dir = data_dir + args.output_train_dir
    args.output_test_dir = data_dir + args.output_test_dir
    args.output_vocab = data_dir + args.output_vocab

    print(args.input_data_dir)
    
    sentences = []
    print('Preprocess Raw Text')
    #check each file in the input data directory
    for fn in tqdm(os.listdir(args.input_data_dir)): #tdqm just diplays a progress bar
        if not fn.endswith('.txt'): continue
        process_sentences = process(os.path.join(args.input_data_dir, fn))
        sentences += process_sentences

    # remove the same sentences
    a = {}
    for set in sentences:
        #check if a sentence is already in the dictionary
        if set not in a:
            a[set] = 0
        a[set] += 1
    #Extract unique sentences by taking the keys of dictionary 'a' and converting them into a list
    sentences = list(a.keys())
    print('Number of sentences: {}'.format(len(sentences)))
    
    print('Build Vocab')
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS,
        punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )

    vocab = {'token_to_idx': token_to_idx}
    print('Number of words in Vocab: {}'.format(len(token_to_idx)))

    # save the vocab
    if args.output_vocab != '':
        with open(args.output_vocab, 'w') as f:
            json.dump(vocab, f)

    print('Start encoding txt')
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)


    print('Writing Data')
    #train-test 90-10 split
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9):]

    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)