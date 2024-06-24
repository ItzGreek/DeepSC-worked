from collections import Counter
import json
import os
from torch.utils.data import DataLoader
from dataset import EurDataset, collate_data

class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return str((self.left, self.right))

def huffman_code_tree(node, binString=''):
    '''
    Function to find Huffman Code
    '''
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, binString + '0'))
    d.update(huffman_code_tree(r, binString + '1'))
    return d

def make_tree(nodes):
    '''
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    '''
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]

def indices_to_text(indices, idx_to_token, token_to_idx):
    # Remove the <START> and <END> tokens and convert indices to text
    start_idx = token_to_idx['<START>']
    end_idx = token_to_idx['<END>']
    text = [idx_to_token[idx] for idx in indices if idx in idx_to_token and idx != token_to_idx['<PAD>']]
    # Remove <START> if it is the first token
    if text and text[0] == idx_to_token[start_idx]:
        text = text[1:]
    # Remove <END> if it is the last token
    if text and text[-1] == idx_to_token[end_idx]:
        text = text[:-1]
    return ' '.join(text)

def save_test_batches(test_iterator, file_path, idx_to_token, token_to_idx):
    if os.path.exists(file_path):
        os.remove(file_path)

    for sents in test_iterator:
        with open(file_path, 'a') as f:
            for sent in sents:
                text = indices_to_text(sent.tolist(), idx_to_token, token_to_idx)
                f.write(text + '\n')

def count_ascii_frequencies(sentences):
    ascii_freq = Counter()
    for sentence in sentences:
        for char in sentence:
            if char == ' ':
                char = '<SPACE>'
            ascii_freq[char] += 1
    return ascii_freq

def huffman_encoder(input_file, output_file, codebook):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    encoded_lines = []
    for line in lines:
        encoded_text = ''
        for char in line.strip():
            if char == ' ':
                encoded_text += codebook['<SPACE>']
            else:
                encoded_text += codebook[char]
        encoded_lines.append(encoded_text)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(encoded_lines))

    print(f"Encoded text saved to {output_file}")

def huffman_decoder(encoded_file, output_file, codebook):
    inverse_codebook = {v: k for k, v in codebook.items()}
    with open(encoded_file, 'r', encoding='utf-8') as file:
        encoded_lines = file.readlines()

    decoded_lines = []
    for encoded_text in encoded_lines:
        decoded_text = ''
        current_code = ''
        for bit in encoded_text.strip():
            current_code += bit
            if current_code in inverse_codebook:
                char = inverse_codebook[current_code]
                if char == '<SPACE>':
                    decoded_text += ' '
                else:
                    decoded_text += char
                current_code = ''
        decoded_lines.append(decoded_text)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(decoded_lines))

    print(f"Decoded text saved to {output_file}")

if __name__ == '__main__':
    save_txt = 0
    if save_txt == 1: 
        vocab = json.load(open('europarl/vocab.json', 'rb'))
        token_to_idx = vocab['token_to_idx']
        idx_to_token = {v: k for k, v in token_to_idx.items()}  # Inverted dictionary

        test_eur = EurDataset('test')
        test_iterator = DataLoader(test_eur, batch_size=128, num_workers=0,
                                   pin_memory=True, collate_fn=collate_data)

        test_batch_file = "test_sentences.txt"
        save_test_batches(test_iterator, test_batch_file, idx_to_token, token_to_idx)

    # Read the saved sentences from file
    with open("test_sentences.txt", 'r') as f:
        sentences = f.readlines()
    sentences = [sentence.strip() for sentence in sentences]  # Remove trailing newline characters

    # Count ASCII frequencies, including spaces
    ascii_freq = count_ascii_frequencies(sentences)

    frequencies = {char: freq for char, freq in ascii_freq.items()}
    
    frequencies = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    node = make_tree(frequencies)
    huffman_codebook = huffman_code_tree(node)
    for i in huffman_codebook:
        print(f'{i} : {huffman_codebook[i]}')

    huffman_encoder("test_sentences.txt", "encoded_sentences.txt", huffman_codebook)
    huffman_decoder("encoded_sentences.txt", "decoded_sentences.txt", huffman_codebook)
