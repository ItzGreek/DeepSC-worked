from collections import Counter
import json
import os
from torch.utils.data import DataLoader
from dataset import EurDataset, collate_data
from dahuffman import HuffmanCodec

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
            ascii_val = ord(char)
            if ascii_val == 32:  # ASCII code for space
                char = '<SPACE>'  # Use a placeholder for space
            ascii_freq[ascii_val] += 1
    return ascii_freq

if __name__ == '__main__':
    vocab = json.load(open('europarl/vocab.json', 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = {v: k for k, v in token_to_idx.items()}  # Inverted dictionary

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=128, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    test_batch_file = "test_sentences.txt"
    save_test_batches(test_iterator, test_batch_file, idx_to_token, token_to_idx)

    # Read the saved sentences from file
    with open(test_batch_file, 'r') as f:
        sentences = f.readlines()
    sentences = [sentence.strip() for sentence in sentences]  # Remove trailing newline characters

    # Count ASCII frequencies, including spaces
    ascii_freq = count_ascii_frequencies(sentences)

    frequencies = {chr(ascii_val): freq for ascii_val, freq in ascii_freq.items()}

    # Create Huffman codec from frequencies
    huffman_codec = HuffmanCodec.from_frequencies(frequencies)

    # Encode sentences and save encoded output
    encoded_output_file = "huffman_encoded_sentences.txt"
    with open(encoded_output_file, 'w') as f_encoded:
        for sentence in sentences:
            encoded_chars = []
            for char in sentence:
                encoding_bits = ''.join(f'{b:08b}' for b in huffman_codec.encode(char))
                encoded_chars.append(encoding_bits)
            encoded_sentence = ''.join(encoded_chars)
            f_encoded.write(encoded_sentence + '\n')

    print(f"Encoded sentences saved to {encoded_output_file}")
