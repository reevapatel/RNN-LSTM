import os
import argparse
import pickle as pkl
import sys

import torch

# Local imports
import utils

def read_lines(filename):
    """Read a file and split it into lines.
    """
    lines = open(filename).read().strip().lower().split('\n')
    return lines


def read_pairs(filename):
    """Reads lines that consist of two words, separated by a space.

    Returns:
        source_words: A list of the first word in each line of the file.
        target_words: A list of the second word in each line of the file.
    """
    lines = read_lines(filename)
    source_words, target_words = [], []
    for line in lines:
        line = line.strip()
        if line:
            source, target = line.split()
            source_words.append(source)
            target_words.append(target)
    return source_words, target_words


def all_alpha_or_dash(s):
    """Helper function to check whether a string is alphabetic, allowing dashes '-'.
    """
    return all(c.isalpha() or c == '-' for c in s)


def filter_lines(lines):
    """Filters lines to consist of only alphabetic characters or dashes "-".
    """
    return [line for line in lines if all_alpha_or_dash(line)]


def load_data():
    """Loads (English, Pig-Latin) word pairs, and creates mappings from characters to indexes.
    """

    source_lines, target_lines = read_pairs('data.txt')

    # Filter lines
    source_lines = filter_lines(source_lines)
    target_lines = filter_lines(target_lines)

    all_characters = set(''.join(source_lines)) | set(''.join(target_lines))

    # Create a dictionary mapping each character to a unique index
    char_to_index = { char: index for (index, char) in enumerate(sorted(list(all_characters))) }

    # Add start and end tokens to the dictionary
    start_token = len(char_to_index)
    end_token = len(char_to_index) + 1
    char_to_index['SOS'] = start_token
    char_to_index['EOS'] = end_token

    # Create the inverse mapping, from indexes to characters (used to decode the model's predictions)
    index_to_char = { index: char for (char, index) in char_to_index.items() }

    # Store the final size of the vocabulary
    vocab_size = len(char_to_index)

    line_pairs = list(set(zip(source_lines, target_lines)))  # Python 3

    idx_dict = { 'char_to_index': char_to_index,
                 'index_to_char': index_to_char,
                 'start_token': start_token,
                 'end_token': end_token }

    return line_pairs, vocab_size, idx_dict

line_pairs,vocab_size,idx_dict = load_data()
words_all = [i[0] for i in line_pairs]
actual_trans = [i[1] for i in line_pairs]
num_lines = int(0.8*len(words_all))
words = words_all[num_lines:]
actual = actual_trans[num_lines:]





def load(opts):
    encoder = torch.load(os.path.join(opts.load, 'encoder.pt'))
    decoder = torch.load(os.path.join(opts.load, 'decoder.pt'))
    idx_dict = pkl.load(open(os.path.join(opts.load, 'idx_dict.pkl'), 'rb'))
    return encoder, decoder, idx_dict


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Path to checkpoint directory.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use GPU.')
    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    if sys.version_info[0] == 3:
        opts.load = 'checkpoints/h10-bs16'
        
    encoder, decoder, idx_dict = load(opts)

    trans=[]
    for word in words:
        translated = utils.translate(word,
                                     encoder,
                                     decoder,
                                     idx_dict,
                                     opts)
        trans.append(translated)
        # print('{} --> {}'.format(word, translated))

        utils.visualize_attention(word,
                                  encoder,
                                  decoder,
                                  idx_dict,
                                  opts,
                                  save=os.path.join(opts.load, '{}.pdf'.format(word)))
        
    c=0
    for i in range(len(trans)):
        if trans[i] == actual[i]:
            c+=1
    
    print('Accuracy on test set is',float(c/len(trans)))