import os
import argparse
import pickle as pkl
import pandas as pd
import sys
import torch

# Local imports
import utils

data_examples = []
with open('/content/data.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        if len(data_examples) < 10000:
            data_examples.append(line)
        else:
            break
df = pd.DataFrame(data_examples, columns=['all'])
df=df.astype(str)
df[['english','transformated_form']] = df['all'].loc[df['all'].str.split().str.len() == 2].str.split(expand=True)
df=df.drop(columns=['all']).head()
words_all = list(df['english'])
actual_trans = list(df['transformated_form'])
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
        print('{} --> {}'.format(word, translated))

        # utils.visualize_attention(word,
        #                           encoder,
        #                           decoder,
        #                           idx_dict,
        #                           opts,
        #                           save=os.path.join(opts.load, '{}.pdf'.format(word)))
        
    c=0
    for i in range(len(trans)):
        if trans[i] == actual[i]:
            c+=1
    
    print('Accuracy on test set is',float(c/len(trans)))