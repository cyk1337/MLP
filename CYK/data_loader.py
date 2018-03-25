#!/usr/bin/env python

# encoding: utf-8

'''
  
             \ \ / /__| | ___   _ _ __    / ___| | | |  / \  |_ _|
              \ V / _ \ |/ / | | | '_ \  | |   | |_| | / _ \  | | 
               | |  __/   <| |_| | | | | | |___|  _  |/ ___ \ | | 
               |_|\___|_|\_\\__,_|_| |_|  \____|_| |_/_/   \_\___
 ==========================================================================
@author: CYK

@license: School of Informatics, Edinburgh

@contact: s1718204@sms.ed.ac.uk

@file: data_loader.py

@time: 11/02/2018 22:28

@desc:         
               
'''              

import __init__
from config.setting import *

import numpy as np
import pandas as pd
import string

def load_imdb():
    ## train data
    train_data = pd.read_csv(train_csv)
    X_train = train_data['text']
    y_train = train_data['target']
    ## val data
    val_data = pd.read_csv(val_csv)
    X_val = val_data['text']
    y_val = val_data['target']
    return (X_train, y_train), (X_val, y_val)

def load_test():
    # test data
    test_data = pd.read_csv(test_csv)
    X_test = test_data['text']
    y_test = test_data['target']
    return (X_test, y_test)



def mini_batch_generator(x,y, vocab, vocab_size, vocab_check, maxlen, batch_size=128):
    while True:
        for i in range(0, len(x), batch_size):
            x_sample = x[i:i+batch_size]
            y_sample = y[i:i+batch_size]
    
            input_data = encode_data(x_sample, maxlen, vocab, vocab_size, vocab_check)
            yield (input_data, y_sample)


# for character level training below
def encode_data(x, maxlen, vocab, vocab_size, check):
    """
    Iterate over the loaded data and create a matrix of size maxlen x vocab size
    In this case that will be 1014 * x.
    3D matrix = data_sample *  maxlen * vocab_size
    Each character will be encoded into a one-hot vector;
    chars not in the vocab will be encoded into an all zero vector.
    :param x:
    :param maxlen:
    :param vocab:
    :param vocab_size:
    :param check:
    :return:
    """
    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower().replace(' ', ''))
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data



def create_vocab_set():
    """
    alpha bet include 69 chars
    :return:
    """
    alphabet = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)


    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t
    return vocab, reverse_vocab, vocab_size, check



if __name__ == '__main__':
    vocab, reverse_vocab, vocab_size, check = create_vocab_set()


