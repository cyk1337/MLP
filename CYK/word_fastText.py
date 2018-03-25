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

@file: fastText.py

@time: 17/02/2018 16:45

@desc:         
               
'''         
import sys
# sys.path.append('D:\\MLP_Project\\MLP')
import __init__
from config.setting import *
from CYK.plot_fit import plot_fit
import os

     
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    #>>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    #>>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    #>>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    #>>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    #>>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    #>>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    #>>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    #>>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 2
max_features = 20000
maxlen = 400
batch_size = 64
embedding_dims = 50
epochs = 15

print('Loading data...')
from CYK.data_loader import load_imdb, load_test
(X_train, y_train), (X_val, y_val) = load_imdb()
(X_test, y_test)=load_test()
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)
x_train = tokenizer.texts_to_sequences(X_train)
x_val = tokenizer.texts_to_sequences(X_val)
x_test = tokenizer.texts_to_sequences(X_test)


print(len(x_train), 'train sequences')
print(len(x_val), 'val sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average val sequence length: {}'.format(np.mean(list(map(len, x_val)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_val, x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_val = add_ngram(x_val, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average val sequence length: {}'.format(np.mean(list(map(len, x_val)), dtype=int)))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    trainable=True))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document

model.add(GlobalAveragePooling1D())

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath = os.path.join(best_model_dir, 'bigram_fasttext_test')
save_best_point = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          callbacks=[save_best_point])

plot_filename = 'fasttext_bigram'
subdir = 'fastText'
# save history info
save_history(history, '{}.csv'.format(plot_filename), subdir=subdir)
# save model
# visialize_model(model, filepath="{}.pdf".format(plot_filename))
# save single history
# plot_fit(history, "{}.pdf".format(plot_filename))

from keras.models import load_model
#filepath = os.path.join(best_model_dir, 'bigram_fasttext_testset')
model = load_model(filepath)
scores = model.evaluate(x_test,y_test)
score={}
score[model.metrics_names[0]] = scores[0]
score[model.metrics_names[1]] = scores[1]
with open('bigram_fasttext_test', 'w') as f:
    f.write(str(score))
print(score)