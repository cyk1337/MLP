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

@file: char_lstm.py

@time: 21/03/2018 21:02

@desc:         
               
'''
import __init__
from config.setting import *

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import datetime, json
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Convolution1D, \
    Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Activation, \
    Flatten, Reshape, LSTM

from CYK.data_loader import load_imdb, create_vocab_set, encode_data, mini_batch_generator, load_test
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history



print('Loading data...')
#Expect x to be a list of sentences. Y to be a one-hot encoding of the chars

# Using keras to load the dataset with the top_words
(X_train, y_train), (X_val, y_val) = load_imdb()
(X_test, y_test) = load_test()

print('Creating vocab ...')
vocab, reverse_vocab, vocab_size, vocab_check = create_vocab_set()

#Maximum length. Longer gets chopped. Shorter gets padded.
# maxlen = 1014
# train_data = encode_data(X_train, maxlen, vocab, vocab_size, vocab_check)
# val_data = encode_data(X_val, maxlen, vocab, vocab_size, vocab_check)
# test_data = encode_data(X_test, maxlen, vocab, vocab_size, vocab_check)


print('Build model...')
#Model params
#Filters for conv layers


units = 512
dropout_rate = 0

maxlen = 1014


# Xtrain_matrix = Xtrain_matrix.reshape((RECORDS_NUM, MAX_NUM_WORDS, 1))
# Xval_matrix = Xval_matrix.reshape((RECORDS_NUM, MAX_NUM_WORDS, 1))
print("Building model...")

inputs = Input(shape=(maxlen, vocab_size,))

l1 = LSTM(units) (inputs)
l2 = LSTM(units) (l1)

predictions = Dense(1, activation='sigmoid',dropout_rate=dropout_rate)(l2)

model = Model(inputs=inputs, outputs=predictions)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




# 35000
train_data_generator = mini_batch_generator(X_train,y_train, vocab, vocab_size, vocab_check, maxlen, batch_size=2500)
# 7500
val_data_generator = mini_batch_generator(X_train,y_train, vocab, vocab_size, vocab_check, maxlen, batch_size=2500)


history = model.fit_generator(train_data_generator, steps_per_epoch=14, epochs=15, verbose=1,
                              validation_data=val_data_generator, validation_steps=3,
                              # use_multiprocessing=True
                              )

subdir = 'cnn_char'
plot_filename = 'cnn_char.pdf'
# save history info
save_history(history, '{}.csv'.format(plot_filename[:-4]), subdir=subdir)
# save model
visialize_model(model, filepath=plot_filename)
# save single history
plot_fit(history, plot_filename=plot_filename)

