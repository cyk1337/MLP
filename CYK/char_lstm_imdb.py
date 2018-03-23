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
import datetime, json, sys
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


# units = 512
#units = 100
units = int(sys.argv[1])
#dropout_rate = 0
dropout_rate = float(sys.argv[2])
maxlen = 1014


print("Building model...")

inputs = Input(shape=( maxlen,vocab_size,))

l1 = LSTM(units,return_sequences=True)(inputs)
l2 = LSTM(units)(l1)
dropout = Dropout(dropout_rate)(l2)
predictions = Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=inputs, outputs=predictions)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



training_batch_size = 500
training_steps = 70

val_batch_size = 500
val_steps = 15

# 35000
train_data_generator = mini_batch_generator(X_train,y_train, vocab, vocab_size, vocab_check, maxlen, batch_size=training_batch_size)
# 7500
val_data_generator = mini_batch_generator(X_train,y_train, vocab, vocab_size, vocab_check, maxlen, batch_size=val_batch_size)


history = model.fit_generator(train_data_generator, steps_per_epoch=training_steps, epochs=15, verbose=1,
                              validation_data=val_data_generator, validation_steps=val_steps
                              )

subdir = 'lstm_char'
plot_filename = 'lstm_char_units{}_dropout_{}.pdf'.format(units, dropout_rate)
# save history info
save_history(history, '{}.csv'.format(plot_filename[:-4]), subdir=subdir)
# save model
# visialize_model(model, filepath=plot_filename)
# save single history
plot_fit(history, plot_filename=plot_filename)

