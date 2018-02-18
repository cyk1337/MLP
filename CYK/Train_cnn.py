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

@file: Train_cnn.py

@time: 13/02/2018 17:18

@desc:         
               
'''

import __init__
from config.setting import *

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Activation, Flatten, Reshape

from CYK.data_loader import load_imdb
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history

MAX_NUM_WORDS=10000


def run_CNN_1layer(Xtrain_matrix, y_train, Xval_matrix, y_val, dropout_rate, plot_filename, subdir):
    filters = 1000
    kernel_size = 4

    # Xtrain_matrix = Xtrain_matrix.reshape((RECORDS_NUM, MAX_NUM_WORDS, 1))
    # Xval_matrix = Xval_matrix.reshape((RECORDS_NUM, MAX_NUM_WORDS, 1))
    print("Building model...")
    model = Sequential()
#    embedding = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, trainable=False)
    # hidden layer 1
    # model.add(embedding)
    # model.add(Dropout(dropout_rate))
    print (MAX_NUM_WORDS)
    model.add(Reshape((MAX_NUM_WORDS, 1,),input_shape=(MAX_NUM_WORDS,)))
    print(model.output_shape)
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, input_shape=(MAX_NUM_WORDS, 1)))
    # temporal maxpooling
    model.add(GlobalMaxPool1D())
    model.add(Dense(150))
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))

    # project and squash it to sigmoid
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(Xtrain_matrix, y_train, epochs=EPOCH_NUM, batch_size=128, validation_data=(Xval_matrix, y_val))

    # save history info
    save_history(history, '{}.csv'.format(plot_filename[:-4]), subdir=subdir)
    # save model
    visialize_model(model, filepath=plot_filename)
    # save single history
    plot_fit(history, plot_filename=plot_filename)


(X_train, y_train), (X_val, y_val) = load_imdb()
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

# count matrix
Xtrain_count = tokenizer.texts_to_matrix(texts=X_train, mode='count')
Xval_count = tokenizer.texts_to_matrix(texts=X_val, mode='count')

"""matrix"""
# # one-hot matrix
# Xtrain_1hot = tokenizer.texts_to_matrix(texts=X_train, mode='binary')
# Xval_1hot = tokenizer.texts_to_matrix(texts=X_val, mode='binary')
#
# # freq matrix
# Xtrain_freq = tokenizer.texts_to_matrix(texts=X_train, mode='freq')
# Xval_freq = tokenizer.texts_to_matrix(texts=X_val, mode='freq')
#
# # tfidf matrix
# Xtrain_tfidf = tokenizer.texts_to_matrix(texts=X_train, mode='tfidf')
# Xval_tfidf = tokenizer.texts_to_matrix(texts=X_val, mode='tfidf')








# subdir to save history
subdir = 'CNN1layer'

dropout_rate = 0.5

# count DNN
run_CNN_1layer(Xtrain_count, y_train, Xval_count, y_val, dropout_rate, 'count_matrix_CNN_hid1_dropout{}.pdf'.format(dropout_rate), subdir)


