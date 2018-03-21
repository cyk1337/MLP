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

@file: simple_encoder.py

@time: 11/02/2018 22:34

@desc:         
               
'''
import __init__
from config.setting import *

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Activation
from keras import regularizers

from CYK.data_loader import load_imdb, load_test
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history


def run_DNN_2layer(Xtrain_matrix, y_train, Xval_matrix, y_val,dropout_rate, plot_filename, subdir):
    print("Building model...")
    model = Sequential()
    # hidden layer 1
    model.add(Dense(250, activation='relu', input_shape=(Xtrain_matrix.shape[1],),

                    ))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(Xtrain_matrix, y_train, epochs=EPOCH_NUM, batch_size=128, validation_data=(Xval_matrix, y_val))

    # save history info
    save_history(history, '{}.csv'.format(plot_filename[:-4]), subdir=subdir)
    # save model
    visialize_model(model,filepath=plot_filename)
    # save single history
    plot_fit(history, plot_filename=plot_filename)





(X_train, y_train), (X_val, y_val) = load_imdb()
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


# count matrix

Xtrain_count = tokenizer.texts_to_matrix(texts=X_train, mode='count')
Xval_count = tokenizer.texts_to_matrix(texts=X_val, mode='count')


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
subdir = 'SimpleEnc'

dropout_rate = 0.5

# count DNN
run_DNN_2layer(Xtrain_count, y_train, Xval_count, y_val, dropout_rate, 'count_matrix_DNN_hid1_dropout{}.pdf'.format(dropout_rate), subdir)

# freq
# run_DNN_2layer(Xtrain_freq, y_train, Xval_freq, y_val, dropout_rate, 'freq_matrix_DNN_hid1_dropout{}.pdf'.format(dropout_rate), subdir)
#
# # one-hot
# run_DNN_2layer(Xtrain_1hot, y_train, Xval_1hot, y_val, dropout_rate, '1hot_matrix_DNN_hid1_dropout{}.pdf'.format(dropout_rate), subdir)
#
# # tfidf DNN
# run_DNN_2layer(Xtrain_tfidf, y_train, Xval_tfidf, y_val, dropout_rate, 'tfidf_matrix_DNN_hid1dropout{}.pdf'.format(dropout_rate), subdir)


# plot_all_history(subdir, plot_filename='Plot_all_DNN_1layer_dropout{}.pdf'.format(dropout_rate))