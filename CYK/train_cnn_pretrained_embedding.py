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

@file: train_cnn_pretrained_embedding.py

@time: 13/02/2018 20:10

@desc:         
               
'''              

import __init__
from config.setting import *
from CYK.plot_fit import plot_fit, visialize_model, save_history
from CYK.help import run_tensorboard
from CYK.data_loader import load_imdb
from CYK.embedding_loader import load_pretrained_model

import numpy as np
import pandas as pd
import os, re


import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Activation, Flatten, Reshape

from CYK.data_loader import load_imdb
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history


def run_CNN_pretrianed_embedding(Xtrain, y_train, Xtest, y_test, embedding_matrix, dropout_rate, plot_filename, subdir):
    filters = 64
    kernel_size = 5

    # Xtrain_matrix = Xtrain_matrix.reshape((RECORDS_NUM, MAX_NUM_WORDS, 1))
    # Xtest_matrix = Xtest_matrix.reshape((RECORDS_NUM, MAX_NUM_WORDS, 1))
    print("Building model...")
    model = Sequential()
    embedding = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH,
                          weights=[embedding_matrix],
                          trainable=False
                          )
    # hidden layer 1
    model.add(embedding)
    model.add(Dropout(dropout_rate))
    # model.add(Reshape((MAX_NUM_WORDS, 1,),input_shape=(MAX_NUM_WORDS,)))
    # print(model.output_shape)

    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1,
                     input_shape=(MAX_NUM_WORDS, 1)))
    # temporal maxpooling
    model.add(GlobalMaxPool1D())
    model.add(Dense(250))
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))

    # project and squash it to sigmoid
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(Xtrain, y_train, epochs=EPOCH_NUM, batch_size=64, validation_data=(Xtest, y_test))

    # save history info
    save_history(history, '{}.csv'.format(plot_filename[:-4]), subdir=subdir)
    # save model
    visialize_model(model, filepath=plot_filename)
    # save single history
    plot_fit(history, plot_filename=plot_filename)


if __name__=='__main__':
    # 1. load pretrained embedding
    embeddings = ['cbow', 'skipgram', 'glove']
    embeddings_path = [CBOW_embedding, SkipGram_embedding, Glove_embedding]
    # embedding index
    embedding_num = 0
    embedding_path = os.path.join(embedding_dir, embeddings_path[embedding_num] )
    embeddings_index = load_pretrained_model(embedding_path)

    # load data
    (X_train, y_train), (X_test, y_test) = load_imdb()

    # tokenize, filter punctuation, lowercase
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True, char_level=False)

    # test index
    # tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocarb_size = len(tokenizer.word_index) + 1
    print("%d word types" % len(tokenizer.word_index))


    # encoding method 0 : Tokenizer.texts_to_sequence
    # ========================
    train_seq = tokenizer.texts_to_sequences(X_train)
    # print(len(encoded_text))

    word_index = tokenizer.word_index

    # print('index len:', len(word_index))
    train_pad_seq = pad_sequences(sequences=train_seq, maxlen=MAX_SEQUENCE_LENGTH)

    # pad test sequence
    test_seq = tokenizer.texts_to_sequences(X_test)
    test_pad_seq = pad_sequences(sequences=test_seq, maxlen=MAX_SEQUENCE_LENGTH)

    # labels = to_categorical(np.asarray(y_train))
    print("padding sequnce(X_input) shape:", train_pad_seq.shape)
    # print("target(y_train) shape:", labels.shape)
    print('-'*80)

    # Embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector =  embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    # subdir to save history
    subdir = 'CNN1layer_pretrained'

    dropout_rate = 0.5


    # count DNN
    run_CNN_pretrianed_embedding(train_pad_seq, y_train, test_pad_seq, y_test,embedding_matrix, dropout_rate, 'pretrained_{}_CNN_hid1_dropout{}.pdf'.format(embeddings[embedding_num], dropout_rate), subdir)


