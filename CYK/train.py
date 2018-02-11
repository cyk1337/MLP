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

@file: test.py

@time: 10/02/2018 10:28

@desc:

'''

import __init__
from config.setting import *
from CYK.plot_fit import plot_fit
from CYK.help import run_tensorboard


import numpy as np
import pandas as pd
import os, re


from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Convolution1D, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard, EarlyStopping


# read from datafile


# encoding method 1: load pre-trained embedding

# build mapping for pretrained models
# dict {word->vector}
# ====================
def load_pretrained_model(embedding_path):
    embedding_index = dict()
    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word.isdigit(): continue
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
    print("Found %s word vectors" % len(embedding_index))
    return embedding_index





if __name__=='__main__':
    # 1. load pretrained embedding
    embedding_path = CBOW_embedding
    embedding_path = os.path.join(embedding_dir, embedding_path)
    embeddings_index = load_pretrained_model(embedding_path)

    # 2. prepare training data and labels

    ## train data
    train_data = pd.read_csv(train_csv)
    X_train = train_data['text']
    y_train = train_data['target']
    # test data
    test_data = pd.read_csv(test_csv)
    X_test = test_data['text']
    y_test = test_data['target']

    # tokenize, filter punctuation, lowercase
    # tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True, char_level=False)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocarb_size = len(tokenizer.word_index) + 1
    print("%d word types" % len(tokenizer.word_index))


    # encoding method 0 : Tokenizer.texts_to_sequence
    # ========================
    train_seq = tokenizer.texts_to_sequences(X_train)
    # print(len(encoded_text))

    word_index = tokenizer.word_index
    print('index',word_index)
    train_pad_seq = pad_sequences(sequences=train_seq, maxlen=MAX_SEQUENCE_LENGTH)

# check TODO
    Xtrain_matrix = tokenizer.texts_to_matrix(X_train)

    Xtest_matrix = tokenizer.texts_to_matrix(X_test)

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

# TODO
    # input length
    # embedding_layer = Embedding(num_words, EMBEDDING_DIM,
    #                             weights=[embedding_matrix],
    #                             input_length=MAX_SEQUENCE_LENGTH,
    #                             trainable = False
    #                             )

    model = Sequential()
    # model.add(embedding_layer)
    #
    # model.add(Conv1D(64, 3, activation='relu', input_shape=(None, 100)))
    # model.add(Conv1D(64, 3, activation='relu'))
    # model.add(MaxPooling1D(3))
    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # model=Sequential()
    model.add(Embedding(num_words,32,input_length=MAX_SEQUENCE_LENGTH))
    model.add(Flatten())
    model.add(Dense(250,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Log to tensorboard
    tensorBoardCallback = TensorBoard(log_dir=log_dir, write_graph=True)

    # early stopping
    earlystopping = EarlyStopping('val_loss', patience=2)

    history = model.fit(train_pad_seq, y_train, epochs=5, callbacks=[tensorBoardCallback, earlystopping], batch_size=64, validation_data=(test_pad_seq, y_test))

    # Evaluation on the test set
    scores = model.evaluate(test_pad_seq, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # save performance
    plot_fit(history, plot_filename='DNN_test.pdf',)

    # run tensorboard
    # tensorboard --logdir=logs
    # run_tensorboard(log_dir)



