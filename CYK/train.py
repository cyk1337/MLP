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

@file: train.py

@time: 10/02/2018 10:28

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


from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Convolution1D, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model




if __name__=='__main__':
    # 1. load pretrained embedding
    embedding_path = CBOW_embedding
    embedding_path = os.path.join(embedding_dir, embedding_path)
    embeddings_index = load_pretrained_model(embedding_path)

    # load data
    (X_train, y_train), (X_val, y_val) = load_imdb()

    # tokenize, filter punctuation, lowercase
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True, char_level=False)

    # val index
    # tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocarb_size = len(tokenizer.word_index) + 1
    print("%d word types" % len(tokenizer.word_index))


    # encoding method 0 : Tokenizer.texts_to_sequence
    # ========================
    train_seq = tokenizer.texts_to_sequences(X_train)
    # print(len(encoded_text))

    word_index = tokenizer.word_index

#    print('index',word_index)

    # print('index len:', len(word_index))
    train_pad_seq = pad_sequences(sequences=train_seq, maxlen=MAX_SEQUENCE_LENGTH)

# check TODO
#     Xtrain_matrix = tokenizer.texts_to_matrix(X_train)
#
#     Xval_matrix = tokenizer.texts_to_matrix(X_val)

    # pad val sequence
    val_seq = tokenizer.texts_to_sequences(X_val)
    val_pad_seq = pad_sequences(sequences=val_seq, maxlen=MAX_SEQUENCE_LENGTH)

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
    model.add(Embedding(num_words,32, input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(250,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # visualize model
    visialize_model(model,filepath='plot_val_DNN.pdf')

    # Log to tensorboard
    tensorBoardCallback = TensorBoard(log_dir=log_dir, write_graph=True)

    # early stopping
    earlystopping = EarlyStopping('val_loss', patience=2)

    history = model.fit(train_pad_seq, y_train, epochs=1, callbacks=[tensorBoardCallback, earlystopping], batch_size=128, validation_data=(val_pad_seq, y_val))

    save_history(history, 'train_val.csv', subdir='val')
    # Evaluation on the val set
    scores = model.evaluate(val_pad_seq, y_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # save performance
    plot_fit(history, plot_filename='DNN_val.pdf',)

    # run tensorboard
    # tensorboard --logdir=logs
    # run_tensorboard(log_dir)



