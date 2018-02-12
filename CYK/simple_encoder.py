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
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout

from CYK.data_loader import load_imdb
from CYK.plot_fit import plot_fit, visialize_model

def run_DNN_2layer(Xtrain_matrix, y_train, Xtest_matrix, y_test, plot_filename):
    print("Building model...")
    model = Sequential()
    # hidden layer 1
    model.add(Dense(250, activation='relu', input_shape=(Xtrain_matrix.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(Xtrain_matrix, y_train, epochs=10, batch_size=64, validation_data=(Xtest_matrix, y_test))
    visialize_model(model,filepath=plot_filename)
    plot_fit(history, plot_filename=plot_filename)



(X_train, y_train), (X_test, y_test) = load_imdb()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# count matrix
Xtrain_count = tokenizer.texts_to_matrix(texts=X_train, mode='count')
Xtest_count = tokenizer.texts_to_matrix(texts=X_test, mode='count')

# one-hot matrix
Xtrain_1hot = tokenizer.texts_to_matrix(texts=X_train, mode='binary')
Xtest_1hot = tokenizer.texts_to_matrix(texts=X_test, mode='binary')

# freq matrix
Xtrain_freq = tokenizer.texts_to_matrix(texts=X_train, mode='freq')
Xtest_freq = tokenizer.texts_to_matrix(texts=X_test, mode='freq')

# tfidf matrix
Xtrain_tfidf = tokenizer.texts_to_matrix(texts=X_train, mode='tfidf')
Xtest_tfidf = tokenizer.texts_to_matrix(texts=X_test, mode='tfidf')

# count DNN
run_DNN_2layer(Xtrain_count, y_train, Xtest_count, y_test, 'count_matrix_DNN_hid1_dropout0.5.pdf')

# freq
run_DNN_2layer(Xtrain_freq, y_train, Xtest_freq, y_test, 'freq_matrix_DNN_hid1_dropout0.5.pdf')

# one-hot
run_DNN_2layer(Xtrain_1hot, y_train, Xtest_1hot, y_test, '1hot_matrix_DNN_hid1_dropout0.5.pdf')

# tfidf DNN
run_DNN_2layer(Xtrain_tfidf, y_train, Xtest_tfidf, y_test, 'tfidf_matrix_DNN_hid1dropout0.5.pdf')


