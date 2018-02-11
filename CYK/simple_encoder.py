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
from keras.layers import Dense

from CYK.data_loader import load_imdb
from CYK.plot_fit import plot_fit

(X_train, y_train), (X_test, y_test) = load_imdb()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
Xtrain_matrix = tokenizer.texts_to_matrix(texts=X_train, mode='count')
Xtest_matrix = tokenizer.texts_to_matrix(texts=X_test, mode='count')



print("Building model...")
model = Sequential()

model.add(Dense(250, activation='relu', input_shape=(Xtrain_matrix.shape[1], )))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(Xtrain_matrix, y_train, epochs=5, batch_size=64, validation_data=(Xtest_matrix, y_train))

plot_fit(history, plot_filename='Count_matrix_test.pdf')