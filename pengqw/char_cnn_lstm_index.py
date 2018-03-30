#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:36:56 2018

@author: s1700808
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:32:33 2018

@author: pqw1995@163.com
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:03:06 2018

@author: pqw1995@163.com
"""

import sys
sys.path.append('D:\\MLP_Project\\MLP')
import __init__
from config.setting import *
from CYK.plot_fit import plot_fit
import os
import numpy as np
import random as rn
import tensorflow as tf

#os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(2018)
#rn.seed(12345)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout,GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from tensorflow.python.client import device_lib
from nltk.corpus import stopwords
import nltk
from keras.layers import Dense, Input, GlobalMaxPooling1D,Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, EarlyStopping,ModelCheckpoint
from CYK.plot_fit import visialize_model,save_history,plot_all_history
from keras import metrics
from keras.optimizers import SGD
import string
from CYK.data_loader import load_imdb, load_test
from keras.models import load_model

MAX_SEQUENCE_LENGTH = 1014
EMBEDDING_DIM=69
MAX_NUM_WORDS=5000
#earlystopping = EarlyStopping(patience=4)

print('Indexing word vectors.')
#f = open('D:\MLP_Project\glove.6B.100d.txt','r',encoding="utf-8")
#f = open(CBOW_embedding, encoding='utf-8')
#f = open(SkipGram_embedding, encoding='utf-8')
####hello
#f = open('D:\MLP_Project\MLP\\embedding\gensim_word2vec.txt','r',encoding='utf-8')



print('Processing text dataset')

(X_train, y_train), (X_val, y_val) = load_imdb()
X_test, y_test = load_test()


tokenizer= Tokenizer(char_level = True,lower=True)

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index


X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_val = pad_sequences(X_val, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print('Preparing embedding matrix.')
num_words = min(MAX_NUM_WORDS, len(word_index))


model = Sequential()
embedding_layer = Embedding(200,
                            69,
                            input_length=MAX_SEQUENCE_LENGTH,
                            )
#
model.add(embedding_layer)
#print ('###########################################################')
#print ('embedding layer output shape is:',model.output_shape)

model.add(Conv1D(256,
                 7,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=3))

model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=3))


model.add(LSTM(128))

#model.add(Dense(1024,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


#model.add(Conv1D(100,
#                 3,
#                 padding='valid',
#                 activation='relu',
#                 strides=1,input_shape=(MAX_SEQUENCE_LENGTH,num_words)))
##model.add(GlobalMaxPooling1D())
#model.add(MaxPooling1D(pool_size=3))
#
#model.add(Conv1D(100,
#                 3,
#                 padding='valid',
#                 activation='relu',
#                 strides=1))
##model.add(GlobalMaxPooling1D())
#model.add(MaxPooling1D(pool_size=3))
#
#model.add(Conv1D(100,
#                 3,
#                 padding='valid',
#                 activation='relu',
#                 strides=1,input_shape=(MAX_SEQUENCE_LENGTH,num_words)))
#model.add(GlobalMaxPooling1D())

#print ('after maxpooling layer the shape is:',model.output_shape)
##model.add(GlobalMaxPooling1D())
#print ('after maxpooling layer the shape is:',model.output_shape)
#
#model.add(Dense(250,activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1,activation='sigmoid'))


#sgd = SGD(lr=0.01, momentum=0.9)
#tensorBoardCallback = TensorBoard(log_dir='./pqw_logs', write_graph=True)

#filepath='keras_models/char_CNN_1_CNNlayer_2DNN_lessunits_{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5'
filepath='../../MLP_models/char_CNN_LSTM_INDEX.hdf5'

#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_acc', mode = 'max')


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train,y_train , validation_data=(X_val,y_val), epochs=15, batch_size=32,callbacks=[checkpoint])


#history=model.fit_generator(data_generator(X_train, y_train), 
#                            steps_per_epoch=35,epochs=15,verbose=1,
#                            validation_data=val_generator(X_val,y_val),
#                            validation_steps=15, callbacks=[checkpoint])

# Evaluation on the test set
#scores = model.evaluate(encode_data(X_test, MAX_SEQUENCE_LENGTH, vocab, vocab_size, check ), y_test, verbose=0)


#print ('=====================the result for test set==============================')
#print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

print (history.history.keys())

write_filename='char_CNN_LSTM_INDEX.pdf'
save_history(history, 'char_CNN_LSTM_INDEX.csv', subdir='Character_Level_Models_INDEX')

print ('the process for {} is done'.format(write_filename))

new_model = load_model('../../MLP_models/char_CNN_LSTM_INDEX.hdf5')
#new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#scores = new_model.evaluate(X_test,y_test, verbose=0)
scores = new_model.evaluate(X_test, y_test, verbose=0)

print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

with open('new_new_test_result.txt', 'a') as f:
    f.write('\n the model name is {0}, the  best loss on test is: {1}, the acc on test is: {2} \n'.format(write_filename, 
            scores[0],scores[1]))


