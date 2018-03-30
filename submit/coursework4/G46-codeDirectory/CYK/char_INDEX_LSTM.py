#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys

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
embeddings_index = {}
#f = open('D:\MLP_Project\glove.6B.100d.txt','r',encoding="utf-8")
#f = open(CBOW_embedding, encoding='utf-8')
#f = open(SkipGram_embedding, encoding='utf-8')
####hello
#f = open('D:\MLP_Project\MLP\\embedding\gensim_word2vec.txt','r',encoding='utf-8')



print('Processing text dataset')

(X_train, y_train), (X_val, y_val) = load_imdb()
X_test, y_test = load_test()

#train_data=pd.read_csv(train_csv)
#test_data=pd.read_csv(test_csv)

tokenizer= Tokenizer(char_level = True)
#
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


#embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
#for word, i in word_index.items():
#    if i >= MAX_NUM_WORDS:
#        continue
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix[i] = embedding_vector



model = Sequential()
embedding_layer = Embedding(200,
                            69,
                            input_length=MAX_SEQUENCE_LENGTH
                            )
model.add(embedding_layer)
print ('###########################################################')
print ('embedding layer output shape is:',model.output_shape)


#model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(1,activation='sigmoid'))


#sgd = SGD(lr=0.01, momentum=0.9)
#tensorBoardCallback = TensorBoard(log_dir='./pqw_logs', write_graph=True)

#filepath='keras_models/char_CNN_1_CNNlayer_2DNN_lessunits_{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5'
filepath='../../MLP_models/char_LSTM(1layer)_128.hdf5'

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

write_filename='char_LSTM(1layer)_128_new.pdf'
save_history(history, 'char_LSTM(1layer)_128_new.csv', subdir='Character_Level_Models_Keras_True')

print ('the process for {} is done'.format(write_filename))


new_model = load_model('../../MLP_models/char_LSTM(1layer)_128.hdf5')
#new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#scores = new_model.evaluate(X_test,y_test, verbose=0)
scores = new_model.evaluate(X_test, y_test, verbose=0)

print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

with open('new_new_test_result.txt', 'a') as f:
    f.write('\n the model name is {0}, the  best loss on test is: {1}, the acc on test is: {2} \n'.format(write_filename, 
            scores[0],scores[1]))


