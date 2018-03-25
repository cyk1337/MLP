#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
#sys.path.append('D:\\MLP_Project\\MLP')
import __init__
from config.setting import *
from CYK.plot_fit import plot_fit
import os
import numpy as np
import random as rn
import tensorflow as tf


np.random.seed(2018)

from keras import backend as K


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
from keras.layers import Dense, Input, GlobalMaxPooling1D,Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from CYK.plot_fit import visialize_model,save_history,plot_all_history
from keras import metrics
from CYK.data_loader import load_imdb, load_test


MAX_SEQUENCE_LENGTH = 100


print('Indexing word vectors.')
embeddings_index = {}

import sys
#unit=int(sys.argv[1])
#unit=32
#unit=64
unit=128


embed_type='CBOW'
#f = open(SkipGram_embedding, encoding='utf-8')
f = open(CBOW_embedding, encoding='utf-8')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


print('Processing text dataset')


(X_train, y_train), (X_val, y_val) = load_imdb()
(X_test, y_test)=load_test()
#tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer= Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(X_train)
#X_train = tokenizer.texts_to_matrix(train_data['text'], mode='count')
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

vocab_size=len(word_index)+1

#text_to_word_sequence
#X_val = tokenizer.texts_to_matrix(val_data['text'], mode='count')
X_val = tokenizer.texts_to_sequences(X_val)



X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_val = pad_sequences(X_val, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

#y_train = np.array(y_train)
#y_val = np.array(y_val)
#y_test = np.array(y_test)


print('Preparing embedding matrix.')
        
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        

model = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH
                            )

model.add(embedding_layer)
print ('###########################################################')
print ('embedding layer output shape is:',model.output_shape)

#model.add(Conv1D(100,
#                 3,
#                 padding='valid',
#                 activation='relu',
#                 strides=1))
#model.add(GlobalMaxPooling1D())
#model.add(MaxPooling1D(pool_size=4))
#print ('after maxpooli ng layer the shape is:',model.output_shape)




model.add(LSTM(unit))
#model.add(Dense(250,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

################################
#model.add(Flatten())
#print ('Flatten layer output shape is:',model.output_shape)
#model.add(Dense(250,activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(250,activation='relu'))
#model.add(Dropout(0.2))

#
#model.add(LSTM(100))
##model.add(Bidirectional(LSTM(64)))
#model.add(Dense(1,activation='sigmoid'))



#tensorBoardCallback = TensorBoard(log_dir='./pqw_logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = os.path.join(best_model_dir, 'CBOW_LSTM_optimal_test.hdf5')
save_best_point = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


#, callbacks=[earlystopping]
history=model.fit(X_train,y_train , validation_data=(X_val,y_val), epochs=15, batch_size=32,
                  callbacks=[save_best_point])
#model.save_weights("own_vecmodel_model.h5")
#plot_model(model, to_file='model.png')
# Evaluation on the val set
#scores = model.evaluate(X_val, y_val, verbose=0)
#print ('=====================the result for val set==============================')
#print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

print (history.history.keys())


write_filename='{}_LSTM_UNIT{}.pdf'.format(embed_type,unit)
save_history(history, '{}_LSTM_UNIT{}.csv'.format(embed_type, unit), subdir='Word_LSTM')
#visialize_model(model, write_filename)
#plot_fit(history, plot_filename=write_filename)

print ('the process for {} is done'.format(write_filename))



from keras.models import load_model
#filepath = os.path.join(best_model_dir, 'bigram_fasttext_testset')
model = load_model(filepath)
scores = model.evaluate(X_test,y_test)
score={}
score[model.metrics_names[0]] = scores[0]
score[model.metrics_names[1]] = scores[1]
with open('LSTM_optimal.txt', 'w') as f:
    f.write(str(score))
print(score)











