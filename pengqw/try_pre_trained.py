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
from CYK.data_loader import load_imdb,load_test
from keras.models import load_model


MAX_SEQUENCE_LENGTH = 1000
#earlystopping = EarlyStopping(patience=4)
csv_logger = CSVLogger('log.csv', append=True, separator=';')

print('Indexing word vectors.')
embeddings_index = {}
#f = open('D:\MLP_Project\glove.6B.100d.txt','r',encoding="utf-8")
#f = open(CBOW_embedding, encoding='utf-8')
#f = open(SkipGram_embedding, encoding='utf-8')

f = open('../../cbow_gram.txt', encoding='utf-8')
#f = open('../../cbow_gram.txt', encoding='utf-8')





####hello
#f = open('D:\MLP_Project\MLP\\embedding\gensim_word2vec.txt','r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


print('Processing text dataset')

#train_data=pd.read_csv(train_csv)
#val_data=pd.read_csv(val_csv)

(X_train, y_train), (X_val, y_val) = load_imdb()
X_test, y_test = load_test()


#tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer= Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(X_train)
#X_train = tokenizer.texts_to_matrix(train_data['text'], mode='count')
X_train = tokenizer.texts_to_sequences(X_train)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

vocab_size=len(word_index)+1

#text_to_word_sequence
#X_val = tokenizer.texts_to_matrix(val_data['text'], mode='count')
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
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False
                            )

model.add(embedding_layer)
print ('###########################################################')
print ('embedding layer output shape is:',model.output_shape)

model.add(Conv1D(100,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
#model.add(MaxPooling1D(pool_size=4))
print ('after maxpooling layer the shape is:',model.output_shape)
#model.add(LSTM(80))
model.add(Dense(250,activation='relu'))
model.add(Dropout(0.5))
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

filepath='keras_models/NEW_CBOW_CNN_WORD.hdf5'
checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode = 'min')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#, callbacks=[earlystopping]
history=model.fit(X_train,y_train , validation_data=(X_val,y_val), epochs=15, batch_size=32,callbacks=[checkpoint])
#model.save_weights("own_vecmodel_model.h5")
#plot_model(model, to_file='model.png')
# Evaluation on the val set
scores = model.evaluate(X_val, y_val, verbose=0)
print ('=====================the result for val set==============================')
print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

print (history.history.keys())


write_filename='NEW_CBOW_CNN_WORD.pdf'
save_history(history, 'NEW_CBOW_CNN_WORD.csv', subdir='EM_TEST')
visialize_model(model, write_filename)
plot_fit(history, plot_filename=write_filename)

print ('the process for {} is done'.format(write_filename))




new_model = load_model('keras_models/NEW_CBOW_CNN_WORD.hdf5')
#new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
scores = new_model.evaluate(X_test,y_test, verbose=0)
print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

with open('test_result.txt', 'a') as f:
    f.write('\n the model name is {0}, the  best loss on test is: {1}, the acc on test is: {2} \n'.format(write_filename, 
            scores[0],scores[1]))










