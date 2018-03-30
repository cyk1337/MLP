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
csv_logger = CSVLogger('log.csv', append=True, separator=';')

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

#tokenizer= Tokenizer(num_words=MAX_NUM_WORDS,
#                     char_level = True)
#
#tokenizer.fit_on_texts(train_data['text'])
#X_train = tokenizer.texts_to_sequences(train_data['text'])
#
#word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
#
#vocab_size=len(word_index)+1
#
#X_test = tokenizer.texts_to_sequences(test_data['text'])

#X_train=np.array(train_data['text'])
#X_test=np.array(test_data['text'])
#
#
#y_train=train_data['target']
#y_test=test_data['target']

#X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
#X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

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
        

def encode_data(x, maxlen, vocab, vocab_size, check):
    #Iterate over the loaded data and create a matrix of size maxlen x vocabsize
    #In this case that will be 1014x69. This is then placed in a 3D matrix of size
    #data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
    #array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower().replace(' ', ''))
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data


def shuffle_matrix(x, y):
    stacked = np.hstack((np.matrix(x).T, y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi


def create_vocab_set():
    #This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check


vocab, reverse_vocab, vocab_size, check=create_vocab_set()
#X_train=encode_data(X_train, MAX_SEQUENCE_LENGTH, vocab, vocab_size, check)
#X_val=encode_data(X_val, MAX_SEQUENCE_LENGTH, vocab, vocab_size, check)

def data_generator(X, y):
    while 1:
        for i in range(35):
             yield encode_data(X[i*1000:(i+1)*1000], MAX_SEQUENCE_LENGTH, vocab, vocab_size, check), y[i*1000:(i+1)*1000]

def val_generator(X_val, y_val):
    while 1:
        for i in range(15):
            yield encode_data(X_val[i*500:(i+1)*500],MAX_SEQUENCE_LENGTH, vocab, vocab_size, check), y_val[i*500:(i+1)*500]  


def test_generator(X_test, y_test):
    while 1:
        for i in range(15):
            yield encode_data(X_test[i*500:(i+1)*500],MAX_SEQUENCE_LENGTH, vocab, vocab_size, check), y_test[i*500:(i+1)*500]  

#encode_data(X_val[i*1000:(i+1)*1000], MAX_SEQUENCE_LENGTH, vocab, vocab_size, check)

num_words = min(MAX_NUM_WORDS, vocab_size)

model = Sequential()
#embedding_layer = Embedding(num_words,
#                            num_words,
##                            weights=[embedding_matrix],
#                            input_length=MAX_SEQUENCE_LENGTH,
#                            trainable=False
#                            #dropout=0.2
#                            )
#
#model.add(embedding_layer)
#print ('###########################################################')
#print ('embedding layer output shape is:',model.output_shape)

model.add(Conv1D(256,
                 7,
                 padding='valid',
                 activation='relu',
                 strides=1,input_shape=(MAX_SEQUENCE_LENGTH,num_words)))
model.add(MaxPooling1D(pool_size=3))
#model.add(Conv1D(256,
#                 7,
#                 padding='valid',
#                 activation='relu',
#                 strides=1))
#model.add(MaxPooling1D(pool_size=3))

#model.add(Conv1D(256,
#                 3,
#                 padding='valid',
#                 activation='relu',
#                 strides=1))
#model.add(Conv1D(256,
#                 3,
#                 padding='valid',
#                 activation='relu',
#                 strides=1))
#model.add(Conv1D(256,
#                 3,
#                 padding='valid',
#                 activation='relu',
#                 strides=1))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=3))


print ('after maxpooling layer the shape is:',model.output_shape)
#model.add(GlobalMaxPooling1D())
print ('after maxpooling layer the shape is:',model.output_shape)


model.add(Flatten())

#model.add(Dense(1024,activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
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
filepath='../../MLP_models/char_CNN_2_CNNlayer_2DNN_newtest.hdf5'

#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_acc', mode = 'max')


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#history=model.fit(X_train,y_train , validation_data=(X_val,y_val), epochs=15, batch_size=64)


history=model.fit_generator(data_generator(X_train, y_train), 
                            steps_per_epoch=35,epochs=15,verbose=1,
                            validation_data=val_generator(X_val,y_val),
                            validation_steps=15, callbacks=[checkpoint])

# Evaluation on the test set
#scores = model.evaluate(encode_data(X_test, MAX_SEQUENCE_LENGTH, vocab, vocab_size, check ), y_test, verbose=0)


#print ('=====================the result for test set==============================')
#print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

print (history.history.keys())

write_filename='char_CNN_2_CNNlayer_2DNN_newtest.pdf'
save_history(history, 'char_CNN_2_CNNlayer_2DNN_newtest.csv', subdir='Character_Level_Models')

print ('the process for {} is done'.format(write_filename))

new_model = load_model('../../MLP_models/char_CNN_2_CNNlayer_2DNN_newtest.hdf5')
#new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#scores = new_model.evaluate(X_test,y_test, verbose=0)
scores = new_model.evaluate(encode_data(X_test, MAX_SEQUENCE_LENGTH, vocab, vocab_size, check ), y_test, verbose=0)

print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

with open('new_new_test_result.txt', 'a') as f:
    f.write('\n the model name is {0}, the  best loss on test is: {1}, the acc on test is: {2} \n'.format(write_filename, 
            scores[0],scores[1]))


