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
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, EarlyStopping,ModelCheckpoint
from CYK.plot_fit import visialize_model,save_history,plot_all_history
from keras import metrics
from keras.optimizers import SGD
from keras.layers.merge import concatenate
import string
from CYK.data_loader import load_imdb, load_test
from keras.models import load_model

MAX_SEQUENCE_LENGTH = 1014
EMBEDDING_DIM=100
MAX_NUM_WORDS=5000


print('Indexing word vectors.')
embeddings_index = {}
#f = open('D:\MLP_Project\glove.6B.100d.txt','r',encoding="utf-8")
f = open(CBOW_embedding, encoding='utf-8')
#f = open(SkipGram_embedding, encoding='utf-8')
#f = open('D:\MLP_Project\MLP\\embedding\gensim_word2vec.txt','r',encoding='utf-8')


print('Processing text dataset')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

(X_train, y_train), (X_val, y_val) = load_imdb()
X_test, y_test = load_test()


X_train_word = X_train.copy()
X_val_word = X_val.copy()
X_test_word = X_test.copy()


y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)




      

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

def data_generator(X, X_word, y):
    while 1:
        for i in range(70):
             yield [encode_data(X[i*500:(i+1)*500], MAX_SEQUENCE_LENGTH, vocab, vocab_size, check), X_word[i*500:(i+1)*500]],y[i*500:(i+1)*500]

def val_generator(X_val, X_word, y_val):
    while 1:
        for i in range(15):
            yield [encode_data(X_val[i*500:(i+1)*500],MAX_SEQUENCE_LENGTH, vocab, vocab_size, check),X_word[i*500:(i+1)*500]], y_val[i*500:(i+1)*500]  


def test_generator(X_test, X_word, y_test):
    while 1:
        for i in range(15):
            yield [encode_data(X_test[i*500:(i+1)*500],MAX_SEQUENCE_LENGTH, vocab, vocab_size, check),X_word[i*500:(i+1)*500]], y_test[i*500:(i+1)*500]  





#####################################word####################
MAX_SEQUENCE_LENGTH_Word=1000
#tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer= Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(X_train_word)
#X_train = tokenizer.texts_to_matrix(train_data['text'], mode='count')
X_train_word = tokenizer.texts_to_sequences(X_train_word)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

        
#text_to_word_sequence
#X_val = tokenizer.texts_to_matrix(val_data['text'], mode='count')
X_val_word = tokenizer.texts_to_sequences(X_val_word)
X_test_word = tokenizer.texts_to_sequences(X_test_word)




X_train_word = pad_sequences(X_train_word, maxlen=MAX_SEQUENCE_LENGTH_Word)
X_val_word = pad_sequences(X_val_word, maxlen=MAX_SEQUENCE_LENGTH_Word)
X_test_word = pad_sequences(X_test_word, maxlen=MAX_SEQUENCE_LENGTH_Word)


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




maxlen = 1014


print("Building model...")

# word
#num_words = min(MAX_NUM_WORDS, len(word_index))
#embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
#for word, i in word_index.items():
#    if i >= MAX_NUM_WORDS:
#        continue
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix[i] = embedding_vector
        
word_input=Input(shape=(MAX_SEQUENCE_LENGTH_Word,), name='word_input')
embedding_layer = Embedding(output_dim=EMBEDDING_DIM, 
                            input_dim=num_words, 
                            input_length=MAX_SEQUENCE_LENGTH_Word, name='WORD_embedding')
embed = embedding_layer(word_input)
print(embedding_layer.output_shape)
word_out = LSTM(128, name='word_lstm')(embed)
# char
char_inputs = Input(shape=(maxlen,vocab_size,),  name='char_INPUT')
char_out = LSTM(64, name='char_lstm')(char_inputs)

concatenated = concatenate([char_out, word_out])
predictions = Dense(1, activation='sigmoid', name='output_layer')(concatenated)

merged_model = Model([char_inputs, word_input], predictions)

print(merged_model.summary())







#filepath='keras_models/char_CNN_1_CNNlayer_2DNN_lessunits_{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5'
filepath='best_/combined_lstm.hdf5'
checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_acc', mode = 'max')


merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#history=model.fit(X_train,y_train , validation_data=(X_val,y_val), epochs=15, batch_size=64)


history=merged_model.fit_generator(data_generator(X_train,X_train_word, y_train),
                            steps_per_epoch=70,epochs=15,verbose=1,
                            validation_data=val_generator(X_val,X_val_word,y_val),
                            validation_steps=15, callbacks=[checkpoint])

# Evaluation on the test set
#scores = model.evaluate(encode_data(X_test, MAX_SEQUENCE_LENGTH, vocab, vocab_size, check ), y_test, verbose=0)


#print ('=====================the result for test set==============================')
#print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

print (history.history.keys())

write_filename='combined_lstm.pdf'
save_history(history, 'combined_lstm.csv', subdir='combined_lstm')

print ('the process for {} is done'.format(write_filename))

#new_model = load_model('keras_models/char_word_CNN.hdf5')
merged_model.load_weights(filepath)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#scores = new_model.evaluate(X_test,y_test, verbose=0)
scores = merged_model.evaluate([encode_data(X_test, MAX_SEQUENCE_LENGTH, vocab, vocab_size, check), X_test_word], y_test, verbose=0)

print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

score={}
score[merged_model.metrics_names[0]] = scores[0]
score[merged_model.metrics_names[1]] = scores[1]
with open('char_result_optimal.txt', 'a') as f:
    f.write('combedin lstm:\n')
    f.write(str(score))
    f.write('\n')
    
print(score)


