# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:03:06 2018

@author: pqw1995@163.com
"""

import sys
sys.path.append('D:\\MLP_Project\\MLP')
import os
import numpy as np
import random as rn
import tensorflow as tf

#os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(1337)
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
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from CYK.plot_fit import plot_fit


MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100
MAX_NUM_WORDS = 20000

csv_logger = CSVLogger('log.csv', append=True, separator=';')

print('Indexing word vectors.')
embeddings_index = {}
f = open('D:\MLP_Project\glove.6B.100d.txt','r',encoding="utf-8")
#f = open('keras_prepro_vec_model_sg0.txt','r',encoding="utf-8")
#f = open('D:\MLP_Project\MLP\\embedding\gensim_word2vec.txt','r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


print('Processing text dataset')

train_data=pd.read_csv('train_data.csv')
test_data=pd.read_csv('test_data.csv')

#train_data['text'] = train_data['text'].str.replace('\d+', '')
#test_data['text'] = test_data['text'].str.replace('\d+', '')

#tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer= Tokenizer()

tokenizer.fit_on_texts(train_data['text'])

X_train = tokenizer.texts_to_sequences(train_data['text'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

vocab_size=len(word_index)+1

#text_to_word_sequence
X_test = tokenizer.texts_to_sequences(test_data['text'])


train_data['score'][train_data['score']<=4]=0
train_data['score'][train_data['score']>=7]=1

test_data['score'][test_data['score']<=4]=0
test_data['score'][test_data['score']>=7]=1

y_train=train_data['score']
y_test=test_data['score']

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
y_train = np.array(y_train)
y_test = np.array(y_test)

print('Preparing embedding matrix.')
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
#num_words = min(MAX_NUM_WORDS, len(word_index))
#embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
#for word, i in word_index.items():
#    if i >= MAX_NUM_WORDS:
#        continue
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix[i] = embedding_vector
        
#labels = to_categorical(np.asarray(labels))

model = Sequential()
embedding_layer = Embedding(vocab_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False
                            #dropout=0.2
                            )
model.add(embedding_layer)

#model.add(Convolution1D(64, 3, activation='relu',input_shape=(None,100)))
#model.add(Convolution1D(64, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Convolution1D(128, 3, activation='relu'))
#model.add(Convolution1D(128, 3, activation='relu'))
#model.add(GlobalAveragePooling1D())
#model.add(Dropout(0.2))
#model.add(Dense(1,activation='sigmoid'))

model.add(Flatten())
model.add(Dense(250,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))



#model.add(Convolution1D(64, 3, padding='same'))
#model.add(Convolution1D(32, 3, padding='same'))
#model.add(Convolution1D(16, 3, padding='same'))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(180,activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(1,activation='sigmoid'))


#model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
#model.add(Dense(1,activation='sigmoid'))

tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train,y_train , validation_data=(X_test,y_test), epochs=15,shuffle=False, callbacks=[tensorBoardCallback], batch_size=128)
model.save_weights("own_vecmodel_model.h5")
#plot_model(model, to_file='model.png')
# Evaluation on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print ('=====================the result for test set==============================')
print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))

print (history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()
plt.savefig('graphs/index_LSTM_accuracy.pdf')

plt.clf()
plt.cla()
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()
plt.savefig('graphs/index_LSTM_loss.pdf')













