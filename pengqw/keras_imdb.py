# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:16:08 2018

@author: pqw1995@163.com
"""
import sys
sys.path.append('D:\\MLP_Project\\MLP')
from config.setting import *

from CYK.plot_fit import visialize_model
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout,MaxPooling1D,GlobalAveragePooling1D,Conv1D,GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model

# Using keras to load the dataset with the top_words



top_words = 50000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Pad the sequence to the same length
max_review_length = 400
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Using embedding from Keras
embedding_vecor_length = 100
model = Sequential()
""" it directly concatenates word embeddings to get the sentence embedding.
    Could try aggregate word embedding values to gain sentence embedding.
"""
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length, trainable=False))  

# Convolutional model (3x conv, flatten, 2x dense)
#model.add(Convolution1D(64, 3, padding='same'))
#model.add(Convolution1D(32, 3, padding='same'))
#model.add(Convolution1D(16, 3, padding='same'))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(180,activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(1,activation='sigmoid'))

#model.add(Flatten())
#model.add(Dense(250,activation='sigmoid'))
#model.add(Dense(1,activation='sigmoid'))

model.add(Dropout(0.2))
model.add(Conv1D(250,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(250,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

#model.add(Convolution1D(64, 3, activation='relu',input_shape=(None,300)))
#model.add(Convolution1D(64, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Convolution1D(128, 3, activation='relu'))
#model.add(Convolution1D(128, 3, activation='relu'))
#model.add(GlobalAveragePooling1D())
#model.add(Dropout(0.2))
#model.add(Dense(1,activation='sigmoid'))

# model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
# model.add(Dense(1,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=15, callbacks=[tensorBoardCallback], batch_size=32)
#model.save_weights("CYK_CNN_model.h5")
#plot_model(model, to_file='model.png')
# Evaluation on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Loss: %.2f,  Accuracy: %.2f%%" % (scores[0],scores[1]*100))





