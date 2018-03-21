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

@file: char_cnn.py

@time: 21/03/2018 21:02

@desc:         
               
'''
import __init__
from config.setting import *

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import datetime, json
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Convolution1D, Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Activation, Flatten, Reshape

from CYK.data_loader import load_imdb, create_vocab_set, shuffle_matrix, encode_data, mini_batch_generator
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history



def cnn_model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter,
          cat_output):
    #Define what the input shape looks like
    inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    #All the convolutional layers...
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(inputs)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu')(conv2)

    conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                          border_mode='valid', activation='relu')(conv3)

    conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                          border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)

    #Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    #Output dense layer with softmax activation
    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(input=inputs, output=pred)

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    return model



#Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 1014

#Model params
#Filters for conv layers
nb_filter = 256
#Number of units in the dense layer
dense_outputs = 1024
#Conv layer kernel size
filter_kernels = [7, 7, 3, 3, 3, 3]
#Number of units in the final output layer. Number of classes.
cat_output = 4

#Compile/fit params
batch_size = 80
nb_epoch = 10

subset = None

print('Loading data...')
#Expect x to be a list of sentences. Y to be a one-hot encoding of the chars

# Using keras to load the dataset with the top_words
(X_train, y_train), (X_val, y_val) = load_imdb()

print('Creating vocab ...')
vocab, reverse_vocab, vocab_size, check = create_vocab_set()

val_data = encode_data(X_val, maxlen, vocab, vocab_size, check)

print('Build model...')

model = cnn_model(filter_kernels, dense_outputs, maxlen, vocab_size,
                       nb_filter, cat_output)

print('Fit model...')
initial = datetime.datetime.now()
for e in range(nb_epoch):
    xi, yi = shuffle_matrix(X_train, y_train)
    xi_test, yi_test = shuffle_matrix(X_val, y_val)
    if subset:
        batches = mini_batch_generator(xi[:subset], yi[:subset],
                                                    vocab, vocab_size, check,
                                                    maxlen,
                                                    batch_size=batch_size)
    else:
        batches = mini_batch_generator(xi, yi, vocab, vocab_size,
                                                    check, maxlen,
                                                    batch_size=batch_size)

    test_batches = mini_batch_generator(xi_test, yi_test, vocab,
                                                     vocab_size, check, maxlen,
                                                     batch_size=batch_size)

    accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    print('Epoch: {}'.format(e))
    for x_train, y_train in batches:
        f = model.train_on_batch(x_train, y_train)
        loss += f[0]
        loss_avg = loss / step
        accuracy += f[1]
        accuracy_avg = accuracy / step
        if step % 100 == 0:
            print('  Step: {}'.format(step))
            print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
        step += 1

    test_accuracy = 0.0
    test_loss = 0.0
    test_step = 1

    for x_test_batch, y_test_batch in test_batches:
        f_ev = model.test_on_batch(x_test_batch, y_test_batch)
        test_loss += f_ev[0]
        test_loss_avg = test_loss / test_step
        test_accuracy += f_ev[1]
        test_accuracy_avg = test_accuracy / test_step
        test_step += 1
    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    print(
        'Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, test_accuracy_avg,
                                                                                    e_elap, t_elap))

#Whether to save model parameters
save = False
model_name_path = 'params/cnn_char.json'
model_weights_path = 'params/cnn_char_model_weights.h5'

if save:
    print('Saving model params...')
    json_string = model.to_json()
    with open(model_name_path, 'w') as f:
        json.dump(json_string, f)

    model.save_weights(model_weights_path)

