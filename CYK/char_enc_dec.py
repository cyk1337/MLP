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

@file: char_lstm.py

@time: 21/03/2018 21:02

@desc:         
               
'''
import __init__
from config.setting import *

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import datetime, json, sys
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Convolution1D, \
    Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Activation, \
    Flatten, Reshape, LSTM
from keras.callbacks import ModelCheckpoint
from CYK.data_loader import load_imdb, create_vocab_set, encode_data, enc_dec_batch_generator,mini_batch_generator, load_test
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history



print('Loading data...')
#Expect x to be a list of sentences. Y to be a one-hot encoding of the chars

# Using keras to load the dataset with the top_words
(X_train, y_train), (X_val, y_val) = load_imdb()
(X_test, y_test) = load_test()

print('Creating vocab ...')
vocab, reverse_vocab, vocab_size, vocab_check = create_vocab_set()

#Maximum length. Longer gets chopped. Shorter gets padded.
# maxlen = 1014
# train_data = encode_data(X_train, maxlen, vocab, vocab_size, vocab_check)
# val_data = encode_data(X_val, maxlen, vocab, vocab_size, vocab_check)
# test_data = encode_data(X_test, maxlen, vocab, vocab_size, vocab_check)


print('Build model...')
#Model params
#Filters for conv layers


# units = 512
latent_dim = 32
#units = int(sys.argv[1])
dropout_rate = 0
#dropout_rate = float(sys.argv[2])
maxlen = 1014

#layer_num = 2

print("Building model...")
print('LSTM units:',latent_dim)
#inputs = Input(shape=( maxlen+1,vocab_size,))


# Define an input sequence and process it.
encoder_inputs =  Input(shape=( maxlen+1,vocab_size,))
encoder = LSTM(latent_dim, return_state=True,name='encoder')
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs =Input(shape=( maxlen+1,vocab_size,))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
#decoder_outputs =Reshape((-1,500))(decoder_outputs)
decoder_dense = Dense(69, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
enc_dec_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(enc_dec_model.summary())


enc_dec_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



training_batch_size = 500
training_steps = 70

#val_batch_size = 500
#val_steps = 15

# 35000
train_data_generator = enc_dec_batch_generator(X_train,y_train, vocab, vocab_size, vocab_check, maxlen, batch_size=training_batch_size)
# 7500
#val_data_generator = enc_dec_batch_generator(X_train,y_train, vocab, vocab_size, vocab_check, maxlen, batch_size=val_batch_size)

#filepath = os.path.join(best_model_dir, 'enc_dec_units{}_dropout_{}.hdf5'.format(latent_dim, dropout_rate))
#save_best_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#callbacks=[save_best_point]
#callbacks=None

history = enc_dec_model.fit_generator(train_data_generator, steps_per_epoch=training_steps, epochs=1, verbose=1)
#                              validation_data=val_data_generator, validation_steps=val_steps,
#                               callbacks=callbacks)


encoder_weights = enc_dec_model.get_layer('encoder').get_weights()
w_name='encoder_weights'
np.save(w_name,encoder_weights)

#======================================
print("Building model...")
print('LSTM units:',latent_dim)
inputs = Input(shape=( maxlen,vocab_size,))

encoder_weights = np.load(w_name)
# Define an input sequence and process it.
encoder_inputs =  Input(shape=( maxlen,vocab_size,))
encoder = LSTM(latent_dim, weights=encoder_weights, name='encoder')
encoder_outputs = encoder(encoder_inputs)

predictions= Dense(1, activation='sigmoid') (encoder_outputs)

model = Model(inputs=inputs, outputs=predictions)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
training_batch_size = 500
training_steps = 70

val_batch_size = 500
val_steps = 15

# 35000
train_data_generator = mini_batch_generator(X_train,y_train, vocab, vocab_size, vocab_check, maxlen, batch_size=training_batch_size)
# 7500
val_data_generator = mini_batch_generator(X_train,y_train, vocab, vocab_size, vocab_check, maxlen, batch_size=val_batch_size)

filepath = os.path.join(best_model_dir, 'SA_lstm_char_units{}_layer_num{}_dropout_{}.hdf5'.format(units, layer_num,  dropout_rate))
save_best_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


history = model.fit_generator(train_data_generator, steps_per_epoch=training_steps, epochs=15, verbose=1,
                              validation_data=val_data_generator, validation_steps=val_steps,
                               callbacks=[save_best_point])




#"""
subdir = 'lstm_char'
plot_filename = 'SA_lstm_char_units{}_layer_num{}_dropout_{}.pdf'.format(latent_dim, layer_num,  dropout_rate)
# save history info
save_history(history, '{}.csv'.format(plot_filename[:-4]), subdir=subdir)
# save model
# visialize_model(model, filepath=plot_filename)
# save single history
#plot_fit(history, plot_filename=plot_filename)

test_data_generator = mini_batch_generator(X_test,y_test, vocab, vocab_size, vocab_check, maxlen, batch_size=val_batch_size)

from keras.models import load_model
model = load_model(filepath)

scores = model.evaluate_generator(test_data_generator, steps=15)
score={}
score[model.metrics_names[0]] = scores[0]
score[model.metrics_names[1]] = scores[1]
with open('char_result_optimal.txt', 'a') as f:
    f.write('SA_lstm_char_units{}_layer_num{}_dropout_{}:\n'.format(latent_dim, layer_num,  dropout_rate))
    f.write(str(score))
    f.write('\n')
    
print(score)

#"""
