#!/usr/bin/env python

# encoding: utf-8


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
from CYK.data_loader import load_imdb, create_vocab_set, encode_data, mini_batch_generator, load_test
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
units = 64
#units = int(sys.argv[1])
dropout_rate = 0
#dropout_rate = float(sys.argv[2])
maxlen = 1014

layer_num = 1

print("Building model...")
print('LSTM units:',units)
inputs = Input(shape=( maxlen,vocab_size,))
####### 1layer ###############
if layer_num == 1:
    l1 = LSTM(units,return_sequences=True)(inputs)
    dropout = Dropout(dropout_rate)(l1)

# ---------------- 2 layer 
elif layer_num == 2:
    l1 = LSTM(units,return_sequences=True)(inputs)
    l2 = LSTM(units,return_sequences=True)(l1)
    dropout = Dropout(dropout_rate)(l2)
# ================ 3 layer
elif layer_num == 3:
    l1 = LSTM(units,return_sequences=True)(inputs)
    l2 = LSTM(units,return_sequences=True)(l1)
    l3 = LSTM(units,return_sequences=True)(l2)
    dropout = Dropout(dropout_rate)(l3)

cnn1 = Conv1D(256,
                 7,
                 padding='valid',
                 activation='relu',
                 strides=1)(dropout)
maxpool1 = MaxPooling1D(pool_size=3)(cnn1)


cnn2 = Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1)(maxpool1)
maxpool2=MaxPooling1D(pool_size=3)(cnn2)

flatten = Flatten()(maxpool2)

predictions = Dense(1, activation='sigmoid')(flatten)

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

filepath = os.path.join(best_model_dir, 'lstm+cnn.hdf5')
save_best_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


history = model.fit_generator(train_data_generator, steps_per_epoch=training_steps, epochs=15, verbose=1,
                              validation_data=val_data_generator, validation_steps=val_steps,
                               callbacks=[save_best_point])



subdir = 'lstm+cnn_char'
plot_filename = 'lstm+cnn.pdf'
# save history info
save_history(history, '{}.csv'.format(plot_filename[:-4]), subdir=subdir)
# save model
# visialize_model(model, filepath=plot_filename)
# save single history
#plot_fit(history, plot_filename=plot_filename)

test_data_generator = mini_batch_generator(X_test,y_test, vocab, vocab_size, vocab_check, maxlen, batch_size=val_batch_size)

from keras.models import load_model
#filepath = os.path.join(best_model_dir, 'bigram_fasttext_testset')
model = load_model(filepath)

scores = model.evaluate_generator(test_data_generator, steps=15)
score={}
score[model.metrics_names[0]] = scores[0]
score[model.metrics_names[1]] = scores[1]
with open('char_result_optimal.txt', 'a') as f:
    f.write('lstm+cnn\n')
    f.write(str(score))
    f.write('\n')
    
print(score)


