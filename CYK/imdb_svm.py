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

@file: imdb_svm.py

@time: 16/02/2018 19:44

@desc:         
               
'''              

import __init__
from config.setting import *

from keras.preprocessing.text import Tokenizer

from sklearn import svm

from CYK.data_loader import load_imdb
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history


(X_train, y_train), (X_test, y_test) = load_imdb()
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)


# count matrix

Xtrain_count = tokenizer.texts_to_matrix(texts=X_train, mode='count')
Xtest_count = tokenizer.texts_to_matrix(texts=X_test, mode='count')
print(Xtrain_count.shape)
clf = svm.SVC()
print('---Beginning fitting...')
clf.fit(Xtrain_count, y_train, verbose=True)
print('Beginning evaluation...')
score = clf.score(Xtest_count, y_test)
print('Validation accuracy:', score)