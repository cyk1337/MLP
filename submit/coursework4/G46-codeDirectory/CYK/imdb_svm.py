#!/usr/bin/env python

# encoding: utf-8

             

import __init__
from config.setting import *

from keras.preprocessing.text import Tokenizer

from sklearn import svm

from CYK.data_loader import load_imdb
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history
import pandas as pd

enc_dict = {}

(X_train, y_train), (X_val, y_val) = load_imdb()
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# =======================
print('count matrix')
# count matrix
Xtrain_count = tokenizer.texts_to_matrix(texts=X_train, mode='count')
Xval_count = tokenizer.texts_to_matrix(texts=X_val, mode='count')
# print(Xtrain_count.shape)
clf_count = svm.SVC(verbose=True)
print('---Beginning fitting...')
clf_count.fit(Xtrain_count, y_train)
print('Beginning count evaluation...')
score_count = clf_count.score(Xval_count, y_val)

enc_dict['score_count'] = score_count
print('Validation accuracy:', score_count)


del Xtrain_count
del Xval_count
print('='*80)

# ==========================
print('one-hot matrix beginning...')
# one-hot matrix
Xtrain_1hot = tokenizer.texts_to_matrix(texts=X_train, mode='binary')
Xval_1hot = tokenizer.texts_to_matrix(texts=X_val, mode='binary')

clf_1hot = svm.SVC(verbose=True)
print('---Beginning 1 hot fitting...')
clf_1hot.fit(Xtrain_1hot, y_train)
print('Beginning 1 hot evaluation...')
score_1hot = clf_1hot.score(Xval_1hot, y_val)
enc_dict['score_1hot'] = score_1hot
print('1 hot validation accuracy:', score_1hot)

del Xtrain_1hot
del Xval_1hot
print('='*80)

#"""
# ==============================
print('freq matrix beginning...')
# freq matrix
Xtrain_freq = tokenizer.texts_to_matrix(texts=X_train, mode='freq')
Xval_freq = tokenizer.texts_to_matrix(texts=X_val, mode='freq')

clf_freq = svm.SVC(verbose=True)
print('---Beginning freq fitting...')
clf_freq.fit(Xtrain_freq, y_train)
print('Beginning freq evaluation...')
score_freq = clf_freq.score(Xval_freq, y_val)
enc_dict['score_freq'] = score_freq
print('1 hot validation accuracy:', score_freq)


del Xtrain_freq
del Xval_freq
print('='*80)
# =============================
print('tfidf matrix beginning...')
# tfidf matrix
Xtrain_tfidf = tokenizer.texts_to_matrix(texts=X_train, mode='tfidf')
Xval_tfidf = tokenizer.texts_to_matrix(texts=X_val, mode='tfidf')

clf_tfidf = svm.SVC(verbose=True)
print('---Beginning tfidf fitting...')
clf_tfidf.fit(Xtrain_tfidf, y_train)
print('Beginning tfidf evaluation...')
score_tfidf = clf_tfidf.score(Xval_tfidf, y_val)
enc_dict['clf_tfidf'] = score_tfidf
print('tfidf validation accuracy:', score_tfidf)

del Xtrain_tfidf
del Xval_tfidf
print('='*80)

#"""

with open('basic_matrix.txt', 'w') as f:
    f.write(str(enc_dict))
score_basic_matrix = pd.DataFrame(list(enc_dict.items()))
score_basic_matrix.to_csv('svm_matrix.csv', index=False, encoding='utf-8')
