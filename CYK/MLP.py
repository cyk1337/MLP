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

@file: MLP.py.py

@time: 21/03/2018 22:27

@desc:         
               
'''              
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import datetime, json
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.layers import Input, Dense, Dropout, Convolution1D, \
    Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Activation,\
    Flatten, Reshape

class imdb_LSTM():
    def __init__(self, unit):
        self.unit = unit

    def __call__(self, *args, **kwargs):
        pass