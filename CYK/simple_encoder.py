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

@file: simple_encoder.py

@time: 11/02/2018 22:34

@desc:         
               
'''              
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential

from CYK.data_loader import load_imdb


(X_train, y_train), (X_test, y_test) = load_imdb()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
tokenizer.texts_to_matrix(texts=X_train, mode='counts')


print("Building model")
model = Sequential()

model.add()

