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

@file: data_loader.py

@time: 11/02/2018 22:28

@desc:         
               
'''              

import __init__
from config.setting import *

import pandas as pd

def load_imdb():
    ## train data
    train_data = pd.read_csv(train_csv)
    X_train = train_data['text']
    y_train = train_data['target']
    ## val data
    val_data = pd.read_csv(val_csv)
    X_val = val_data['text']
    y_val = val_data['target']
    return (X_train, y_train), (X_val, y_val)

def load_test():
    # test data
    test_data = pd.read_csv(test_csv)
    X_test = test_data['text']
    y_test = test_data['target']
    return (X_test, y_test)
