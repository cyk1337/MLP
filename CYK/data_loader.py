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
    # test data
    test_data = pd.read_csv(test_csv)
    X_test = test_data['text']
    y_test = test_data['target']
    return (X_train, y_train), (X_test, y_test)