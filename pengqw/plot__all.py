# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:03:38 2018

@author: pqw1995@163.com
"""
import sys
sys.path.append('D:\\MLP_Project\\MLP')
from config.setting import *
from CYK.plot_fit import *

subdir = 'LSTM_CBOW_UNIT'
plot_all_history(subdir, plot_filename='LSTM_CBOW_X_UNIT_100length.pdf',figsize=(16,8))


