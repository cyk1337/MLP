# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:03:38 2018

@author: pqw1995@163.com
"""
import sys
sys.path.append('D:\\MLP_Project\\MLP')
from config.setting import *
from CYK.plot_fit import *

subdir = 'Character_Level_Models/draw'
plot_all_history(subdir, plot_filename='CNN_character_level_different_layer_units.pdf',figsize=(26,13))


