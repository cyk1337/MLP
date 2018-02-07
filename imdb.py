# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:49:20 2018

@author: pqw1995@163.com
"""

import json
import os
from keras.preprocessing.text import Tokenizer
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from keras.engine import Input
from keras.layers import Embedding, merge
from keras.models import Model
import numpy as np
from nltk.corpus import stopwords
import multiprocessing
import glob
import re

txt_file='dataset/aclImdb/train/pos/*.txt'
txt_neg='dataset/aclImdb/train/neg/*.txt'
filename_pos= glob.glob(txt_file)
filename_neg= glob.glob(txt_neg)
train_data_pos=pd.DataFrame()
train_data_neg=pd.DataFrame()
train_data=pd.DataFrame()

for file in filename_pos:
    with open (file, 'rb') as f:
        data = f.read().decode('utf-8')
        score = re.findall(r'_(\d+).',file)
        #train_data_pos=train_data_pos.append({'score': score[0], 'text': data},ignore_index=True)
        train_data_pos=train_data_pos.append({'score': score[0], 'text': data},ignore_index=True)
        
for file in filename_neg:
    with open (file, 'rb') as f:
        data = f.read().decode('utf-8')
        score = re.findall(r'_(\d+).',file)
        train_data_neg=train_data_neg.append({'score': score[0], 'text': data},ignore_index=True)
    
train_data=pd.concat([train_data_pos,train_data_neg],ignore_index=True, axis=0)
train_data=train_data.sample(frac=1)

for inde, x in enumerate(train_data['text']):
    mm= [w.lower() for w in x]
    train_data['text'][inde]=mm

print ('the preprocessing is done')
    
    



    