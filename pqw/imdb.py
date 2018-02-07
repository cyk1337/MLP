# -*- coding: utf-8 -*-

from config.setting import *
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

txt_file=os.path.join(train_pos,'*.txt')
txt_neg=os.path.join(train_neg, '*.txt') 
filename_pos= glob.glob(txt_file)
filename_neg= glob.glob(txt_neg)
train_data_pos=pd.DataFrame()
train_data_neg=pd.DataFrame()
train_data=pd.DataFrame()

for file in filename_pos:
    with open (file, 'rb') as f:
        data = f.read().decode('utf-8')
        data=re.sub('<br />|<br />',' ',data)
        score = re.findall(r'_(\d+).',file)
        #train_data_pos=train_data_pos.append({'score': score[0], 'text': data},ignore_index=True)
        train_data_pos=train_data_pos.append({'score': score[0], 'text': data},ignore_index=True)
        
for file in filename_neg:
    with open (file, 'rb') as f:
        data = f.read().decode('utf-8')
        data=re.sub('<br />|<br />',' ',data)
        score = re.findall(r'_(\d+).',file)
        train_data_neg=train_data_neg.append({'score': score[0], 'text': data},ignore_index=True)
    
train_data=pd.concat([train_data_pos,train_data_neg],ignore_index=True, axis=0)
train_data=train_data.sample(frac=1)

train_data['text'] = train_data.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
for inde, x in enumerate(train_data['text']):
    mm= [w.lower() for w in x]
    train_data['text'][inde]=mm

print ('the preprocessing is done')
vec_model = Word2Vec(train_data['text'],size=200, window=5, min_count=5, workers=multiprocessing.cpu_count()*2, sg=2, iter=20,compute_loss=True)
print ('vector model training process is done')
print ('vocabulary size is :', len(vec_model.wv.index2word))
vec_model.save('vec_model')

    
