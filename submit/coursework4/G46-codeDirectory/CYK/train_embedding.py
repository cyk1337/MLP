# -*- coding: utf-8 -*-

import sys

import __init__
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
from string import punctuation
from keras.preprocessing.text import text_to_word_sequence
from CYK.data_loader import load_imdb



#txt_file=os.path.join(train_pos,'*.txt')
#txt_neg=os.path.join(train_neg, '*.txt') 

############################write data to local

#txt_file='D:\MLP_Project\MLP\dataset\\aclImdb\\train\pos\*.txt'
#txt_neg='D:\MLP_Project\MLP\dataset\\aclImdb\\train\\neg\*.txt'
#
#filename_pos= glob.glob(txt_file)
#filename_neg= glob.glob(txt_neg)
#train_data_pos=pd.DataFrame()
#train_data_neg=pd.DataFrame()
#train_data=pd.DataFrame()
#
#for file in filename_pos:
#    with open (file, 'rb') as f:
#        data = f.read().decode('utf-8')
#        data=re.sub(r'<br />|<br />','',data)
#        score = re.findall(r'_(\d+).',file)
#        #train_data_pos=train_data_pos.append({'score': score[0], 'text': data},ignore_index=True)
#        train_data_pos=train_data_pos.append({'score': score[0], 'text': data},ignore_index=True)
#        
#for file in filename_neg:
#    with open (file, 'rb') as f:
#        data = f.read().decode('utf-8')
#        data=re.sub(r'<br />|<br />','',data)
#        score = re.findall(r'_(\d+).',file)
#        train_data_neg=train_data_neg.append({'score': score[0], 'text': data},ignore_index=True)
#    
#train_data=pd.concat([train_data_pos,train_data_neg],ignore_index=True, axis=0)
#train_data=train_data.sample(frac=1)
#
#train_data.to_csv('train_data.csv',index=False, encoding='utf-8')

#\d+|[\u4e00-\u9fff]+

#
#train_data=pd.read_csv(train_csv)
#
#train_data['text'] = train_data.apply(lambda row: text_to_word_sequence(row['text']), axis=1)

(X_train, y_train), (X_val, y_val) = load_imdb()
#
X_train = X_train.apply(text_to_word_sequence)


#for inde, x in enumerate(train_data['text']):
##    mm= [w.lower() for w in x]
#    mm= [word for word in x if word.isalpha()]
#    train_data['text'][inde]=mm

print ('the preprocessing is done')
skip_vec_model = Word2Vec(X_train,size=100, window=5, min_count=5, workers=multiprocessing.cpu_count()*2, sg=1, iter=40,compute_loss=True)

print ('vector model training process is done') 
print ('vocabulary size is :', len(skip_vec_model.wv.index2word))
print ('the latese loss is :', skip_vec_model.get_latest_training_loss())
#vec_model.save('vec_model_sg')
#skip_path = os.path.join(embedding_dir, 'new/100d_skipgram.txt')
#cbow_path = os.path.join(embedding_dir, '100d_cbow.txt')

skip_vec_model.wv.save_word2vec_format('../../skip_gram.txt',binary=False)

cbow_vec_model = Word2Vec(X_train,size=100, window=5, min_count=5, workers=multiprocessing.cpu_count()*2, sg=0, iter=40,compute_loss=True)
cbow_vec_model.wv.save_word2vec_format('../../cbow_gram.txt',binary=False)




    
