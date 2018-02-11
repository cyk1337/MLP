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

@file: embedding_reader.py

@time: 11/02/2018 20:21

@desc:         
               
'''

# encoding method 1: load pre-trained embedding

# build mapping for pretrained models
# dict {word->vector}
# ====================
def load_pretrained_model(embedding_path):
    embedding_index = dict()
    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word.isdigit(): continue
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
    print("Found %s word vectors" % len(embedding_index))
    return embedding_index
