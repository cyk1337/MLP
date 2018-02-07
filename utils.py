# coding: utf-8
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models import Word2Vec
# sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
# model = Word2Vec(sentences, min_count=1)
# say_vector = model['say']  # get vector for word


import pandas as pd


data_path = 'MLP/dataset/yelp/yelp.csv'
for ch in pd.read_csv(data_path, iterator=True, chunksize=1000):
    print(ch['text'].to_string)