import __init__
from config.setting import *

from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import os, re

from keras.preprocessing.text import Tokenizer

def clean_html(raw_html):
    # clean html tag
    pat_html = re.compile('<.*?>')
    remove_html = re.sub(pat_html, '', raw_html)
    return remove_html

def process_doc(raw_doc):
    # clean html tags
    doc = clean_html(raw_doc)
    #
    # tokenize, filter punctuation, lowercase
    tokenizer = Tokenizer(num_words=None, lower=True, char_level=False)
    tokenizer.fit_on_texts(doc)
    # remove
    return tokenizer


if __name__=="__main__":
    pass