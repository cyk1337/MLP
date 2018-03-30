#!/usr/bin/env python

# encoding: utf-8

import numpy as np

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
