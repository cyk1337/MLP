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

@file: embedding_visual_comparision.py

@time: 17/02/2018 20:22

@desc:         
               
'''
# !/usr/bin/env python

# encoding: utf-8

import __init__
from config.setting import *
from CYK.embedding_loader import load_pretrained_model

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

embeddings = ['cbow', 'skipgram', 'glove']
embeddings_path = [CBOW_embedding, SkipGram_embedding, Glove_embedding]
# embedding index
embedding_num = 2 # glove
embedding_path = os.path.join(embedding_dir, embeddings_path[embedding_num])
embeddings_index = load_pretrained_model(embedding_path)

CBOW_path = os.path.join(embedding_dir, embeddings_path[0])
embeddings_CBOW = load_pretrained_model(CBOW_path)
cbow_list = list(embeddings_CBOW.keys())[50:550]
# print(embeddings_index)


tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
word_dict = {k: v for k, v in embeddings_CBOW.items() if k in cbow_list}
df = pd.DataFrame(word_dict).T
low_dim_embed = tsne.fit_transform(df)
labels = df.index
"""
from adjustText import adjust_text
fig, ax = plt.subplots()
texts = []
for x, y, text in zip(low_dim_embed[:, 0], low_dim_embed[:, 1], labels):
    plt.scatter(x, y)
    texts.append(ax.text(x, y, text))
adjust_text(texts, force_text=0.05, arrowprops=dict(arrowstyle="->",
                                                    color='r', alpha=0.5))
"""

plt.figure(figsize=(38, 18))
plt.subplot(121)
for i, label in enumerate(labels):
    x, y = low_dim_embed[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 fontsize = 20,
                 ha='right',
                 va='bottom')
plt.title('CBOW embedding', fontsize=20)


tsne2 = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
word_dict = {k: v for k, v in embeddings_index.items() if k in cbow_list}
df2 = pd.DataFrame(word_dict).T
low_dim_embed2 = tsne2.fit_transform(df2)
labels2 = df2.index


plt.subplot(122)
for i, label in enumerate(labels2):
    x, y = low_dim_embed2[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 fontsize=20,
                 ha='right',
                 va='bottom')
plt.title('GloVe embedding', fontsize=20)


savepath = os.path.join(embedding_dir, 'CBOW_GloVe_comparison.pdf')
plt.savefig(savepath)
print('Saved to',savepath)
plt.show()
