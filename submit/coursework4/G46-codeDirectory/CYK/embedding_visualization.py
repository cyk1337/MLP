#!/usr/bin/env python

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
embedding_num = 2
embedding_path = os.path.join(embedding_dir, embeddings_path[embedding_num])
embeddings_index = load_pretrained_model(embedding_path)
# print(embeddings_index)


tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
# test_list = ['king', 'queen', 'man', 'woman']
# test_dict = {k:v for k,v in embeddings_index.items() if k in test_list}
test_list = list(embeddings_index.keys())[50:550]
test_dict = {k:v for k,v in embeddings_index.items() if k in test_list}
df = pd.DataFrame(test_dict).T
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

plt.figure(figsize=(18, 18))
for i, label in enumerate(labels):
    x, y = low_dim_embed[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

savepath = os.path.join(embedding_dir, '{}_.pdf'.format(embeddings[embedding_num]))
plt.savefig(savepath)
plt.show()
