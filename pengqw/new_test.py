# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 20:13:12 2018

@author: pqw1995@163.com
"""


train_data=pd.read_csv(train_csv)
test_data=pd.read_csv(test_csv)


chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# split the corpus into sequences of length=maxlen
#input is a sequence of 40 chars and target is also a sequence of 40 chars shifted by one position
#for eg: if you maxlen=3 and the text corpus is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [b,c,d], [b,c,d]--->[c,d,e]....and so on
maxlen = 40
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen+1, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i+1:i +1+ maxlen])
    #if i<10 :
       # print (text[i: i + maxlen])
        #print(text[i+1:i +1+ maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences),maxlen, len(chars)), dtype=np.bool) # y is also a sequence , or  a seq of 1 hot vectors
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

for i, sentence in enumerate(next_chars):
    for t, char in enumerate(sentence):
        y[i, t, char_indices[char]] = 1
    

print ('vetorization completed')
