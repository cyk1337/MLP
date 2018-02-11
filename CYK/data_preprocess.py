import __init__
from config.setting import *

from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import shutil
import pandas as pd
import numpy as np
import os, re

from keras.preprocessing.text import Tokenizer

# remove html tags
def clean_html(raw_html):
    # clean html tag
    pat_html = re.compile('<.*?>')
    remove_html = re.sub(pat_html, '', raw_html)
    return remove_html

# pre-process text except tokenizer operation
def process_doc(raw_doc):
    # clean html tags
    remove_doc = clean_html(raw_doc)
    # remove
    return remove_doc

def read_all_texts(filepath):
    pre_doc = list()
    labels_index = dict()
    for file in os.listdir(filepath):
        file_path = os.path.join(filepath, file)
        # parse label
        match = re.findall('_(\d+).txt',file)

        with open(file_path, encoding='utf-8') as f:
            text = process_doc(f.read())
            if len(match) > 0:
                label = match[0]
                label_id = len(pre_doc)
                labels_index[label_id]=label
        pre_doc.append(text)
    return pre_doc, labels_index

#
def get_doc_list(folder_list):
    records = list()
    for i, folder in enumerate(folder_list):
        texts, labels_index = read_all_texts(folder)
        # append records(lines)
        for j, text in enumerate(texts):
            records.append((text,labels_index[j]))
    assert len(records) == 25000, "Review size doesn't match!"
    return records

def get_file_num(folder):
    return len(os.listdir(folder))

def generate_csv(folders, csv_file, shuffle_seed=2018):
    record = get_doc_list(folders)

    # hand label then shuffle
    # =============================
    # num_neg = get_file_num(train_neg)
    # num_pos = get_file_num(train_pos)
    # # label
    # labels = np.concatenate((np.zeros(num_neg), np.ones(num_pos)))
    # =============================

    dataset = pd.DataFrame.from_records(record, columns=['text', 'star'])
    dataset['target'] = 0
    dataset.loc[dataset['star'].astype('int64') >= 7, 'target'] = 1
    shuffle_data = dataset.sample(frac=1, random_state=shuffle_seed)

    if not os.path.exists(csv_file):

        print('Generating train_csv...')
        shuffle_data.to_csv(csv_file, index=False, encoding='utf-8')
        print('Finished')
    else:
        print('%s already exists!' % csv_file)

def del_csv(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    return

if __name__=="__main__":

    train_folders = [train_neg, train_pos]
    test_folders = [test_neg, test_pos]

    # del file and regenerate
    del_csv(data_file)

    # generate train and test csv file if not exist
    if not os.path.exists(data_file):
        os.mkdir(data_file)
    if not os.path.exists(train_csv):
        generate_csv(train_folders, csv_file=train_csv)
    if not os.path.exists(test_csv):
        generate_csv(test_folders, csv_file=test_csv)
    print('Train and test data generated!\n','-'*80)




