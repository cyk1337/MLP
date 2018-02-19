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
    assert len(records) == RECORDS_NUM, "Review size doesn't match!"
    return records

def get_file_num(folder):
    return len(os.listdir(folder))

def generate_csv(folders, csv_file, shuffle_seed=SEED):
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

        print('Generating {}...'.format(csv_file))
        shuffle_data.to_csv(csv_file, index=False, encoding='utf-8')
        print('Finished')
    else:
        print('%s already exists!' % csv_file)



def split_data(folders, split_bound=[0.70, 0.15, 0.15], shuffle_seed=SEED):
    # hand label then shuffle
    # =============================
    # num_neg = get_file_num(train_neg)
    # num_pos = get_file_num(train_pos)
    # # label
    # labels = np.concatenate((np.zeros(num_neg), np.ones(num_pos)))
    # =============================
    records = []
    for folder in folders:
        record = get_doc_list(folder)
        records.extend(record)
    dataset = pd.DataFrame.from_records(records, columns=['text', 'star'])
    dataset['target'] = 0
    dataset.loc[dataset['star'].astype('int64') >= 7, 'target'] = 1

    shuffle_data = dataset.sample(frac=1, random_state=shuffle_seed)

    # split
    RECORDS_NUM = len(shuffle_data)
    train_bound = int(split_bound[0]* RECORDS_NUM)
    val_bound = int((split_bound[0] + split_bound[1]) * RECORDS_NUM)
    train_data = shuffle_data.iloc[:train_bound, :]
    val_data = shuffle_data.iloc[train_bound:val_bound, :]
    test_data = shuffle_data.iloc[val_bound:, :]



    if not os.path.exists(data_file):
        os.mkdir(data_file)
    if not os.path.exists(train_csv):
        print('Generating {}...'.format(train_csv))
        train_data.to_csv(train_csv, index=False, encoding='utf-8')
        print('Finished', '-'*80)
    if not os.path.exists(val_csv):
        print('Generating {}...'.format(val_csv))
        val_data.to_csv(val_csv, index=False, encoding='utf-8')
        print('Finished','-'*80)
    if not os.path.exists(test_csv):
        print('Generating {}...'.format(test_csv))
        test_data.to_csv(test_csv, index=False, encoding='utf-8')
        print('Finished','-'*80)
    else:
        print('%s, %s , %s already exists!' % (train_csv, val_csv, test_csv))


def del_csv(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    return

if __name__=="__main__":

    train_folders = [train_neg, train_pos]
    test_folders = [test_neg, test_pos]

    # del file and regenerate
    if os.path.exists(data_file):
        del_csv(data_file)
    """   
    # generate train and test csv file if not exist
    if not os.path.exists(data_file):
        os.mkdir(data_file)
    # generate training set
    if not os.path.exists(train_csv):
        generate_csv(train_folders, csv_file=train_csv)
    print('Training set generated!')
    """

    # split test and validation set
    if not os.path.exists(test_csv) and not os.path.exists(val_csv):
        split_data(folders=[train_folders, test_folders])
    print('Validation and test set generated!')
    print('-'*80)




