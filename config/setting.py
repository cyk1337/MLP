import os
# Imdb dataset path config
work_dir = os.path.dirname(os.path.abspath(__file__))
Imdb_file = os.path.join(work_dir,'dataset/aclImdb/')
# data file
train_pos = os.path.join( Imdb_file, 'train/pos')
train_neg = os.path.join(Imdb_file, 'train/neg')
test_pos = os.path.join(Imdb_file, 'train/pos')
test_neg = os.path.join(work_dir, Imdb_file, 'train/neg')