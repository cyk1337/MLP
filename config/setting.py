import os
# Imdb dataset path config
work_dir = os.path.dirname(os.path.abspath(__file__))
Imdb_file = os.path.join(work_dir,'dataset', 'aclImdb')
# data file
<<<<<<< HEAD
train_pos = os.path.join( Imdb_file, 'train', 'pos')
train_neg = os.path.join(Imdb_file, 'train', 'neg')
test_pos = os.path.join(Imdb_file, 'train', 'pos')
test_neg = os.path.join(work_dir, Imdb_file, 'train', 'neg')
=======
train_pos = os.path.join( Imdb_file, 'train/pos')
train_neg = os.path.join(Imdb_file, 'train/neg')
test_pos = os.path.join(Imdb_file, 'test/pos')
test_neg = os.path.join(work_dir, Imdb_file, 'test/neg')
>>>>>>> 4c49d8f3889fd31158f77fd7bcfa28cbd23ecd4f
