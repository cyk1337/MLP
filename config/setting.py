import os

#=======================
# Imdb dataset path config
#=======================

work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Imdb_file = os.path.join(work_dir, 'dataset', 'aclImdb')
# data path
train_pos = os.path.join( Imdb_file, 'train', 'pos')
train_neg = os.path.join(Imdb_file, 'train', 'neg')
test_pos = os.path.join(Imdb_file, 'train', 'pos')
test_neg = os.path.join(work_dir, Imdb_file, 'train', 'neg')

# csv raw docs
data_file = 'csv'
train_csv = os.path.join(data_file, 'train.csv')
test_csv = os.path.join(data_file,'test.csv')

# dir path
embedding_dir = os.path.join(work_dir, 'embedding')
plot_dir = os.path.join(work_dir, 'plot')
log_dir = os.path.join(work_dir, 'logs')

# sequence param
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100
VALIDATION_SLPIT = 0.2

SEED = 2018




