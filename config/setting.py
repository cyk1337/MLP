import os

#=======================
# Imdb dataset path config
#=======================

work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Imdb_file = os.path.join(work_dir, 'dataset', 'aclImdb')
# data path
train_pos = os.path.join( Imdb_file, 'train', 'pos')
train_neg = os.path.join(Imdb_file, 'train', 'neg')
test_pos = os.path.join(Imdb_file, 'test', 'pos')
test_neg = os.path.join(work_dir, Imdb_file, 'test', 'neg')

# csv raw docs
data_file = os.path.join(work_dir, 'csv')
train_csv = os.path.join(data_file, 'train.csv')
val_csv = os.path.join(data_file,'val.csv')
test_csv = os.path.join(data_file,'test.csv')


# dir path
embedding_dir = os.path.join(work_dir, 'embedding')
plot_dir = os.path.join(work_dir, 'plot')
log_dir = os.path.join(work_dir, 'logs')
model_dir = os.path.join(work_dir, 'model')
history_dir = os.path.join(work_dir, 'history')

CBOW_embedding = os.path.join(embedding_dir, '100d_cbow.txt')
SkipGram_embedding = os.path.join(embedding_dir, '100d_skipgram.txt')
Glove_embedding = os.path.join(embedding_dir, 'glove.6B.100d.txt')


# sequence param
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100
EPOCH_NUM = 15
BATCH_SIZE = 128

SEED = 2018
RECORDS_NUM = 25000




