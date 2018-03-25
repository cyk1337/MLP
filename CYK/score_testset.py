import __init__
from config.setting import *
from CYK.data_loader import load_test

from keras.models import load_model

(X_test, y_test)=load_test()

filepath = os.path.join(best_model_dir, 'bigram_fasttext_{epoch:02d}')
model = load_model(filepath)
# model.evaluate()