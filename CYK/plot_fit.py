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

@file: plot_fit.py

@time: 11/02/2018 10:31

@desc:         
               
'''              

import os
import matplotlib.pyplot as plt



def plot_fit(history, plot_filename):
    assert len(history.history)==4, "Error: did not fit validation data!"
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'b', label='Training acc')
    # b is for "solid blue line"
    plt.plot(epochs, val_acc, 'y', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    save_fig(plt, plot_filename=plot_filename)
    plt.show()


def save_fig(plt, plot_filename, plot_dir='plot'):

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    filename = os.path.join(plot_dir, plot_filename)
    plt.savefig('{}.pdf'.format(filename))
    print('{} saved!'.format(filename))
