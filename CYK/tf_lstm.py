import tensorflow as tf

import __init__
from config.setting import *
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re, time
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from collections import namedtuple

from CYK.data_loader import load_imdb, load_test
from CYK.plot_fit import plot_fit, visialize_model, save_history, plot_all_history

max_num=500
(X_train, y_train), (X_val, y_val) = load_imdb()
(X_test, y_test) = load_test()
tokenizer = Tokenizer(num_words=max_num)
tokenizer.fit_on_texts(X_train)
x_train = tokenizer.texts_to_sequences(X_train)
x_valid = tokenizer.texts_to_sequences(X_val)
x_test = tokenizer.texts_to_sequences(X_test)

maxlen = max(len(i) for i in x_train)
x_train = sequence.pad_sequences(x_train, maxlen)
x_valid = sequence.pad_sequences(x_valid, maxlen)
x_test = sequence.pad_sequences(x_test, maxlen)


def get_batches(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


def get_test_batches(x, batch_size):
    '''Create the batches for the testing data'''
    n_batches = len(x)//batch_size
    x = x[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size]


def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers,
              dropout, learning_rate, multiple_fc, fc_units):
    '''Build the Recurrent Neural Network'''

    tf.reset_default_graph()

    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Create the embeddings
    with tf.name_scope("embeddings"):
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

    # Build the RNN layers
    with tf.name_scope("RNN_layers"):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

    # Set the initial state
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)

    # Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
        #shapes_list = [item.get_shape().as_list() for item in cell]
        #initial_states_shapes_list = [item.get_shape().as_list() for item in initial_state]
        # print(inputs.get_shape().as_list())
        # print(embed.get_shape().as_list())
        # print(initial_states_shapes_list)
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                 initial_state=initial_state)

     # Create the fully connected layers
    with tf.name_scope("fully_connected"):
        # Initialize the weights and biases
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()

        dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                                                  num_outputs=fc_units,
                                                  activation_fn=tf.sigmoid,
                                                  weights_initializer=weights,
                                                  biases_initializer=biases)
        dense = tf.contrib.layers.dropout(dense, keep_prob)

        # Depending on the iteration, use a second fully connected layer
        if multiple_fc == True:
            dense = tf.contrib.layers.fully_connected(dense,
                                                      num_outputs=fc_units,
                                                      activation_fn=tf.sigmoid,
                                                      weights_initializer=weights,
                                                      biases_initializer=biases)
            dense = tf.contrib.layers.dropout(dense, keep_prob)

    # Make the predictions
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(dense,
                                                        num_outputs=1,
                                                        activation_fn=tf.sigmoid,
                                                        weights_initializer=weights,
                                                        biases_initializer=biases)
        tf.summary.histogram('predictions', predictions)

    # Calculate the cost
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('cost', cost)

    # Train the model
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Determine the accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # Merge all of the summaries
    merged = tf.summary.merge_all()

    # Export the nodes
    export_nodes = ['inputs', 'labels', 'keep_prob', 'initial_state', 'final_state', 'accuracy',
                    'predictions', 'cost', 'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


def train(model, epochs, log_string):
    '''Train the RNN'''

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        valid_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/3/train/{}'.format(log_string), sess.graph)
        valid_writer = tf.summary.FileWriter('./logs/3/valid/{}'.format(log_string))

        for e in range(epochs):
            state = sess.run(model.initial_state)

            # Record progress with each epoch
            train_loss = []
            train_acc = []
            val_acc = []
            val_loss = []

            with tqdm(total=len(x_train)) as pbar:
                for _, (x, y) in enumerate(get_batches(x_train, y_train, batch_size), 1):
                    # print(y.shape, y.values.reshape(-1,1).shape, '\n', '-'*80)
                    feed = {model.inputs: x,
                            model.labels: y.values.reshape(-1,1),
                            model.keep_prob: dropout,
                            model.initial_state: state}
                    summary, loss, acc, state, _ = sess.run([model.merged,
                                                             model.cost,
                                                             model.accuracy,
                                                             model.final_state,
                                                             model.optimizer],
                                                            feed_dict=feed)

                    # Record the loss and accuracy of each training batch
                    train_loss.append(loss)
                    train_acc.append(acc)

                    # Record the progress of training
                    train_writer.add_summary(summary, iteration)

                    iteration += 1
                    pbar.update(batch_size)

            # Average the training loss and accuracy of each epoch
            avg_train_loss = np.mean(train_loss)
            avg_train_acc = np.mean(train_acc)

            val_state = sess.run(model.initial_state)
            with tqdm(total=len(x_valid)) as pbar:
                for x, y in get_batches(x_valid, y_valid, batch_size):
                    feed = {model.inputs: x,
                            model.labels: y.values.reshape(-1,1),
                            model.keep_prob: 1,
                            model.initial_state: val_state}
                    summary, batch_loss, batch_acc, val_state = sess.run([model.merged,
                                                                          model.cost,
                                                                          model.accuracy,
                                                                          model.final_state],
                                                                         feed_dict=feed)

                    # Record the validation loss and accuracy of each epoch
                    val_loss.append(batch_loss)
                    val_acc.append(batch_acc)
                    pbar.update(batch_size)

            # Average the validation loss and accuracy of each epoch
            avg_valid_loss = np.mean(val_loss)
            avg_valid_acc = np.mean(val_acc)
            valid_loss_summary.append(avg_valid_loss)

            # Record the validation data's progress
            valid_writer.add_summary(summary, iteration)

            # Print the progress of each epoch
            print("Epoch: {}/{}".format(e, epochs),
                  "Train Loss: {:.3f}".format(avg_train_loss),
                  "Train Acc: {:.3f}".format(avg_train_acc),
                  "Valid Loss: {:.3f}".format(avg_valid_loss),
                  "Valid Acc: {:.3f}".format(avg_valid_acc))

            # Stop training if the validation loss does not decrease after 3 epochs
            if avg_valid_loss > min(valid_loss_summary):
                print("No Improvement.")
                stop_early += 1
                if stop_early == 3:
                    break

            # Reset stop_early if the validation loss finds a new low
            # Save a checkpoint of the model
            else:
                print("New Record!")
                stop_early = 0
                checkpoint = "sentiment_{}.ckpt".format(log_string)
                saver.save(sess, checkpoint)




word_index = tokenizer.word_index


# The default parameters of the model
n_words = max_num
#len(word_index)
embed_size = 300
batch_size = 100
lstm_size = 64
num_layers = 1
dropout = 0.5
learning_rate = 0.001
epochs = 100
multiple_fc = False
fc_units = 256


# Train the model with the desired tuning parameters
for lstm_size in [64,128]:
    for multiple_fc in [True, False]:
        for fc_units in [128, 256]:
            log_string = 'ru={},fcl={},fcu={}'.format(lstm_size,
                                                      multiple_fc,
                                                      fc_units)
            model = build_rnn(n_words = n_words,
                              embed_size = embed_size,
                              batch_size = batch_size,
                              lstm_size = lstm_size,
                              num_layers = num_layers,
                              dropout = dropout,
                              learning_rate = learning_rate,
                              multiple_fc = multiple_fc,
                              fc_units = fc_units)
            print('Start train...')
            train(model, epochs, log_string)
            print('End train...')