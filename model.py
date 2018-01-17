from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time
import math
import tempfile
from termcolor import colored

from tensorflow.contrib import rnn

import tensorflow as tf

FLAGS = None

n_hidden = 30
n_input = 1
n_output = 2
NLAYERS = 2

def lstmnet(pkeep, phase, global_step, Hin):

    with tf.name_scope('Input') as scope:
        # [ BATCHSIZE, SEQLEN, n_input]
        x = tf.placeholder(tf.float32, [None, None, n_input], name = 'x')
        # [BATCHSIZE, n_hidden, NLAYERS] Remember: Only H needed for first GRU,
        # since its H will be used for the next a.s.o.
        H = tf.placeholder(tf.float32, [None, n_hidden, NLAYERS], name='H')
        # [BATCHSIZE, SEQLEN, n_output]
        y_ = tf.placeholder(tf.float32, [None, None, n_output], name = 'y')

    with tf.name_scope('params') as scope:
        # Just give them names:
        pkeep = tf.identity(pkeep, 'Dropout pkeep')
        learning_rate = tf.identity(learning_rate, 'Learning Rate')
        global_step = tf.identity(global_step, 'Global Step')

    with tf.name_scope('NeuralNet') as scope:
        X = tf.reshape(x, [-1, n_input])
        #rnn_cell = rnn.BasicLSTMCell(n_hidden)
        cells = [rnn.GRUCell(n_hidden) for _ in range(NLAYERS)]
        # "naive dropout" implementation
        dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
        multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
        multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep) # dropout for the softmax layer
        # y, states = rnn.static_rnn(rnn_cell, X, dtype=tf.float32)
        Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=Hin)
        # Yr: [ BATCHSIZE, SEQLEN, n_hidden ]
        # H:  [ BATCHSIZE, n_hidden*NLAYERS ] # this is the last state in the sequence
        Yflat = tf.reshape(Yr, [-1, n_hidden]) [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
        Yflat_ = tf.reshape(y__, [-1, n_output]) # [ BATCHSIZE x SEQLEN, n_output ]
        output = tf.matmul(Yflat[-1], W1['out']) + B1['out']

    with tf.name_scope('LearingRate') as scope:
        starter_learning_rate = 0.03
        learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, 100000, 4)

    with tf.name_scope('Loss_Accuracy') as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Yflat_, logits=output))

    with tf.name_scope('Train_Step') as scope:
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # accuracy
    #with tf.name_scope('Summary') as scope:
    #    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #    tf.summary.scalar(phase + "/loss", cross_entropy)
    #    tf.summary.scalar(phase + "/acc", accuracy)
    #    tf.summary.scalar(phase + "/lr", learning_rate)
    #    summary_op = tf.summary.merge_all()

    return x, pkeep, states, y_, train_step, summary_op

def model(W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, phase, global_step):

    with tf.name_scope('Images') as scope:9
        x = tf.placeholder(tf.float32, [None, 784])

    with tf.name_scope('params') as scope:
        pkeep = tf.placeholder(tf.float32)

    with tf.name_scope('NeuralNet') as scope:

        X = tf.reshape(x, [-1, 28, 28, 1])
        Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + B1, name='conv_1')
        Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,2,2,1], padding='SAME') + B2, name='conv_2')
        Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,2,2,1], padding='SAME') + B3, name='conv_3')
        YY = tf.reshape(Y3, shape=[-1, 7*7*24], name='reshape')
        Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4, name='dense_4')
        Y4f = tf.nn.dropout(Y4, pkeep, name='dense_4_dropout')
        y = tf.nn.softmax(tf.matmul(Y4f, W5) + B5, name='softmax')

        # True output
        y_ = tf.placeholder(tf.float32, [None, 10], name='label')

    with tf.name_scope('LearningRate') as scope:
        starter_learning_rate = 0.03
        learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, 100000, 4)

    with tf.name_scope('Loss_Accuracy') as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    with tf.name_scope('Train_Step') as scope:
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # accuracy
    with tf.name_scope('Summary') as scope:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar(phase + "/loss", cross_entropy)
        tf.summary.scalar(phase + "/acc", accuracy)
        tf.summary.scalar(phase + "/lr", learning_rate)
        summary_op = tf.summary.merge_all()

    return x, pkeep, y_, train_step, summary_op
