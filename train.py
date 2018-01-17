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

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main(_):

    SEQLEN = 100
    BATCHSIZE = 200

    # init
#    istate = np.zeros([BATCHSIZE, n_hidden*NLAYERS])  # initial zero input state
#    init = tf.global_variables_initializer()
#    sess = tf.Session()
#    sess.run(init)
#    global_step = 0

    with tf.name_scope('Global_Step') as scope:
       global_step = tf.Variable(0, trainable=False, name='Global_Step_Var')
       increment_global_step_op = tf.assign(global_step, global_step + 1)

    with tf.name_scope('WeightsBiases') as scope:

        W1 = tf.Variable(tf.zeros([2,2]1), name='W1')
        B1 = tf.Variable(tf.zeros([2,1]), name='B1')

   train_x, train_dropout, H, train_y, trian_step, train_summary_op = model(W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, "train", global_step)

   # Setup logging with Tensorboard
   timestamp = str(math.trunc(time.time()))
   graph_location = "log" + tempfile.mkdtemp()
   print(colored('Saving graph to: ' + graph_location, 'red'))
   writer = tf.summary.FileWriter(graph_location, graph=tf.get_default_graph())

   with tf.Session() as sess:
       tf.global_variables_initializer().run()
       # Train
       for step in range(200000):
           batch_xs, batch_ys = mnist.train.next_batch(150)
           if step % 100 == 0:  # summary step
           _, training_summary, _, test_summary = sess.run([train_step, train_summary_op, test_step, test_summary_op],
                                                        feed_dict={train_dropout: 0.75, train_x: batch_xs, train_target: batch_ys,
                                                                   test_dropout: 1.0, test_x: mnist.test.images, test_target: mnist.test.labels})

            writer.add_summary(training_summary, step)
         writer.add_summary(test_summary, step)
     else:
         _ = sess.run([train_step], feed_dict={train_dropout: 0.75, train_x: batch_xs, train_target: batch_ys})

     # Incr counter
     sess.run(increment_global_step_op)



    # Old code
   # Import data
   mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

   with tf.name_scope('Global_Step') as scope:
       global_step = tf.Variable(0, trainable=False, name='Global_Step_Var')
       increment_global_step_op = tf.assign(global_step, global_step + 1)

   # Overkill Stack
   cnn_1_chans = 6
   cnn_2_chans = 12
   cnn_3_chans = 24
   dense_5_neurons = 200

   with tf.name_scope('WeightsBiases') as scope:

       W1 = tf.Variable(tf.truncated_normal([6,6,1,cnn_1_chans], stddev=0.1), name='W1')
       B1 = tf.Variable(tf.ones([cnn_1_chans])/10, name='B1')

       W2 = tf.Variable(tf.truncated_normal([5,5,cnn_1_chans,cnn_2_chans], stddev=0.1), name='W2')
       B2 = tf.Variable(tf.ones([cnn_2_chans])/10, name='B2')

       W3 = tf.Variable(tf.truncated_normal([4,4,cnn_2_chans,cnn_3_chans], stddev=0.1), name='W3')
       B3 = tf.Variable(tf.ones([cnn_3_chans])/10, name='B3')

       W4 = tf.Variable(tf.truncated_normal([7*7*cnn_3_chans, dense_5_neurons], stddev=0.1), name='W4')
       B4 = tf.Variable(tf.ones([dense_5_neurons])/10, name='B4')

       W5 = tf.Variable(tf.truncated_normal([dense_5_neurons, 10], stddev=0.1), name='W5')
       B5 = tf.Variable(tf.zeros([10]), name='B5')

   train_x, train_dropout, train_target, train_step, train_summary_op = model(W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, "train", global_step)
   test_x, test_dropout, test_target, test_step, test_summary_op = model(W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, "test", global_step)

   # Setup logging with Tensorboard
   timestamp = str(math.trunc(time.time()))
   graph_location = "log" + tempfile.mkdtemp()
   print(colored('Saving graph to: ' + graph_location, 'red'))
   writer = tf.summary.FileWriter(graph_location, graph=tf.get_default_graph())

   with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # Train
    for step in range(200000):
     batch_xs, batch_ys = mnist.train.next_batch(150)
     if step % 100 == 0:  # summary step
         _, training_summary, _, test_summary = sess.run([train_step, train_summary_op, test_step, test_summary_op],
                                                        feed_dict={train_dropout: 0.75, train_x: batch_xs, train_target: batch_ys,
                                                                   test_dropout: 1.0, test_x: mnist.test.images, test_target: mnist.test.labels})

         writer.add_summary(training_summary, step)
         writer.add_summary(test_summary, step)
     else:
         _ = sess.run([train_step], feed_dict={train_dropout: 0.75, train_x: batch_xs, train_target: batch_ys})

     # Incr counter
     sess.run(increment_global_step_op)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
   FLAGS, unparsed = parser.parse_known_args()
   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
