import math
import os
import time
import tempfile
from termcolor import colored

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from function_classifier_gru import lstmnet
import gen_data

GENERATE_DATA = True


def main():

    # Create checkpoint+checkpoint_path
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)

    # Create training data.
    if GENERATE_DATA or not os.path.exists(config.data_tmp_folder):
        if not os.path.exists(config.data_tmp_folder):
            os.makedirs(config.data_tmp_folder)
        print("Generating Data CSV")
        # List of lambdas: [lambda x: math.sin(x)]
        gen_data.gen_function_vals_csv(-50, -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02, lambda x: math.sin(x),
                                       config.data_tmp_folder + 'sine.csv')
        gen_data.gen_function_vals_csv(-50, -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02, lambda x: x*0.8 + 0.04,
                                       config.data_tmp_folder + 'lin.csv')

    print("Reading Data from CSV")
    # Labels for sine and linear
    gen_data.set_string_from_label_index(0, 'sine')
    gen_data.set_string_from_label_index(1, 'linear')
    sine_x, data_sine = gen_data.read_function_vals_csv('x', 'y', config.data_tmp_folder + 'sine.csv')
    # sine_x: [TOTAL_LENGTH, 1]
    # data_sine:  [TOTAL_LENGTH, INPUT_DIMENSION]
    lin_x, data_lin = gen_data.read_function_vals_csv('x', 'y', config.data_tmp_folder + 'lin.csv')
    # lin_x: [TOTAL_LENGTH, 1]
    # data_lin:  [TOTAL_LENGTH, INPUT_DIMENSION]

    print("Writing TFRecords")
    datasequences = np.stack((data_sine, data_lin), axis=0)
    # datasequences: [ OUTPUT_DIMENSION, SEQUENCE_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    functionsequences, labels = gen_data.all_sequences_from_datasequence(datasequences, config.sequence_length)
    # functionsequences: [ TOTAL_SEQUENCE_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # labels: [ TOTAL_SEQUENCE_NUM ]
    # Set apart some test data
    test_functionsequences, test_labels = gen_data.rand_sequences_from_datasequences(functionsequences, labels, config.test_epoch_size, True)
    # functionsequences: [ EPOCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # labels: [ EPOCH_SIZE ]
    # test_functionsequences: [ TEST_EPOCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # test_labels: [ TEST_EPOCH_SIZE ]
    gen_data.function_sequences_to_tfrecord(functionsequences, labels, config.data_tmp_folder+config.data_tfrecord_filename)
    gen_data.function_sequences_to_tfrecord(test_functionsequences, test_labels, config.data_tmp_folder+config.test_tfrecord_filename)

    with tf.name_scope('Input') as scope:

        data_queue = tf.train.string_input_producer([config.data_tmp_folder + config.data_tfrecord_filename])
        test_queue = tf.train.string_input_producer([config.data_tmp_folder + config.test_tfrecord_filename])

        sequences_batch, labels_batch = gen_data.read_and_decode(data_queue, config.batch_size, config.sequence_length, config.input_dimension, config.shuffle_capacity, config.shuffle_threads, config.shuffle_min_after_dequeue)
        test_sequences_batch, test_labels_batch = gen_data.read_and_decode(test_queue, config.batch_size, config.sequence_length, config.input_dimension, config.shuffle_capacity, config.shuffle_threads, config.shuffle_min_after_dequeue)

    # Global Step Counter
    with tf.name_scope('Global_Step') as scope:
        global_step = tf.Variable(0, trainable=False, name='Global_Step_Var')
        increment_global_step_op = tf.assign(global_step, global_step + 1)

    # Create model
    print("Creating Model")

    # Model
    Hin = np.zeros([config.batch_size, config.hidden_layer_size *
                    config.hidden_layer_depth], dtype=np.float32)
    # Hin: [ BATCH_SIZE, INTERNALSIZE * NLAYERS ]

    train_H, train_keep, train_step, train_summary_op = lstmnet(
        sequences_batch, labels_batch, global_step, "train", False)
    test_H, test_keep, test_step, test_summary_op = lstmnet(
        test_sequences_batch, test_labels_batch, global_step, "test", True)

    # Setup logging with Tensorboard
    print("Setup Tensorboard")
    timestamp = str(math.trunc(time.time()))
    graph_location = "log" + tempfile.mkdtemp()
    print(colored('    Saving graph to: ' + graph_location, 'red'))
    writer = tf.summary.FileWriter(graph_location, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    # Limit used gpu memory.
    print("Configuring Tensorflow")
    tfconfig = tf.ConfigProto()
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.75
    init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    # train model.
    with tf.Session(config=tfconfig) as sess:
        print("Setup")
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("Training")
        for step in range(config.iters):

            if step % config.summary_iters == 0:  # summary step
                _, training_summary, test_summary = sess.run([train_step, train_summary_op, test_summary_op],
                                                             feed_dict={train_keep: config.pkeep, train_H: Hin,
                                                                        test_keep: 1.0, test_H: Hin})

                saver.save(sess, config.checkpoint_path)
                writer.add_summary(training_summary, step)
                writer.add_summary(test_summary, step)
            else:
                _ = sess.run([train_step], feed_dict={train_keep: config.pkeep, train_H: Hin})

            # Increment global step Counter
            # sess.run(increment_global_step_op)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()
