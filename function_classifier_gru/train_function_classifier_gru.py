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
        gen_data.gen_function_vals_csv(-50, -50 + (config.epoch_size + config.sequence_length)*0.02, 0.02, lambda x: math.sin(x),
                                       config.data_tmp_folder + 'sine.csv')
        gen_data.gen_function_vals_csv(-50, -50 + (config.epoch_size + config.sequence_length)*0.02, 0.02, lambda x: x*0.8 + 0.04,
                                       config.data_tmp_folder + 'lin.csv')

    print("Reading Data from CSV")
    # Labels for sine and linear
    gen_data.set_string_from_label_index(0, 'sine')
    gen_data.set_string_from_label_index(1, 'linear')
    sine_x, sine_y = gen_data.read_function_vals_csv('x', 'y', config.data_tmp_folder + 'sine.csv')
    # sine_x: [TOTAL_LENGTH, 1]
    # sine_y:  [TOTAL_LENGTH, INPUT_DIMENSION]
    lin_x, lin_y = gen_data.read_function_vals_csv('x', 'y', config.data_tmp_folder + 'lin.csv')
    # lin_x: [TOTAL_LENGTH, 1]
    # lin_y:  [TOTAL_LENGTH, INPUT_DIMENSION]
    train_labels = np.array([1, 2])
    # train_labels: [OUTPUT_DIMENSION]
    # ONly use y-values, time is useless information (time invariant Classification)
    data_sine = sine_y
    # data_sine: [TOTAL_LENGTH, 1]
    data_lin = lin_y
    # data_lin: [TOTAL_LENGTH, 1]

    print("Generating Epoch containing Batches")
    datasequences = np.stack((data_sine, data_lin), axis=0)
    # datasequences: [ OUTPUT_DIMENSION, TOTAL_LENGTH, INPUT_DIMENSION ]
    train_batches, train_labels = gen_data.all_batches_from_datasequences(
        datasequences, config.batch_num, config.batch_size, config.sequence_length)
    # train_batches: [ BATCHNUM, BATCHSIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # train_labels: [ BATCHNUM, BATCHSIZE ]
    test_batch, test_labels = gen_data.gen_batches_from_datasequences(
        datasequences, 1, config.test_batches_size, config.sequence_length)
    test_batch = test_batch[0]
    test_labels = test_labels[0]
    # test_batch: [ TEST_BATCHSIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # test_labels: [ TEST_BATCHSIZE ]

    # Create model
    print("Creating Model")

    # Global Step Counter
    with tf.name_scope('Global_Step') as scope:
        global_step = tf.Variable(0, trainable=False, name='Global_Step_Var')
        increment_global_step_op = tf.assign(global_step, global_step + 1)

    # Model
    Hin = np.zeros([config.batch_size, config.hidden_layer_size *
                    config.hidden_layer_depth], dtype=np.float32)
    # Hin: [ BATCH_SIZE, INTERNALSIZE * NLAYERS]
    train_x, train_y, train_H, train_keep, train_step, train_summary_op = lstmnet(
        global_step, "train", False)
    test_x, test_y, test_H, test_keep, test_step, test_summary_op = lstmnet(
        global_step, "test", True)

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

    # Plot test batch
    # test_batch: [ TEST_BATCHSIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # test_batch_plt = np.reshape(test_batch, (config.test_batches_size, config.sequence_length))
    # for i in range(config.test_batches_size):
    #     print(test_batch_plt[i].shape)
    #     plt.plot(test_batch_plt[i].tolist())
    #     print(test_labels[i])
    #     plt.show(block=True)

    # train model.
    with tf.Session(config=tfconfig) as sess:
        print("Setup")
        tf.global_variables_initializer().run()

        print("Training")

        for step in range(config.iters):
            # batch_xs, batch_ys = gen_data.rand_batch_from_dataepochs(
            #    train_features, config.batch_size)
            # test_batch_xs, test_batch_ys = gen_data.rand_batch_from_dataepochs(
            #    validation_features, config.batch_size)
            batch_xs = train_batches[step % (config.batch_num)]
            batch_ys = train_labels[step % (config.batch_num)]
            test_batch_xs = test_batch
            test_batch_ys = test_labels
            # batch_xs: [ BATCHSIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
            # batch_ys: [ BATCHSIZE ]
            # test_batch_xs: [ TEST_BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
            # test_batch_ys: [ TEST_BATCHSIZE ]
            if step % 100 == 0:  # summary step
                _, training_summary, test_summary = sess.run([train_step, train_summary_op, test_summary_op],
                                                             feed_dict={train_x: batch_xs, train_y: batch_ys, train_keep: config.pkeep, train_H: Hin,
                                                                        test_x: test_batch_xs, test_y: test_batch_ys, test_keep: 1.0, test_H: Hin})

                saver.save(sess, config.checkpoint_path)
                writer.add_summary(training_summary, step)
                writer.add_summary(test_summary, step)
            else:
                _ = sess.run([train_step], feed_dict={
                             train_x: batch_xs, train_y: batch_ys, train_keep: config.pkeep,  train_H: Hin})

            # Increment global step Counter
            sess.run(increment_global_step_op)


if __name__ == "__main__":
    main()
