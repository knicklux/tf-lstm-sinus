import math
import os
import time
import tempfile
from termcolor import colored

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from function_classifier_gru_estimator import lstmnet
import gen_data

# Implementation with tfrecords and compliance with tf estimator API

GENERATE_DATA = False


def classifier_gru_train_in_fn(train_file, test_file,
                               batch_size,
                               sequence_length,
                               input_dimension,
                               shuffle_capacity,
                               shuffle_threads,
                               shuffle_min_after_dequeue):

    with tf.name_scope('Input_Queue') as scope:

        train_data_queue = tf.train.string_input_producer([train_file])
        train_features, train_labels = gen_data.read_and_decode(
            train_data_queue, batch_size, sequence_length, input_dimension, shuffle_capacity, shuffle_threads, shuffle_min_after_dequeue)
        test_data_queue = tf.train.string_input_producer([test_file])
        test_features, test_labels = gen_data.read_and_decode(
            test_data_queue, batch_size, sequence_length, input_dimension, shuffle_capacity, shuffle_threads, shuffle_min_after_dequeue)

        data_dict = {
            'train_features': train_features,
            'test_features': test_features
        }

        labels_dict = {
            'train_labels': train_labels,
            'test_labels': test_labels
        }

    return data_dict, labels_dict


def main(argv):

    timestamp = str(math.trunc(time.time()))
    checkpoint_location = config.checkpoint_path + tempfile.mkdtemp()
    # Create checkpoint+checkpoint_path
    if not os.path.exists(checkpoint_location):
        os.makedirs(checkpoint_location)
    print(colored('    Saving graph to: ' + checkpoint_location, 'red'))

    # Create training data.
    if GENERATE_DATA or not os.path.exists(config.data_tmp_folder):
        if not os.path.exists(config.data_tmp_folder):
            os.makedirs(config.data_tmp_folder)
        print("Generating Data CSV")
        # List of lambdas: [lambda x: math.sin(x)]
        gen_data.gen_function_vals_csv(-50, -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02,
                                       lambda x: math.sin(x),
                                       config.data_tmp_folder + 'sine.csv')
        gen_data.gen_function_vals_csv(-50, -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02,
                                       lambda x: x*0.8 + 0.04,
                                       config.data_tmp_folder + 'lin.csv')

        print("Reading Data from CSV")
        sine_x, data_sine = gen_data.read_function_vals_csv(
            'x', 'y', config.data_tmp_folder + 'sine.csv')
        # sine_x: [TOTAL_LENGTH, 1]
        # data_sine:  [TOTAL_LENGTH, INPUT_DIMENSION]
        lin_x, data_lin = gen_data.read_function_vals_csv(
            'x', 'y', config.data_tmp_folder + 'lin.csv')
        # lin_x: [TOTAL_LENGTH, 1]
        # data_lin:  [TOTAL_LENGTH, INPUT_DIMENSION]

        print("Writing TFRecords")
        datasequences = np.stack((data_sine, data_lin), axis=0)
        # datasequences: [ OUTPUT_DIMENSION, TOTAL_LENGTH, INPUT_DIMENSION ]

        functionsequences, labels = gen_data.all_sequences_from_datasequence(
            datasequences, config.sequence_length)
        # functionsequences: [ TOTAL_SEQUENCE_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
        # labels: [ TOTAL_SEQUENCE_NUM ]
        # Set apart some test data
        test_functionsequences, test_labels = gen_data.rand_sequences_from_datasequences(
            functionsequences, labels, config.test_epoch_size, True)
        # test_functionsequences: [ TEST_EPOCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
        # test_labels: [ TEST_EPOCH_SIZE ]
        # functionsequences: [ SEQUENCE_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
        # labels: [ SEQUENCE_NUM ]

        gen_data.function_sequences_to_tfrecord(
            functionsequences, labels, config.data_tmp_folder+config.data_tfrecord_filename)
        gen_data.function_sequences_to_tfrecord(
            test_functionsequences, test_labels, config.data_tmp_folder+config.test_tfrecord_filename)

    # Limit used gpu memory.
    print("Configuring Tensorflow")
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.25
    econfig = tf.estimator.RunConfig(model_dir=checkpoint_location,
                                     tf_random_seed=config.seed,
                                     save_summary_steps=config.summary_iters,
                                     session_config=tfconfig,
                                     log_step_count_steps=config.summary_iters
                                     )

    # Create model
    print("Creating Model")

    Hin = np.zeros([config.batch_size, config.hidden_layer_size *
                    config.hidden_layer_depth], dtype=np.float32)
    # Hin: [ BATCH_SIZE, INTERNALSIZE * NLAYERS ]

    feature_cols = []
    feature_cols.append(tf.feature_column.numeric_column(key='sequence_values',
                                                         shape=[config.sequence_length, config.input_dimension],
                                                         dtype=tf.float32))
    test_feature_cols = []
    test_feature_cols.append(tf.feature_column.numeric_column(key='sequence_values',
                                                              shape=[config.sequence_length, config.input_dimension],
                                                              dtype=tf.float32))

    # Model
    classifier = tf.estimator.Estimator(model_fn=lstmnet,
                                        params={
                                            'feature_columns': feature_cols,
                                            'test_feature_columns': test_feature_cols,
                                            'Hin': Hin,
                                            'batch_size': config.batch_size,
                                            'sequence_length': config.sequence_length,
                                            'input_dimension': config.input_dimension,
                                            'hidden_layer_size': config.hidden_layer_size,
                                            'hidden_layer_depth': config.hidden_layer_depth,
                                            'output_dimension': config.output_dimension,
                                            'learning_rate': config.learning_rate,
                                            'decay_rate': config.decay_rate,
                                            'decay_steps': config.decay_steps,
                                            'pkeep': config.pkeep,
                                            'do_test': True,
                                        },
                                        config=econfig
                                        )

    # Input Function to pass:
    # Let's try an input dimension of 2, 1 is boring
    def pipe_train(): return classifier_gru_train_in_fn(config.data_tmp_folder + config.data_tfrecord_filename,
                                                        config.data_tmp_folder + config.test_tfrecord_filename,
                                                        config.batch_size,
                                                        config.sequence_length,
                                                        1,
                                                        config.shuffle_capacity,
                                                        config.shuffle_threads,
                                                        config.shuffle_min_after_dequeue
                                                        )
    def pipe_test(): return classifier_gru_train_in_fn(config.data_tmp_folder + config.test_tfrecord_filename,
                                                       config.data_tmp_folder + config.test_tfrecord_filename,
                                                       config.test_batch_size,
                                                       config.sequence_length,
                                                       1,
                                                       config.shuffle_capacity,
                                                       config.shuffle_threads,
                                                       config.shuffle_min_after_dequeue
                                                       )

    # train
    print("Training")
    classifier.train(input_fn=pipe_train, steps=config.iters)

    print("Evaluating")
    eval_result = classifier.evaluate(input_fn=pipe_test, steps=1500)
    print('\nTest set accuracy: {train_streamed_accuracy:0.3f}\n'.format(**eval_result))

    print('Evaluating reconstructed net for verification')

    eval_classifier = tf.estimator.Estimator(model_fn=lstmnet,
                                             params={
                                                 'feature_columns': feature_cols,
                                                 'Hin': Hin,
                                                 'batch_size': config.batch_size,
                                                 'sequence_length': config.sequence_length,
                                                 'input_dimension': config.input_dimension,
                                                 'hidden_layer_size': config.hidden_layer_size,
                                                 'hidden_layer_depth': config.hidden_layer_depth,
                                                 'output_dimension': config.output_dimension,
                                                 'learning_rate': config.learning_rate,
                                                 'decay_rate': config.decay_rate,
                                                 'decay_steps': config.decay_steps,
                                                 'pkeep': 1.0,
                                                 'do_test': False
                                             },
                                             config=econfig)

    eval_result = eval_classifier.evaluate(input_fn=pipe_test, steps=1500)
    print('\nVerification test set accuracy: {train_streamed_accuracy:0.3f}\n'.format(**eval_result))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
