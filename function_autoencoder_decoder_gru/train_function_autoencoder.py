import math
import os
import time
import tempfile
from termcolor import colored

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from function_autoencoder_gru import lstmnet
from function_convautoencoder_gru import convlstmnet
from function_autoencoder_gru_v2 import lstmnetv2
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
        train_features_dict, train_labels = gen_data.read_and_decode(
            train_data_queue, batch_size, sequence_length, input_dimension, shuffle_capacity, shuffle_threads, shuffle_min_after_dequeue)
        test_data_queue = tf.train.string_input_producer([test_file])
        test_features_dict, test_labels = gen_data.read_and_decode(
            test_data_queue, batch_size, sequence_length, input_dimension, shuffle_capacity, shuffle_threads, shuffle_min_after_dequeue)
        train_features = train_features_dict['sequence_values']
        test_features = test_features_dict['sequence_values']

        # Mentioned modification is here:
        # Get elements 1 to seqlen-1 for features
        train_features_s = tf.slice(train_features, [0,1,0],[train_features.shape[0], train_features.shape[1]-1,train_features.shape[2]])
        test_features_s = tf.slice(test_features, [0,1,0],[test_features.shape[0], test_features.shape[1]-1,test_features.shape[2]])
        # And elements 0 to seqlen -2 for labels
        train_labels = tf.slice(train_features, [0,0,0],[train_features.shape[0], train_features.shape[1]-1,train_features.shape[2]])
        test_labels = tf.slice(test_features, [0,0,0],[test_features.shape[0], test_features.shape[1]-1,test_features.shape[2]])

        data_dict = {
            'train_features': {'sequence_values': train_features_s},
            'test_features': {'sequence_values': test_features_s}
        }
        labels_dict = {
            'train_labels': train_labels,
            'test_labels': test_labels
        }

    return data_dict, labels_dict


def main(argv):

    timestamp = str(math.trunc(time.time()))
    checkpoint_location = config.checkpoint_path + "/evalv2net17"
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
        gen_data.gen_function_vals_csv(-50,
                                       -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02,
                                       lambda x: math.sin(x/5),
                                       config.data_tmp_folder + 'slow_sine.csv')
        gen_data.gen_function_vals_csv(-50,
                                       -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02,
                                       lambda x: math.sin(x),
                                       config.data_tmp_folder + 'medium_sine.csv')
        gen_data.gen_function_vals_csv(-50,
                                       -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02,
                                       lambda x: math.sin(3*x),
                                       config.data_tmp_folder + 'fast_sine.csv')
        gen_data.gen_function_vals_csv(-50,
                                       -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02,
                                       lambda x: x*0.8 + 0.04,
                                       config.data_tmp_folder + 'pos_lin.csv')
        gen_data.gen_function_vals_csv(-50,
                                       -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02,
                                       lambda x: -x*0.8 + 0.04,
                                       config.data_tmp_folder + 'neg_lin.csv')
        gen_data.gen_function_vals_csv(-50,
                                       -50 + (config.epoch_size + config.test_epoch_size + config.sequence_length)*0.02, 0.02,
                                       lambda x: 0.8 + 0.04,
                                       config.data_tmp_folder + 'const_lin.csv')

        print("Reading Data from CSV")
        _, slow_data_sine = gen_data.read_function_vals_csv(
            'x', 'y', config.data_tmp_folder + 'slow_sine.csv')
        _, medium_data_sine = gen_data.read_function_vals_csv(
            'x', 'y', config.data_tmp_folder + 'medium_sine.csv')
        _, fast_sine_data_sine = gen_data.read_function_vals_csv(
            'x', 'y', config.data_tmp_folder + 'fast_sine.csv')
        # data_sine:  [TOTAL_LENGTH, INPUT_DIMENSION]

        _, pos_data_lin = gen_data.read_function_vals_csv(
            'x', 'y', config.data_tmp_folder + 'pos_lin.csv')
        _, neg_data_lin = gen_data.read_function_vals_csv(
            'x', 'y', config.data_tmp_folder + 'neg_lin.csv')
        _, const_data_lin = gen_data.read_function_vals_csv(
            'x', 'y', config.data_tmp_folder + 'const_lin.csv')
        # data_lin:  [TOTAL_LENGTH, INPUT_DIMENSION]

        print("Writing TFRecords")
        datasequences = np.stack((slow_data_sine, medium_data_sine, fast_sine_data_sine,
                                  pos_data_lin, neg_data_lin, const_data_lin), axis=0)
        # datasequences: [ FUNC_NUM, TOTAL_LENGTH, INPUT_DIMENSION ]

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
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.225
    econfig = tf.estimator.RunConfig(model_dir=checkpoint_location,
                                     tf_random_seed=config.seed,
                                     save_summary_steps=config.summary_iters,
                                     session_config=tfconfig,
                                     log_step_count_steps=config.summary_iters
                                     )

    # Input Function to pass:
    # ! Modified to return the function values as labels.
    # Hacky, but it's a prototype.
    def pipe_train(): return classifier_gru_train_in_fn(config.data_tmp_folder + config.data_tfrecord_filename,
                        config.data_tmp_folder + config.test_tfrecord_filename,
                        config.batch_size,
                        config.sequence_length,
                        config.dimension,
                        config.shuffle_capacity,
                        config.shuffle_threads,
                        config.shuffle_min_after_dequeue
                        )
    def pipe_test(): return classifier_gru_train_in_fn(config.data_tmp_folder + config.test_tfrecord_filename,
                       config.data_tmp_folder + config.test_tfrecord_filename,
                       config.test_batch_size,
                       config.sequence_length,
                       config.dimension,
                       config.shuffle_capacity,
                       config.shuffle_threads,
                       config.shuffle_min_after_dequeue
                       )

    feature_cols = []
    feature_cols.append(tf.feature_column.numeric_column(key='sequence_values',
                                                         shape=[config.sequence_length-1, config.dimension],
                                                         dtype=tf.float32))
    test_feature_cols = []
    test_feature_cols.append(tf.feature_column.numeric_column(key='sequence_values',
                                                              shape=[config.sequence_length-1, config.dimension],
                                                              dtype=tf.float32))

    # Create model
    print("Creating Model")

    en_Hin = np.zeros([config.batch_size, config.encoder_hidden_layer_size *
                    config.encoder_hidden_layer_depth], dtype=np.float32)
    # Hin: [ BATCH_SIZE, ENCODER_INTERNALSIZE * ENCODER_NLAYERS ]
    de_Hin = np.zeros([config.batch_size, config.decoder_hidden_layer_size *
                    config.decoder_hidden_layer_depth], dtype=np.float32)
    # Hin: [ BATCH_SIZE, DECODER_INTERNALSIZE * DECODER_NLAYERS ]
    decoder_inital_time_sample = np.zeros([config.batch_size, config.dimension], dtype=np.float32)
    # decoder_inital_time_sample: [ DIMENSION ]

    # Model
    classifier = tf.estimator.Estimator(model_fn=lstmnetv2,
                                        params={
                                            'feature_columns': feature_cols,
                                            'test_feature_columns': test_feature_cols,
                                            'encoder_Hin': en_Hin,
                                            'decoder_Hin': de_Hin,
                                            'decoder_inital_time_sample': decoder_inital_time_sample,
                                            'sequence_length': config.sequence_length-1,
                                            'dimension': config.dimension,
                                            'encoder_hidden_layer_size': config.encoder_hidden_layer_size,
                                            'encoder_hidden_layer_depth': config.encoder_hidden_layer_depth,
                                            'bottleneck_size': config.bottleneck_size,
                                            'decoder_hidden_layer_size': config.decoder_hidden_layer_size,
                                            'decoder_hidden_layer_depth': config.decoder_hidden_layer_depth,
                                            'learning_rate': config.learning_rate,
                                            'decay_rate': config.decay_rate,
                                            'decay_steps': config.decay_steps,
                                            'max_gradient_norm': config.max_gradient_norm,
                                            'parallel_iters': config.parallel_iters,
                                            'pkeep': config.pkeep,
                                            'do_test': True,
                                        },
                                        config=econfig
                                        )

    # Train
    print("Training")
    classifier.train(input_fn=pipe_train, steps=config.iters)

    print("Evaluating")
    eval_result = classifier.evaluate(input_fn=pipe_test, steps=1500)
    print('\nTest set test_square_error: {test_square_error:0.3f}\n'.format(**eval_result))

    print('Inference')

    pred_en_Hin = np.zeros([config.eval_batch_size, config.encoder_hidden_layer_size *
                    config.encoder_hidden_layer_depth], dtype=np.float32)
    # Hin: [ BATCH_SIZE, ENCODER_INTERNALSIZE * ENCODER_NLAYERS ]
    pred_de_Hin = np.zeros([config.eval_batch_size, config.decoder_hidden_layer_size *
                    config.decoder_hidden_layer_depth], dtype=np.float32)
    # Hin: [ BATCH_SIZE, DECODER_INTERNALSIZE * DECODER_NLAYERS ]
    decoder_inital_time_sample = np.zeros([1, config.dimension], dtype=np.float32)
    # decoder_inital_time_sample: [ DIMENSION ]

    eval_classifier = tf.estimator.Estimator(model_fn=lstmnetv2,
                                             params={
                                                 'feature_columns': feature_cols,
                                                 'test_feature_columns': test_feature_cols,
                                                 'encoder_Hin': pred_en_Hin,
                                                 'decoder_Hin': pred_de_Hin,
                                                 'decoder_inital_time_sample': decoder_inital_time_sample,
                                                 'sequence_length': config.sequence_length-1,
                                                 'dimension': config.dimension,
                                                 'encoder_hidden_layer_size': config.encoder_hidden_layer_size,
                                                 'encoder_hidden_layer_depth': config.encoder_hidden_layer_depth,
                                                 'bottleneck_size': config.bottleneck_size,
                                                 'decoder_hidden_layer_size': config.decoder_hidden_layer_size,
                                                 'decoder_hidden_layer_depth': config.decoder_hidden_layer_depth,
                                                 'learning_rate': config.learning_rate,
                                                 'decay_rate': config.decay_rate,
                                                 'decay_steps': config.decay_steps,
                                                 'max_gradient_norm': config.max_gradient_norm,
                                                 'parallel_iters': config.parallel_iters,
                                                 'pkeep': 1.0,
                                                 'do_test': False,
                                             },
                                             config=econfig)

    # Read some functions from TFRecords
    print("Filename:")
    print(config.data_tmp_folder + config.data_tfrecord_filename)
    tfrecords_filename = config.data_tmp_folder + config.test_tfrecord_filename
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for idx in range(config.eval_seq_num):
        string_record = next(record_iterator)
        # Parse the next example
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # Get the features you stored (change to match your tfrecord writing code)
        sequence_length = int(example.features.feature['seqlen']
                                     .int64_list
                                     .value[0])

        dimension = int(example.features.feature['dim']
                                    .int64_list
                                    .value[0])

        sequence_string = (example.features.feature['seq_raw']
                                      .bytes_list
                                      .value[0])

        label_string = (example.features.feature['label_raw']
                                      .bytes_list
                                      .value[0])

        # Convert to a numpy array (change dtype to the datatype you stored)
        sequence = np.fromstring(sequence_string, dtype=np.float32)
        sequence = np.reshape(sequence, newshape=(sequence_length, dimension))
        sequence = sequence[:,0]
        labels = np.fromstring(label_string, dtype=np.uint8)
        labels = np.reshape(labels, newshape=(1,))
        # Print the shape; does it match your expectations?
        print('Sequence: ')
        print(sequence_length)
        print('Dimension: ')
        print(dimension)
        print('sequence-Shape: ')
        print(sequence.shape)
        print('Labels-Shape: ')
        print(labels.shape)
        print('Labels:')
        print(labels)

        # Plot the original function
        chain_size = int(sequence_length/1)
        points = list()
        for i in range(1):
            points.append((i+1)*chain_size)

        plt.plot(sequence.tolist())
        plt.show(block=True)

        # Prediction
        sequence = sequence[1:]
        sequence = np.tile(sequence[np.newaxis,:], [config.eval_batch_size, 1])
        # Find out, how to use a batch size of 1 ...
        predict_dict = {'train_features': {'sequence_values': sequence}}
        def pipe_eval(): return predict_dict, None

        predictions = eval_classifier.predict(input_fn=pipe_eval,
                                              predict_keys=('encoding', 'decoding'))
        print(predictions)
        zipped_predictions = zip(predictions)
        print(zipped_predictions)
        pred_dict = next(zipped_predictions)[0]
        print(pred_dict)
        encoded = pred_dict['encoding']
        decoded = pred_dict['decoding']

        # Print encoded vector
        print(encoded.tolist())

        # Plot decoded function
        plt.plot(decoded.tolist())
        plt.show(block=True)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
