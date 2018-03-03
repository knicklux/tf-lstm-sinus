import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.ops.metrics_impl import metric_variable

from fixed_len_numeric_training_helper import fixed_len_sample_ids_shape
from fixed_len_numeric_training_helper import create_fixed_len_numeric_training_helper

# lstmnet:

# features dict:
# sequence_vals: [ BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION]
# Maybe later: Hin
# Maybe later: Context info at each timestep

# train_labels: [ BATCH_SIZE ]
# test_labels: [ BATCH_SIZE ]

# features: dictionary mapping:
#   'train_features' to an estimator string->tensor dictionary
#   'test_features' to an estimator string->tensor dictionary
# labels: dictionary mapping:
#   'train_labels' to training labels
#   'test_labels' to test labels

# Params: dictionary with keys:
# test_features_dict (if do_test)
# test_feature_columns (if do_test)
# test_labels (if do_test)
# feature_columns
# encoder_Hin
# decoder_Hin
# decoder_inital_time_sample
# sequence_length
# dimension
# encoder_hidden_layer_size
# encoder_hidden_layer_depth
# bottleneck_size
# decoder_hidden_layer_size
# decoder_hidden_layer_depth
# learning_rate
# decay_rate
# decay_steps
# max_gradient_norm
# parallel_iters
# pkeep
# do_test

# Major shortcoming of estimators: They do not plot test loss during training.
# However, test loss is a major indication for overfitting.
# So how could this be solved?

# 1) Training hook
# 2) Separate loss op
# 3) Test data input queue in params dict
# Sound solution. Test data would be optional and only affect testing
# 4) Train a bit, test a bit, train a bit, test a bit
# And wait for the HDD while storing and restoring parameters
# On the other hand... Not much work involved :^)
# Chose a ... variant of 3)

# How to specify different pkeeps during training for testing?
# 1) Two networks... plausible, but might cause problems with var reuse
# and export/import
# 2) Set pkeep to something else
# 3) Something like feed_dict
# Chose 1)


def _lstmnet(
        features,  # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params,
        is_test):

    with tf.variable_scope('EncoderNet') as scope:
        if is_test:
            scope.reuse_variables()

        if(mode == tf.estimator.ModeKeys.TRAIN and not is_test):
            # Train graph
            pkeep = params['pkeep']
        else:
            # Test or inference graph
            pkeep = 1.0

        x = tf.feature_column.input_layer(
            features,
            feature_columns=params['feature_columns'])
        X = tf.reshape(x, shape=[x.get_shape()[0],
                                 params['sequence_length'], params['dimension']])
        X = tf.identity(X, name='X')
        # X: [ BATCH_SIZE, SEQUENCE_LENGTH, DIMENSION]
        if labels is not None:
            Labels = tf.reshape(labels, shape=[x.get_shape()[0], params['sequence_length'], params['dimension']])
        else:
            Labels = None
        encoder_Hin = params['encoder_Hin']
        # encoder_Hin: [ BATCH_SIZE, ENCODER_INTERNALSIZE * ENCODER_NLAYERS]
        seqlen = tf.Variable(params['sequence_length'], name='seqlen')
        seqlen = tf.reshape(seqlen, shape=[1])
        seqdescr = tf.tile(seqlen, multiples=[x.get_shape()[0]])
        # seqdescr: [ BATCHSIZE ]
        inital_time_sample = params['decoder_inital_time_sample']
        # inital_time_sample: [ BATCH_SIZE, DIMENSION ]

        encoder_cells = [rnn.GRUBlockCell(params['encoder_hidden_layer_size'])
                         for _ in range(params['encoder_hidden_layer_depth'])]
        # "naive dropout" implementation
        encoder_dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep)
                             for cell in encoder_cells]
        encoder_multicell = rnn.MultiRNNCell(encoder_dropcells, state_is_tuple=False)
        # Input wrapper to keep symmetry with decoder
        encoder_multicell = rnn.InputProjectionWrapper (encoder_multicell,
                                                        num_proj=params['bottleneck_size'],
                                                        activation=tf.nn.relu)
        # dropout for the softmax layer
        # No dropout in bottleneck layer!
        # encoder_multicell = rnn.DropoutWrapper(encoder_multicell, output_keep_prob=pkeep)

        encoded_Yr, encoded_H = tf.nn.dynamic_rnn(encoder_multicell, X, dtype=tf.float32,
                                                  initial_state=encoder_Hin, scope='EncoderNet',
                                                  parallel_iterations=params['parallel_iters'])
        encoded_H = tf.identity(encoded_H, name='encoded_H')  # just to give it a name
        encoded_Yr = tf.identity(encoded_Yr, name='endoded_Yr')
        # encoder_Yr: [ BATCH_SIZE, SEQUENCE_LENGTHLEN, ENCODER_INTERNALSIZE ]
        # encoder_H:  [ BATCH_SIZE, ENCODER_INTERNALSIZE * ENCODER_NLAYERS ] # this is the last state in the sequence

        encoded_V = tf.reshape(encoded_H, [x.get_shape()[0], -1])
        # encoded_V: [ BATCH_SIZE, BOTTLENECK_SIZE ]

    with tf.variable_scope('NetDecoder') as scope:
        if is_test:
            scope.reuse_variables()

        if(mode == tf.estimator.ModeKeys.TRAIN and not is_test):
            pkeep = params['pkeep']
        else:
            pkeep = 1.0

        decoder_Hin = encoded_H
        # decoder_Hin: [ BATCH_SIZE, DECODER_INTERNALSIZE * DECODER_NLAYERS]

        decoder_cells = [rnn.GRUBlockCell(params['decoder_hidden_layer_size'])
                         for _ in range(params['decoder_hidden_layer_depth'])]
        # "naive dropout" implementation
        decoder_dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep)
                             for cell in decoder_cells]
        decoder_multicell = rnn.MultiRNNCell(decoder_dropcells, state_is_tuple=False)
        # dropout for the softmax layer
        decoder_multicell = rnn.DropoutWrapper(decoder_multicell, output_keep_prob=pkeep)
        # dense layer to adjust dimensions
        decoder_multicell = rnn.OutputProjectionWrapper(decoder_multicell,
                                                        params['dimension'],
                                                        activation=None)

        custom_Helper = create_fixed_len_numeric_training_helper(inital_time_sample, params['sequence_length'], X.dtype)
        #helper = tf.contrib.seq2seq.TrainingHelper(inputs=Labels,
        #                                           sequence_length=seqdescr,
        #                                           time_major=False)
        decoder = seq2seq.BasicDecoder(cell=decoder_multicell,
                                       helper=custom_Helper,
                                       initial_state=decoder_Hin)
        decoded_Yr, decoded_H, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                  output_time_major=False,
                                                                  impute_finished=False,
                                                                  maximum_iterations=None,
                                                                  parallel_iterations=params['parallel_iters'])

        decoded_Yr = decoded_Yr.rnn_output
        print('decoded_Yr')
        print(decoded_Yr)
        decoded_Yr.set_shape([decoded_Yr.get_shape()[0],params['sequence_length'], decoded_Yr.get_shape()[2]])
        print(decoded_Yr)
        decoded_H = tf.identity(decoded_H, name='decoded_H')
        decoded_Yr = tf.identity(decoded_Yr, name='decoded_Yr')
        # decoder_Yr: [ BATCH_SIZE, SEQUENCE_LENGTHLEN, DIMENSION ]
        # decoder_H:  [ BATCH_SIZE, DECODER_INTERNALSIZE * DECODER_NLAYERS ] # this is the last state in the sequence

    return decoded_Yr, encoded_V  # = encoded_H reshaped


def lstmnetv2(
        features,  # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params):

    # Extract labels
    if labels is not None:
        train_labels = labels['train_labels']
    else:
        train_labels = None
    if labels is not None and params['do_test']:
        test_labels = labels['test_labels']
    # labels: [ BATCH_SIZE, SEQUENCE_LENGTH, DIMENSION ]

    # Extract Features
    train_features = features['train_features']
    if params['do_test']:
        test_features = features['test_features']

    # Build model
    Y, encoded_V = _lstmnet(train_features, train_labels, mode, params, is_test=False)
    # Y: [ BATCH_SIZE, SEQUENCE_LENGTHLEN, DIMENSION ]
    # encoded_V: [ BATCH_SIZE, BOTTLENECK_SIZE ]
    do_test = params['do_test']
    if do_test:
        test_Y, test_encoded_V = _lstmnet(test_features,
                                          test_labels,
                                          mode, params, is_test=True)
        # test_Y: [ BATCH_SIZE, SEQUENCE_LENGTHLEN, DIMENSION ]

    with tf.variable_scope('Prediction') as scope:

        # Create model again with batch size of 1
        # predicted_Y, predicted_V = _lstmnet(train_features, train_labels, mode, params, is_test=True)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'encoding': encoded_V,
                'decoding': Y,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.variable_scope('Metrics') as scope:

        # important training loss
        # Only use latest half of sequence for training
        # train_labels = train_labels[:,int(params['sequence_length']/2),:]
        # Y = Y[:,int(params['sequence_length']/2),:]
        square_error = tf.reduce_mean(tf.losses.mean_squared_error(train_features['sequence_values'], Y))

        # Again for proper metrics implementation
        train_square_error = metric_variable([], tf.float32, name='train_square_error')
        train_square_error_op = tf.assign(train_square_error, tf.reduce_mean(
            tf.losses.mean_squared_error(train_features['sequence_values'], Y)))

        if do_test:
            test_square_error = metric_variable([], tf.float32, name='test_square_error')
            test_square_error_op = tf.assign(test_square_error, tf.reduce_mean(
                tf.losses.mean_squared_error(test_features['sequence_values'], test_Y)))

    with tf.variable_scope('Training') as scope:
        scope.reuse_variables()

        learning_rate = metric_variable([], tf.float32, name='learning_rate')
        starter_learning_rate = params['learning_rate']
        learning_rate_ti = tf.train.inverse_time_decay(starter_learning_rate,
                                                       tf.train.get_global_step(),
                                                       params['decay_steps'],
                                                       params['decay_rate'])
        learning_rate_ex = tf.train.exponential_decay(starter_learning_rate,
                                                      tf.train.get_global_step(),
                                                      params['decay_steps'],
                                                      params['decay_rate'])
        learning_rate_op = tf.assign(learning_rate, learning_rate_ex, name='learning_rate')

    with tf.variable_scope('Evaluation') as scope:

        metrics = {
            'train_square_error': (train_square_error_op, train_square_error_op),
            'learning_rate': (learning_rate_op, learning_rate_op),
        }
        tf.summary.scalar('train_square_error', train_square_error_op)
        tf.summary.scalar('learning_rate', learning_rate_op)

        if do_test:
            test_metrics = {
                'test_square_error': (test_square_error_op, test_square_error_op),
            }
            tf.summary.scalar('test_square_error', test_square_error_op)

            metrics = {**metrics, **test_metrics}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=test_square_error_op, eval_metric_ops=metrics)

    with tf.variable_scope('Training') as scope:

        trainables = tf.trainable_variables()
        gradients = tf.gradients(square_error, trainables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])

        optimizer_RMS = tf.train.RMSPropOptimizer(
            learning_rate=params['learning_rate'], decay=params['decay_rate'])
        optimizer_adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = optimizer_RMS

        # train_op = optimizer.apply_gradients(zip(clipped_gradients, trainables), global_step=tf.train.get_global_step())
        train_op = optimizer.minimize(square_error, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=square_error, train_op=train_op)
