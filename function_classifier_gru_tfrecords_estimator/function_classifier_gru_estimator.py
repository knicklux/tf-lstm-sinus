import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from tensorflow.python.ops.metrics_impl import metric_variable

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
# Hin
# sequence_length
# input_dimension
# hidden_layer_size
# hidden_layer_depth
# output_dimension
# learning_rate
# decay_rate
# decay_steps
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

    with tf.variable_scope('NeuralNet') as scope:
        if is_test:
            scope.reuse_variables()

        x = tf.feature_column.input_layer(
            features,
            feature_columns=params['feature_columns'])
        X = tf.reshape(x, shape=[x.get_shape()[0],
                                 params['sequence_length'], params['input_dimension']])
        # X: [ BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION]
        Hin = params['Hin']
        # Hin: [ BATCH_SIZE, INTERNALSIZE * NLAYERS]

        if(mode == tf.estimator.ModeKeys.TRAIN and not is_test):
            pkeep = params['pkeep']
        else:
            pkeep = 1.0

        cells = [rnn.GRUBlockCell(params['hidden_layer_size'])
                 for _ in range(params['hidden_layer_depth'])]
        # "naive dropout" implementation
        dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
        multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
        # dropout for the softmax layer
        multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

        Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32,
                                  initial_state=Hin, scope='NeuralNet',
                                  parallel_iterations=params['parallel_iters'])
        H = tf.identity(H, name='H')  # just to give it a name
        # Yr: [ BATCH_SIZE, SEQUENCE_LENGTHLEN, INTERNALSIZE ]
        # H:  [ BATCH_SIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

        # Select last output.
        output = tf.transpose(Yr, [1, 0, 2])
        # output: [ SEEQLEN, BATCH_SIZE, params.output_dimension]
        last = tf.gather(output, int(output.get_shape()[0])-1)
        # last: [ BATCH_SIZE , params.output_dimension]

        # Last layer to evaluate INTERNALSIZE LSTM output to logits
        # One-Hot-Encoding the answer using new API:
        YLogits = layers.fully_connected(
            last, params['output_dimension'], activation_fn=None)
        # YLogits: [ BATCH_SIZE, params.output_dimension ]

    return YLogits


def lstmnet(
        features,  # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params):

    train_features = features['train_features']
    train_labels = labels['train_labels']
    YLogits = _lstmnet(train_features, train_labels, mode, params, is_test=False)

    do_test = params['do_test']
    if do_test:
        test_features = features['test_features']
        test_labels = labels['test_labels']
        test_YLogits = _lstmnet(test_features,
                                test_labels,
                                mode, params, is_test=True)

    with tf.variable_scope('Prediction') as scope:

        predicted_classes = tf.argmax(YLogits, 1)
        # [ BATCH_SIZE ]

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(YLogits),
                'logits': YLogits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.variable_scope('Metrics') as scope:

        # important training loss
        yo_ = tf.one_hot(train_labels, params['output_dimension'], 1.0, 0.0)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=yo_, logits=YLogits))

        # Again for proper metrics implementation
        train_cross_entropy = metric_variable([], tf.float32, name='train_cross_entropy')
        train_cross_entropy_op = tf.assign(train_cross_entropy, tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=yo_, logits=YLogits)))

        # other nice-to-have metrics
        train_accuracy = tf.metrics.accuracy(
            labels=train_labels, predictions=predicted_classes, name='train_acc_')
        # returns an acc tensor and an update_op
        # correct_prediction = tf.equal(predicted_classes8, labels) why does this not work?
        train_batch_accuracy = metric_variable([], tf.float32, name='train_batch_accuracy')
        train_correct_prediction = tf.equal(tf.argmax(YLogits, 1), tf.argmax(yo_, 1))
        train_batch_accuracy_op = tf.assign(train_batch_accuracy, tf.reduce_mean(
            tf.cast(train_correct_prediction, tf.float32), name='train_batch_acc_'))

        if do_test:
            test_yo_ = tf.one_hot(test_labels, params['output_dimension'], 1.0, 0.0)
            test_predicted_classes = tf.argmax(test_YLogits, 1)
            test_cross_entropy = metric_variable([], tf.float32, name='test_cross_entropy')
            test_cross_entropy_op = tf.assign(test_cross_entropy, tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=test_yo_, logits=test_YLogits)))

            test_accuracy = tf.metrics.accuracy(
                labels=test_labels, predictions=test_predicted_classes, name='test_acc_')

            test_batch_accuracy = metric_variable([], tf.float32, name='test_batch_accuracy')
            test_correct_prediction = tf.equal(tf.argmax(test_YLogits, 1), tf.argmax(test_yo_, 1))
            test_batch_accuracy_op = tf.assign(test_batch_accuracy, tf.reduce_mean(
                tf.cast(test_correct_prediction, tf.float32), name='test_batch_acc_'))

    with tf.variable_scope('Training') as scope:

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
        learning_rate = tf.assign(learning_rate, learning_rate_ex, name='learning_rate')

    with tf.variable_scope('Evaluation') as scope:

        metrics = {'train_streamed_accuracy': train_accuracy,
                   'train_batch_accuracy': (train_batch_accuracy_op, train_batch_accuracy_op),
                   'train_loss': (train_cross_entropy_op, train_cross_entropy_op),
                   'learning_rate': (learning_rate, learning_rate),
                   }
        tf.summary.scalar('train_streamed_accuracy', train_accuracy[1])
        tf.summary.scalar('train_batch_accuracy', train_batch_accuracy_op)
        tf.summary.scalar('train_loss', train_cross_entropy_op)
        tf.summary.scalar('learning_rate', learning_rate)

        if do_test:
            test_metrics = {'test_streamed_accuracy': test_accuracy,
                            'test_batch_accuracy': (test_batch_accuracy_op, test_batch_accuracy_op),
                            'test_loss': (test_cross_entropy_op, test_cross_entropy_op),
                            }
            tf.summary.scalar('test_streamed_accuracy', test_accuracy[1])
            tf.summary.scalar('test_batch_accuracy', test_batch_accuracy_op)
            tf.summary.scalar('test_loss', test_cross_entropy_op)

            metrics = {**metrics, **test_metrics}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=cross_entropy, eval_metric_ops=metrics)

    with tf.variable_scope('Training') as scope:

        optimizer_RMS = tf.train.RMSPropOptimizer(
            learning_rate=params['learning_rate'], decay=params['decay_rate'])
        optimizer_adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = optimizer_adam
        train_op = optimizer.minimize(cross_entropy, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=cross_entropy, train_op=train_op)
