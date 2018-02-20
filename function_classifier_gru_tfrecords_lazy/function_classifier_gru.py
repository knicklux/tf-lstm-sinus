import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

import config
import gen_data


def lstmnet(input_tensor, label_tensor, global_step, phase, reuse_weights):
    # input_tensor: [ BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION]
    # label_tensor: [ BATCH_SIZE ]
    # global_step: [ 1 ]

    with tf.variable_scope('NeuralNet', reuse=tf.AUTO_REUSE) as scope:
        if reuse_weights:
            scope.reuse_variables()

        X = tf.reshape(input_tensor, [config.batch_size,
                                      config.sequence_length, config.input_dimension])
        # X: [ BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION]

        pkeep = tf.placeholder(tf.float32)

        Hin = tf.placeholder(
            tf.float32, [config.batch_size, config.hidden_layer_size * config.hidden_layer_depth], name='Hin')
        # Hin: [ BATCH_SIZE, INTERNALSIZE * NLAYERS]

        cells = [rnn.GRUBlockCell(config.hidden_layer_size)
                 for _ in range(config.hidden_layer_depth)]
        # "naive dropout" implementation
        dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
        multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
        # dropout for the softmax layer
        multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

        Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32,
                                  initial_state=Hin, parallel_iterations=config.batch_size)
        H = tf.identity(H, name='H')  # just to give it a name
        Yr_shaped = tf.reshape(
            Yr, [config.batch_size, config.sequence_length, config.hidden_layer_size])
        # Yr: [ BATCH_SIZE, SEQUENCE_LENGTHLEN, INTERNALSIZE ]
        # H:  [ BATCH_SIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
        Yr_lazy = Yr_shaped[:, config.lazy_cell_num:, :]
        # Yr_lazy: [ BATCH_SIZE, LABEL_LENGTH, INTERNALSIZE ]
        Yr_lazys = tf.split(Yr_lazy, config.label_length, axis=1)
        # Yr_lazys: [ LABEL_LENGTH ][ BATCH_SIZE, INTERNALSIZE ]

        # Append a fully connected layer after each non-lazy grucell output
        Ys = list()
        reuse = reuse_weights
        for Yl in Yr_lazys:
            Yl = tf.reshape(Yl, [config.batch_size, config.hidden_layer_size])

            with tf.variable_scope('NeuraNetFullyConnLayer', reuse=tf.AUTO_REUSE) as scope:
                if reuse:
                    scope.reuse_variables()
                Y = layers.fully_connected(Yl, config.output_dimension,
                                        activation_fn=None, reuse=reuse_weights, scope='NeuralNetFullyConnLayer')
            reuse = True
            Ys.append(Y)
        YLogits = tf.stack(Ys, axis=1, name='Ys')
        # YLogits: [ BATCH_SIZE, LABEL_LENGTH, OUTPUT_DIMENSION ]

    with tf.variable_scope('TrainingAndLoss', reuse=tf.AUTO_REUSE) as scope:
        if reuse_weights:
            scope.reuse_variables()

        starter_learning_rate = config.learning_rate
        learning_rate = tf.train.inverse_time_decay(
            starter_learning_rate, global_step, config.decay_steps, config.decay_rate)

        y_ = tf.reshape(label_tensor, [config.batch_size])
        # y_: [BATCH_SIZE] # int(s) identifying correct function
        # One-Hot encoode y_
        yo_ = tf.one_hot(y_, config.output_dimension, 1.0, 0.0)
        yos_ = tf.reshape(yo_, shape=[config.batch_size, 1, config.output_dimension])
        # yos_: [ BATCH_SIZE, config.output_dimension ]
        yot_ = tf.tile(yos_, [1, config.label_length, 1])
        # yot_: [ BATCHSIZE, LABEL_LENGTH, OUTPUT_DIMENSION ]
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=yot_, logits=YLogits))
        train_op = tf.train.RMSPropOptimizer(
            learning_rate=config.learning_rate, decay=config.decay_rate).minimize(cross_entropy)

    # accuracy
    with tf.name_scope('Summary') as scope:
        # select last output:
        output = tf.transpose(YLogits, [1, 0, 2])
        # output: [ SEEQLEN, BATCH_SIZE, config.output_dimension]
        Ylast = tf.gather(output, int(output.get_shape()[0])-1)
        # last: [ BATCH_SIZE , config.output_dimension]
        correct_prediction = tf.equal(tf.argmax(Ylast, 1), tf.argmax(yo_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar(phase + "/loss", cross_entropy)
        tf.summary.scalar(phase + "/acc", accuracy)
        summary_op = tf.summary.merge_all()

    return Hin, pkeep, train_op, summary_op
