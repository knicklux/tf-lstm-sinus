import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

import config
import gen_data

def lstmnet_link(input_tensor, output_tensor, Hin, pkeep, phase, reuse_weights):
    # input_tensor: [ BATCH_SIZE, SEQUENCE_LENGTH, DIMENSION]
    # output_tensor: [ BATCH_SIZE, DIMENSION ]
    # Hin: [ BATCH_SIZE, INTERNALSIZE*NLAYERS ]

    with tf.variable_scope('NeuralNet', reuse=tf.AUTO_REUSE) as scope:
        if reuse_weights:
            scope.reuse_variables()

        X = tf.reshape(input_tensor, [config.batch_size, config.link_size, config.dimension])
        # X: [ BATCH_SIZE, LINK_SIZE, DIMENSION]

        cells = [rnn.GRUCell(config.hidden_layer_size) for _ in range(config.hidden_layer_depth)]
        # "naive dropout" implementation
        dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
        multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
        multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

        Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=Hin)
        H = tf.identity(H, name='H')  # just to give it a name
        # Yr: [ BATCH_SIZE, LINK_SIZE, INTERNALSIZE ]
        # H:  [ BATCH_SIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

        # Select last output.
        output = tf.transpose(Yr, [1, 0, 2])
        # output: [ LINK_SIZE, BATCH_SIZE, DIMENSION ]
        last = tf.gather(output, int(output.get_shape()[0])-1)
        # last: [ BATCH_SIZE, DIMENSION ]

        # Last layer to evaluate INTERNALSIZE LSTM output to logits
        # One-Hot-Encoding the answer using new API:
        Y = layers.fully_connected(last, config.dimension, activation_fn=None, reuse=reuse_weights, scope='NeuralNet')
        # Y: [ BATCH_SIZE, DIMENSION ]

    return H, Y

def lstmnet_chain(input_tensor, output_tensor, global_step, phase, reuse_weights):
    # input_tensor: [ BATCH_SIZE, SEQUENCE_LENGTH, DIMENSION ]
    # output_tensor: [ BATCH_SIZE, CHAIN_SIZE, DIMENSION ]
    # global_step: [ 1 ]
    # Preprocess Tensors in data generation and safe them correctly in TFRecords
    # file, otherwise, the GPU would need to reorder them every run
    # space/speed-Tradeoff in favor of speed.

    with tf.variable_scope('Input_Processing', reuse=tf.AUTO_REUSE) as scope:
        if reuse_weights:
            scope.reuse_variables()

        # Split Tensors
        input_tensors = tf.split(input_tensor, config.chain_size, 1)
        output_tensors = tf.split(output_tensor, config.chain_size, 1)

        # placeholder variables
        Hin = tf.placeholder(tf.float32, [config.batch_size, config.hidden_layer_size * config.hidden_layer_depth], name='Hin')
        # Hin: [ BATCH_SIZE, INTERNALSIZE*NLAYERS ]
        pkeep = tf.placeholder(tf.float32)
        # pkeep: [ 1 ]

    Ys = list()

    with tf.variable_scope('LSTM_CHAIN', reuse=tf.AUTO_REUSE) as scope:
        # First chain link is special
        prev_H, prev_Y = lstmnet_link(input_tensors[0], output_tensors[0], Hin, pkeep, phase, reuse_weights)
        Ys.append(prev_Y)

        for i in range(1, config.chain_size):
            prev_H, prev_Y = lstmnet_link(input_tensors[i], output_tensors[i], prev_H, pkeep, phase, True)
            Ys.append(prev_Y)

        Y = tf.stack(Ys, axis=1, name='Y')

    with tf.variable_scope('TrainingAndLoss', reuse=tf.AUTO_REUSE) as scope:
        if reuse_weights:
            scope.reuse_variables()

        y_ = output_tensor
        y_ = tf.identity(y_, name='y_')
        # [ BATCH_SIZE, DIMENSION ]

        starter_learning_rate = config.learning_rate
        learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, config.decay_steps, config.decay_rate)

        error = tf.reduce_mean(tf.losses.mean_squared_error(y_, Y))
        train_op = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate, decay=config.decay_rate).minimize(error)

    # accuracy
    with tf.name_scope('Summary') as scope:
        tf.summary.scalar(phase + "/error", error)
        summary_op = tf.summary.merge_all()

    return Hin, pkeep, train_op, summary_op
