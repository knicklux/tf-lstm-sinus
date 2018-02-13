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

        X = tf.reshape(input_tensor, [config.batch_size, config.sequence_length, config.input_dimension])
        # X: [ BATCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION]

        pkeep = tf.placeholder(tf.float32)

        Hin = tf.placeholder(tf.float32, [config.batch_size, config.hidden_layer_size * config.hidden_layer_depth], name='Hin')
        # Hin: [ BATCH_SIZE, INTERNALSIZE * NLAYERS]

        # using a NLAYERS=2 layers of GRU cells, unrolled SEQLEN=30 times
        # dynamic_rnn infers SEQLEN from the size of the inputs Xo

        # How to properly apply dropout in RNNs: see README.md
        cells = [rnn.GRUCell(config.hidden_layer_size) for _ in range(config.hidden_layer_depth)]
        # "naive dropout" implementation
        dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
        multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
        multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

        Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=Hin)
        H = tf.identity(H, name='H')  # just to give it a name
        # Yr: [ BATCH_SIZE, SEQUENCE_LENGTHLEN, INTERNALSIZE ]
        # H:  [ BATCH_SIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

        # Select last output.
        output = tf.transpose(Yr, [1, 0, 2])
        # output: [ SEEQLEN, BATCH_SIZE, config.output_dimension]
        last = tf.gather(output, int(output.get_shape()[0])-1)
        # last: [ BATCH_SIZE , config.output_dimension]

        # Maybe useful line for prediction LSTM
        #Yflat = tf.reshape(Yr, [-1, self.hyper_params.arch.hidden_layer_size])    # [ BATCH_SIZE x SEQLEN, INTERNALSIZE ]

        # Last layer to evaluate INTERNALSIZE LSTM output to logits
        # One-Hot-Encoding the answer
        YLogits = layers.linear(last, config.output_dimension)
        # Using new API:
        # YLogits = layers.fully_connected(last, config.output_dimension, activation_fn=None, reuse=reuse_weights, scope='NeuralNet')
        # YLogits: [ BATCH_SIZE, config.output_dimension ]

    with tf.variable_scope('TrainingAndLoss', reuse=tf.AUTO_REUSE) as scope:
        if reuse_weights:
            scope.reuse_variables()

        starter_learning_rate = config.learning_rate
        learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, config.decay_steps, config.decay_rate)

        y_ = tf.reshape(label_tensor, [config.batch_size])
        # y_: [BATCH_SIZE] # int(s) identifying correct function.
        # One-Hot encoode y_
        yo_ = tf.one_hot(y_, config.output_dimension, 1.0, 0.0)
        # [BATCH_SIZE, config.output_dimension]
        # train_y_ = tf.reshape(train_y_, [-1, config.output_dimension])
        # [BATCH_SIZE, config.output_dimension]
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=YLogits, labels=yo_))
        # Using new API
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yo_, logits=YLogits))
        train_op = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate, decay=config.decay_rate).minimize(cross_entropy)

    # accuracy
    with tf.name_scope('Summary') as scope:
        correct_prediction = tf.equal(tf.argmax(YLogits, 1), tf.argmax(yo_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar(phase + "/loss", cross_entropy)
        tf.summary.scalar(phase + "/acc", accuracy)
        # tf.summary.scalar(phase + "/lr", learning_rate)
        summary_op = tf.summary.merge_all()

    return Hin, pkeep, train_op, summary_op
