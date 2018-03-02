import numpy as np
import tensorflow as tf

from tensorflow.contrib import seq2seq

# To train seq2seq RNNs, the encoder can be written as a straight forward
# dynamic RNN. However, the decoder is a bit more tricky. It needs to be fed
# with the initial_state (output state of encoder) and at each time step the
# its previous output. This can be easily achieved during training by using a
# TrainingHelper that uses a special input vector created from the training
# data for this sequential feed. However, this vector is not know during
# inference. There are many examples on how to do inference with discrete
# seq2seq models, however, there is no Helper that simply feeds the RNNs
# last output unmodified back to the RNN.

# Here are the functions needed to create a CustomTrainingHelper that simply
# feeds the RNNs last output unmodified back to the RNN.
# It may be useful to reqrite this as a class, but let's first try if it works.

# Dimensions:
# batch: [ BATCH_SIZE, SEQUENCE_LENGTH, DIMENSION ]
# initial_time_sample: [ BATCH_SIZE, DIMENSION ]
# sequence: [ SEQUENCE_LENGTH, DIMENSION ]
# sample: [ DIMENSION ]
# Dimensions for sample are taken from the inital_sample and input_batch

def batch_of_bools(val, batch_size):
    return tf.constant(val, shape=[batch_size, 1], dtype=tf.bool)

def fixed_len_initialize_fn(init_inputs, seqlen, batch_size):
    if seqlen > 1:
        finished = batch_of_bools(False, batch_size)
    else:
        finished = batch_of_bools(True, batch_size)
    print('Inital Inputs')
    print(init_inputs.shape)
    print('Finished:')
    print(finished)

    return finished, init_inputs

def fixed_len_sample_fn(time, outputs, state):
    print('outputs sample')
    print(outputs)
    return outputs

def fixed_len_next_inputs_fn(time, outputs, state, sample_ids, seqlen, batch_size):
    if time != seqlen:
        finished = batch_of_bools(False, batch_size)
    else:
        finished = batch_of_bools(True, batch_size)
    print('nextinputs')
    print(outputs)
    print('state')
    print(state)
    print('finished:')
    print(finished)
    return finished, outputs, state

def fixed_len_sample_ids_shape(batch_size, dimension):
    return tf.constant([batch_size, 1], dtype=tf.int32, name='sample_ids_shape')

def create_fixed_len_numeric_training_helper(initial_time_sample, sequence_length, dtype):
    batch_size = initial_time_sample.shape[0]
    dimension = initial_time_sample.shape[1]

    def initialize_fn():
        return fixed_len_initialize_fn(initial_time_sample, sequence_length, batch_size)
    def sample_fn(time, outputs, state):
        return fixed_len_sample_fn(time, outputs, state)
    def next_inputs_fn(time, outputs, state, sample_ids):
        return fixed_len_next_inputs_fn(time, outputs, state, sample_ids, sequence_length, batch_size)
    sample_ids_shape = fixed_len_sample_ids_shape(batch_size, dimension)
    sample_ids_dtype = dtype

    print(batch_size)
    print(sequence_length)
    print(initialize_fn)
    print(sample_fn)
    print(next_inputs_fn)
    print(sample_ids_shape)
    print(sample_ids_dtype)

    return seq2seq.CustomHelper(initialize_fn, sample_fn, next_inputs_fn)
