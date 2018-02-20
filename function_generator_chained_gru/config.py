import tensorflow as tf
import numpy as np
import math

# Training

learning_rate = 0.005
decay_rate = 2
decay_steps = 100000
pkeep = 1.0
summary_iters = 1000

# Dataset

batch_size = 200
epoch_size = 10000
epochs = 10
test_batch_size = 200
test_epoch_size = 2000
test_epochs = int(math.ceil((epoch_size*epochs) / (test_epoch_size*summary_iters)))

# Now the input dimension of the LSTM for one cell without H
# So the dimension of your function's output
dimension = 2

# Training II

iters = 50000

# Files

checkpoint_path = "./checkpoints/gru_function_generator/"
data_tmp_folder = "./data/records/gru_function_generator/"
data_tfrecord_filename = "datamixed.tfrecord"
test_tfrecord_filename = "testmixed.tfrecord"

# Miscs

shuffle_capacity=500*batch_size
shuffle_threads=4
shuffle_min_after_dequeue=50*batch_size

# Net Params

sequence_length = 60
chain_size = 20
link_size = int(sequence_length/chain_size)
hidden_layer_size = 5
hidden_layer_depth = 1
