import tensorflow as tf
import numpy as np
import math

# Training

learning_rate = 0.0001
decay_rate = 0.9
decay_steps = 100000
pkeep = 0.7
summary_iters = 100

# Dataset

batch_size = 50
epoch_size = 10000
test_batch_size = 800
test_epoch_size = 2000
eval_seq_num = 12

# Now the input dimension of the LSTM for one cell without H
# So the dimension of your function's output
# This includes context information you may have passed
# Remember: An autoencoder/decoder learns to reconstruct its input
dimension = 1

# Training II

iters = 50000
seed = 1234

# Files

checkpoint_path = "./checkpoints/gru_function_autoencoder"
data_tmp_folder = "./data/records/gru_function_autoencoder/"
data_tfrecord_filename = "datamixed.tfrecord"
test_tfrecord_filename = "testmixed.tfrecord"

# Miscs

shuffle_capacity=500*batch_size
shuffle_threads=4
shuffle_min_after_dequeue=50*batch_size
parallel_iters=256

# Net Params

sequence_length = 50
encoder_hidden_layer_size = 20
encoder_hidden_layer_depth = 2
bottleneck_size = 15
decoder_hidden_layer_size = 20
decoder_hidden_layer_depth = 1
