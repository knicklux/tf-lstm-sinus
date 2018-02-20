import tensorflow as tf
import numpy as np
import math

# Training

learning_rate = 0.0001
decay_rate = 0.94
decay_steps = 100000
pkeep = 1.0
summary_iters = 1000

# Dataset

labels = {'sine', 'linear'}
batch_size = 200
epoch_size = 10000
epochs = 10
test_batch_size = 200
test_epoch_size = 2000
test_epochs = int(math.ceil((epoch_size*epochs) / (test_epoch_size*summary_iters)))

# Confusing: Output dimension of the LSTM
# So basically labels.length
output_dimension = len(labels)
# Now the input dimension of the LSTM for one cell without H
# So the dimension of your function's output
input_dimension = 1

# Training II

iters = 50000

# Files

checkpoint_path = "./checkpoints/gru_function_classifier/"
data_tmp_folder = "./data/records/gru_function_classifier/"
data_tfrecord_filename = "datamixed.tfrecord"
test_tfrecord_filename = "testmixed.tfrecord"

# Miscs

shuffle_capacity=500*batch_size
shuffle_threads=4
shuffle_min_after_dequeue=50*batch_size

# Net Params

sequence_length = 100
lazy_cell_num = 30
label_length = sequence_length - lazy_cell_num
hidden_layer_size = 5
hidden_layer_depth = 2
