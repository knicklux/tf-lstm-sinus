import tensorflow as tf
import numpy as np

# Training

learning_rate = 0.0001
decay_rate = 0.94
decay_steps = 100000
batch_size = 100
summary_iters = 50
epoch_size = 10000
batch_num = int(epoch_size/batch_size)
test_batches_size = 100
checkpoint_path = "./checkpoints/gru_function_classifier/"
data_tmp_folder = "./data/records/gru_function_classifier/"
labels = {'sine', 'linear'}

# Confusing: Output dimension of the LSTM
# So basically labels.length
output_dimension = len(labels)
# Now the input dimension of the LSTM for one cell without H
# So the dimension of your function's output
input_dimension = 1

sequence_length = 100
hidden_layer_size = 5
hidden_layer_depth = 2
pkeep = 1.0


iters = 1000*batch_num
