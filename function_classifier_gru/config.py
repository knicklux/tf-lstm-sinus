import tensorflow as tf
import numpy as np

# Training

learning_rate = 0.0001
decay = 0.9
batch_size = 100
iters = 50000
summary_iters = 50
epoch_size = 800
test_epoch_size = 50
checkpoint_path = "./checkpoints/gru_function_classifier/chkpt"
data_tmp_folder = "./data/records/gru_function_classifier/"
training_examples_number = 10000
validation_examples_number = 1000
labels = {'sine', 'linear'}

# Confusing: Output dimension of the LSTM
# So basically labels.length
output_dimension = len(labels)
# Now the input dimension of the LSTM for one cell without H
# So the dimension of your function's output
input_dimension = 1

sequence_length = 100
hidden_layer_size = 30
hidden_layer_depth = 2
pkeep = 0.5
