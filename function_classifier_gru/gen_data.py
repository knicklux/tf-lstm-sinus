import numpy as np

import tensorflow as tf
import pandas as pd

# Ideal: MNIST-like interface
# Declare data source
# And call get_next_train_batch

labels = {}

def set_string_from_label_index(index, string):
    global labels
    labels[index] = string

def get_string_from_label_index(index):
    global labels
    return labels[index]

# Generate function values from one input and store in CSV File

def gen_function_vals_csv(start, end, step, function, filename):
    # pd.Series(np.random.randn(5))
    y = []
    x = []
    t = float(start)
    while t < end :
        function(0)
        y.append(function(t))
        x.append(t)
        t = t + step

    xSer = pd.Series(x)
    ySer = pd.Series(y)

    d = {'x' : xSer, 'y' : ySer}
    df = pd.DataFrame(d)

    df.to_csv(filename)

# Read f(x) = y from CSV into multiple batches
# How to associate the label with this data?

def read_function_vals_csv(xlabel, ylabel, filename):
    xdf = pd.read_csv(filename, usecols={xlabel})
    ydf = pd.read_csv(filename, usecols={ylabel})
    xdf = xdf.fillna(value=0)
    ydf = ydf.fillna(value=0)
    dataY = ydf.values
    dataX = xdf.values
    return dataX, dataY

def rand_seq_from_datasequence(dataseq, seqlen):
    maxlen = dataseq.shape[0]-seqlen
    point = np.random.randint(0, maxlen)
    seq = dataseq[point:seqlen+point, :]
    return seq

def rand_sequences_from_datasequence(dataseq, seqlen, epochsize):
    seqar = np.ndarray(shape=(epochsize, seqlen, dataseq.shape[1]))
    # seqar: [EPOCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION]
    for index in range(0, epochsize):
        seq = rand_seq_from_datasequence(dataseq, seqlen)
        # Write seq to seqs at position i
        seqar[index,:,:] = seq
    return seqar

def rand_batch_from_dataepochs(dataepochs, batchsize):
    # dataepochs: [ EPOCH_SIZE, SEQUENCE_LENGTH, INPUT_DIMENSION, OUTPUT_DIMENSION ]
    epoch_size = dataepochs.shape[0]
    seqlen = dataepochs.shape[1]
    dim = dataepochs.shape[2]
    labelnum = dataepochs.shape[3]
    features = np.ndarray(shape=(batchsize, seqlen, dim), dtype=np.float32)
    labels = np.ndarray(shape = (batchsize), dtype=np.uint8)
    for index in range(0, batchsize):
        sel = np.random.randint(0, labelnum)
        seq = np.random.randint(0, epoch_size)
        features[index,:,:] = dataepochs[seq,:,:,sel]
        labels[index] = sel
    # features: [BATCHSIZE, Seqlen, Dimension]
    # labels: [BATCHSIZE]
    return features, labels
