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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

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

def gen_batches_from_datasequences(datasequences, batchnum, batchsize, sequence_length):
    # datasequences: [ OUTPUT_DIMENSION, TOTAL_LENGTH, INPUT_DIMENSION ]
    labelnum = datasequences.shape[0]
    dim = datasequences.shape[2]
    batches = np.ndarray(shape=(batchnum, batchsize, sequence_length, dim), dtype=np.float32)
    labels = np.ndarray(shape=(batchnum, batchsize), dtype=np.uint8)
    for i in range(batchnum):
        for j in range(batchsize):
            sel = np.random.randint(0, labelnum)
            seq = rand_seq_from_datasequence(datasequences[sel], sequence_length)
            batches[i,j,:,:] = seq
            labels[i,j] = sel
    # batches: [ BATCHNUM, BATCHSIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # labels: [ BATCHNUM, BATCHSIZE ]
    return batches, labels

def all_batches_from_datasequences(datasequences, batchnum, batchsize, sequence_length):
    # datasequences: [ OUTPUT_DIMENSION, TOTAL_LENGTH, INPUT_DIMENSION ]
    labelnum = datasequences.shape[0]
    dim = datasequences.shape[2]
    batches = np.ndarray(shape=(batchnum, batchsize, sequence_length, dim), dtype=np.float32)
    labels = np.ndarray(shape=(batchnum, batchsize), dtype=np.uint8)
    for i in range(batchnum):
        for j in range(batchsize):
            sel = np.random.randint(0, labelnum)
            seq = datasequences[sel,i*batchnum+j:i*batchnum+j+sequence_length,:]
            batches[i,j,:,:] = seq
            labels[i,j] = sel
    # batches: [ BATCHNUM, BATCHSIZE, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # labels: [ BATCHNUM, BATCHSIZE ]
    return batches, labels

def all_sequences_from_datasequence(datasequences, sequence_length):
    # datasequences: [ OUTPUT_DIMENSION, TOTAL_LENGTH, INPUT_DIMENSION ]
    labelnum = datasequences.shape[0]
    total_length = datasequences.shape[1]
    dim = datasequences.shape[2]
    seqnum = total_length - sequence_length
    total_seqnum = labelnum * seqnum
    sequences = np.ndarray(shape=(total_seqnum, sequence_length, dim), dtype=np.float32)
    labels = np.ndarray(shape=(total_seqnum), dtype=np.float32)

    # let's see, how good tf.train.shuffle is :^)
    for i in range(labelnum):
        for j in range(seqnum):
            seq = datasequences[i,j:j+sequence_length,:]
            labels[i*seqnum + j] = i
            sequences[i*seqnum + j, :,:] = seq

    # sequences: [ TOTAL_SEQUENCE_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # labels:    [ TOTAL_SEQUENCE_NUM ]
    return sequences, labels

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

def rand_sequences_from_datasequences(datasequences, labels, seqnum, delete=False):
    # datasequences: [ TOTAL_SEQUENCE_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    seqar = np.ndarray(shape=(seqnum, datasequences.shape[1], datasequences.shape[2]))
    newlabels = np.ndarray(shape=(seqnum))
    # seqar" [ SEQ_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    for i in range(seqnum):
        idx = np.random.randint(0, datasequences.shape[0]-i)
        seq = datasequences[idx]
        label = labels[idx]
        if(delete):
            np.delete(datasequences, idx, axis=0)
        seqar[i] = seq
        newlabels[i] = label

    # seqar" [ SEQ_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    return seqar, newlabels

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

def function_sequences_to_tfrecord(functionsequences, labels, filename):

    writer = tf.python_io.TFRecordWriter(filename)
    functionsequences = functionsequences.astype(np.float32)
    labels = labels.astype(np.uint8)

    # functionsequences: [ SEQUENCE_NUM, SEQUENCE_LENGTH, INPUT_DIMENSION ]
    # labels: [ SEQUENCE_NUM ]
    sequence_num = functionsequences.shape[0]
    sequence_length = functionsequences.shape[1]
    function_dimension = functionsequences.shape[2]

    for i in range(sequence_num):
        sequence = functionsequences[i]
        label = labels[i]
        seqraw = sequence.tostring()
        labelraw = label.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
        'seqlen': _int64_feature(sequence_length),
        'dim': _int64_feature(function_dimension),
        'seq_raw': _bytes_feature(seqraw),
        'label_raw': _bytes_feature(labelraw)}))

        writer.write(example.SerializeToString())

    writer.close()

def read_and_decode(filename_queue, batch_size, const_sequence_length, const_dim, capacity, num_threads, min_after_dequeue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'seqlen': tf.FixedLenFeature([], tf.int64),
        'dim': tf.FixedLenFeature([], tf.int64),
        'seq_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string)
        })

    # Reshape to proper dimensions
    seqlen = tf.cast(features['seqlen'], tf.int64)
    dim = tf.cast(features['dim'], tf.int64)
    sequence_bytes = tf.decode_raw(features['seq_raw'], tf.float32)
    label_bytes = tf.decode_raw(features['label_raw'], tf.uint8)

    sequence_shape = tf.stack([const_sequence_length, const_dim])
    label_shape = tf.stack([1])
    sequence_shape_const = tf.constant((const_sequence_length, const_dim), dtype=tf.float32)
    label_shape_const = tf.constant((1), dtype=tf.float32)
    sequence = tf.reshape(sequence_bytes, sequence_shape)
    label = tf.reshape(label_bytes, label_shape)

    # create and shuffle batch
    sequences, labels = tf.train.shuffle_batch([sequence, label], batch_size, capacity, num_threads, min_after_dequeue)
    sequences.set_shape([batch_size, const_sequence_length, const_dim])
    labels.set_shape([batch_size, 1])

    return sequences, labels
