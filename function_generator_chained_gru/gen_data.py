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

def all_sequences_from_datasequence(datasequences, sequence_length, controlvals_num, link_size):
    # datasequences: [ TOTAL_LENGTH, DIMENSION ]
    # verify:
    if(link_size*controlvals_num != sequence_length):
        raise ValueError("link_size * chain_size must equal sequence_length!")

    sequence_length = sequence_length + 1
    total_length = datasequences.shape[0]
    dim = datasequences.shape[1]
    seqnum = total_length - sequence_length
    sequences = np.ndarray(shape=(seqnum, sequence_length, dim), dtype=np.float32)
    controlvals = np.ndarray(shape=(seqnum, controlvals_num, dim), dtype=np.float32)

    for i in range(seqnum):
        seq = datasequences[i:i+sequence_length,:]
        # seq: [ SEQUENCE_LENGTH, DIMENSION ]
        sequences[i,:,:] = seq
        # calculate controlvals
        for j in range(controlvals_num):
            controlvals[i,j,:] = seq[ (j+1)*link_size,:]

    # sequences: [ TOTAL_SEQUENCE_NUM, SEQUENCE_LENGTH, DIMENSION ]
    # controlvals:    [ TOTAL_SEQUENCE_NUM, CONTROLVALS_NUM, DIMENSION ]
    sequences = np.delete(sequences, sequence_length-1, axis=1)
    return sequences, controlvals

def rand_sequences_from_datasequences(datasequences, controlvals, seqnum, delete=False):
    # datasequences: [ TOTAL_SEQUENCE_NUM, SEQUENCE_LENGTH, DIMENSION ]
    # controlvals: [ TOTAL_SEQUENCE_NUM, CONTROLVALS_NUM, DIMENSION ]
    seqar = np.ndarray(shape=(seqnum, datasequences.shape[1], datasequences.shape[2]))
    newcontrolvals = np.ndarray(shape=(seqnum, controlvals.shape[1], controlvals.shape[2]))
    # seqar" [ SEQ_NUM, SEQUENCE_LENGTH, DIMENSION ]
    # newcontrolvals: [ SEQ_NUM, CONTROLVALS_NUM, DIMENSION]
    for i in range(seqnum):
        idx = np.random.randint(0, datasequences.shape[0]-i)
        seq = datasequences[idx]
        var = controlvals[idx]
        if(delete):
            np.delete(datasequences, idx, axis=0)
            np.delete(controlvals, idx, axis=0)
        seqar[i] = seq
        newcontrolvals[i] = var

    # seqar: [ SEQ_NUM, SEQUENCE_LENGTH, DIMENSION ]
    # controlvals: [ SEQ_NUM, CONTROLVALS_NUM, DIMENSION ]
    return seqar, newcontrolvals

# rewrite this with variable shape
def function_sequences_to_tfrecord(functionsequences, labels, filename):
    # functionsequences: [ SEQUENCE_NUM, SEQUENCE_LENGTH, DIMENSION ]
    # labels: [ SEQUENCE_NUM, CONTROLVALS_NUM, DIMENSION ]

    writer = tf.python_io.TFRecordWriter(filename)
    functionsequences = functionsequences.astype(np.float32)
    labels = labels.astype(np.float32)
    sequence_num = functionsequences.shape[0]
    sequence_length = functionsequences.shape[1]
    function_dimension = functionsequences.shape[2]
    labelnum = labels.shape[1]

    for i in range(sequence_num):
        sequence = functionsequences[i]
        label = labels[i]
        seqraw = sequence.tostring()
        labelraw = label.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
        'seqlen': _int64_feature(sequence_length),
        'dim': _int64_feature(function_dimension),
        'labelnum': _int64_feature(labelnum),
        'seq_raw': _bytes_feature(seqraw),
        'label_raw': _bytes_feature(labelraw)}))

        writer.write(example.SerializeToString())

    writer.close()

def _read_tf_record(record_filename, sequence_shape, label_shape, sequence_dtype, label_dtype):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(record_filename)

    data = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features = {
        'seqlen': tf.FixedLenFeature([], tf.int64),
        'dim': tf.FixedLenFeature([], tf.int64),
        'labelnum': tf.FixedLenFeature([], tf.int64),
        'seq_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string)
        })

        # Reshape to proper dimensions
    seqlen = tf.cast(data['seqlen'], tf.int64)
    dim = tf.cast(data['dim'], tf.int64)
    sequence_bytes = tf.decode_raw(data['seq_raw'], tf.float32)
    label_bytes = tf.decode_raw(data['label_raw'], tf.float32)

    sequence = tf.reshape(sequence_bytes, sequence_shape)
    label = tf.reshape(label_bytes, label_shape)

    return sequence, label


def read_and_decode(filename_queue, batch_size, const_sequence_length, const_control_vals, const_dim, capacity, num_threads, min_after_dequeue):

    #sequence_shape = tf.stack([const_sequence_length, const_dim], dtype=tf.int32)
    #label_shape = tf.stack([const_control_vals, const_dim], dtype=tf.int32)

    sequence_shape_const = tf.constant((const_sequence_length, const_dim), dtype=tf.int32)
    label_shape_const = tf.constant((const_control_vals, const_dim), dtype=tf.int32)
    sequences_shape_const = tf.constant((batch_size, const_sequence_length, const_dim), dtype=tf.int32)
    labels_shape_const = tf.constant((batch_size, const_control_vals, const_dim), dtype=tf.int32)

    sequence_dtype = tf.float32
    label_dtype = tf.float32

    readers = [_read_tf_record(filename_queue, sequence_shape_const, label_shape_const, sequence_dtype, label_dtype) for _ in range(num_threads)]

    # create and shuffle batch
    sequences, labels = tf.train.shuffle_batch_join(readers, batch_size, capacity, num_threads, min_after_dequeue)

    sequences_shape = tf.stack([batch_size, const_sequence_length, const_dim])
    labels_shape = tf.stack([batch_size, const_control_vals, const_dim])
    #sequences.set_shape(sequences_shape_const)
    #labels.set_shape(labels_shape_const)

    return sequences, labels
