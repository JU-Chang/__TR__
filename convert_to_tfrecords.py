
import os
import tensorflow as tf
import numpy as np
import collections
import random


"""
attention: parameter feature_col is 4096 for CNN feature,
           while 512 for gist feature
"""
tf.app.flags.DEFINE_string('directory', '/home/chang/tourismData',
                           'Directory store data files and'
                           'write the converted result')
tf.app.flags.DEFINE_integer('feature_col', ,
                            'Column size of the feature.')
tf.app.flags.DEFINE_string('train_dir', '/home/chang/tourismData',
                           'Directory store data files and'
                           'write the converted result')
FLAGS = tf.app.flags.FLAGS


Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_data(dataset_type='train'):
    """
    load *.npy data
    :param dataset_type: ['train','valid','test']
    :return: a collection:Dataset(data=data, target=target)
    """
    data = []
    target = []
    path = os.path.join(FLAGS.directory, dataset_type)
    label_lst = os.listdir(path)
    # label_lst = map(int, label_lst)
    for label in label_lst:
        path_of_feature = os.path.join(path, label)
        file_lst = os.listdir(path_of_feature)
        for filename in file_lst:
            one_example_feature = np.load(os.path.join(path_of_feature, filename))
            data.append(one_example_feature)
            target.append(int(label)-1)
    if not len(data) == len(target):
        raise ValueError('feature size %d does not match label size %d.' % (len(data), len(target)))
    perm = range(len(data))
    random.shuffle(perm)
    data = [data[i] for i in perm]
    target = [target[i] for i in perm]
    dataset = Dataset(data=data, target=target)
    return dataset


def load_datasets():
    test_set = load_data('test')
    training_set = load_data('train')
    validation_set = load_data('valid')
    datasets = Datasets(train=training_set, validation=validation_set, test=test_set)
    return datasets


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecords(data_set, name):
    """
    a function to generate the tfrecords data file
    :param data_set: data to convert,it should be a collection of Dataset(data=data, target=target)
    :param name: the filename of the generated tfrecords file
    """
    feature = data_set.data
    labels = data_set.target

    # file_num = 0
    # filename = os.path.join(FLAGS.directory, name + str(file_num) + '.tfrecords')
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(len(feature)):
        # if index > 0 and index % (len(feature) / 4) == 0:
        #     writer.close()
        #     file_num += 1
        #     filename = os.path.join(FLAGS.directory, name + str(file_num) + '.tfrecords')
        #     writer = tf.python_io.TFRecordWriter(filename)

        one_feature = feature[index].reshape(feature[index].size).tolist()
        rows = feature[index].shape[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'rows': _int64_feature(rows),
            'label': _int64_feature(labels[index]),
            'one_feature': _float_feature(one_feature)}))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename_queue):
    """
    decode tfrecords
    :param filename_queue: the filename
    :return: one_feature, label, rows(image number of one scene spot, that is, rows of the data of a .npy file)
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'rows': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'one_feature': tf.VarLenFeature(tf.float32)})

    one_feature = features['one_feature']
    # The code below does not work, you just cannot cast a SparseTensor
    # one_feature = tf.cast(features['one_feature'], tf.float32)

    """
    for CNN feature: n*4096
    """
    # one_feature = tf.sparse_reshape(one_feature, [-1, 4096])

    """
    for gist feature: n*512
    """
    one_feature = tf.sparse_reshape(one_feature, [-1, FLAGS.feature_col])
    label = tf.cast(features['label'], tf.int32)
    rows = tf.cast(features['rows'], tf.int32)
    return one_feature, label, rows


def inputs(data_set_type, batch_size, num_epochs):
    """convert tourismData without queue and multi-thread.
    :param data_set_type: which dataset to convert('train' or 'test' or 'validation')
    :param batch_size: batch size
    :param num_epochs: number of epochs
    :return: two Tensor (labels,rows) and one SparseTensor (images)
    """
    if not num_epochs:
        num_epochs = None
    # filename = FLAGS.train_dir+'/'+data_set_type+'.tfrecords'
    filename = []
    # for i in range(4):
    #     filename_i = os.path.join(FLAGS.train_dir,
    #                               data_set_type + str(i) + '.tfrecords')
    #     filename.append(filename_i)

    filename_i = os.path.join(FLAGS.train_dir, data_set_type + '.tfrecords')
    filename.append(filename_i)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filename, num_epochs=num_epochs)

    # decode datasets from tfrecords
    images, labels, rows = read_and_decode(filename_queue)

    # generate batches

    # shuffle_batch cannot be used here for the parameter images is a SparseTensor, while
    # shuffle_batch only works when the parameter are all Tensors
    # images, labels, rows = tf.train.shuffle_batch(
    #     [images, labels, rows], batch_size=batch_size, num_threads=2,
    #     capacity=1000 + 3 * batch_size,
    #     min_after_dequeue=1000)

    images, labels, rows = tf.train.batch(
         tensors=[images, labels, rows],
         batch_size=batch_size,
         dynamic_pad=True,
         name='data_batch'
     )
    images = tf.sparse_to_dense(images.indices, images.shape, images.values)
    return images, labels, rows


def main(argv):
    # Get the data.
    data_sets = load_datasets()

    # Convert to Examples and write the result to TFRecords.
    convert_to_tfrecords(data_sets.validation, 'validation')
    convert_to_tfrecords(data_sets.test, 'test')
    convert_to_tfrecords(data_sets.train, 'train')
    print('successfully')


if __name__ == '__main__':
    tf.app.run()
