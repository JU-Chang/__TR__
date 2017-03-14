# import os
# import tensorflow as tf
#
#
# def read_and_decode(filename_queue):
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(
#         serialized_example,
#         features={
#             'rows': tf.FixedLenFeature([], tf.int64),
#             'label': tf.FixedLenFeature([], tf.int64),
#             'one_feature': tf.VarLenFeature(tf.float32)})
#
#     one_feature = features['one_feature']
#     # The code below does not work, you just cannot cast a SparseTensor
#     # one_feature = tf.cast(features['one_feature'], tf.float32)
#     # one_feature = tf.sparse_reshape(one_feature, [-1, 4096])
#     label = tf.cast(features['label'], tf.int32)
#     rows = tf.cast(features['rows'], tf.int32)
#     return one_feature, label, rows
#
#
# filename = []
# for i in range(1):
#     filename_i = os.path.join('/home/chang/tourismData/',
#                               'test' + '.tfrecords')
#     filename.append(filename_i)
#
# with tf.name_scope('input'):
#     filename_queue_ = tf.train.string_input_producer(
#         filename)
#
# # decode datasets from tfrecords
# images, labels, rows_ = read_and_decode(filename_queue_)
#
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     while not coord.should_stop():
#         try:
#             i, l, r = sess.run([images, labels, rows_])
#             print i.shape
#         except tf.errors.OutOfRangeError:
#             print('Done')
#         finally:
#             coord.request_stop()
#
#     coord.join(threads)
#     sess.close()

import numpy as np

data = np.load('/home/chang/tourismData/train/1/feat_0.npy')
print data.shape
