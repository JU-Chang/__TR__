import time
import tensorflow as tf
import convert_to_tfrecords as ctt
import tourism_lstm as tl
import numpy as np


def evaluation(inputs_placeholder, rows_placeholder, labels_placeholder, test_step, data_type, data_sets, eval_correct, sess):

    # # Extract data from tfrecords
    #
    # eval_images, eval_labels, eval_rows = ctt.inputs(data_set_type=data_type,
    #                                                  batch_size=tl.FLAGS.batch_size,
    #                                                  num_epochs=tl.FLAGS.num_epochs)

    true_count = 0  # Counts the number of correct predictions.
    num_examples = 0
    for step in range(test_step):

        if data_type == 'validation':
            feature_data, labels_data, sequence_len = sess.run(data_sets[1])
        else:
            feature_data, labels_data, sequence_len = sess.run(data_sets[2])

        labels_data = tl.one_hot(labels_data, sequence_len)
        eval_correct_ = sess.run(eval_correct,
                                 feed_dict={inputs_placeholder: feature_data,
                                            rows_placeholder: sequence_len,
                                            labels_placeholder: labels_data})
        eval_correct_ -= (max(sequence_len) * tl.FLAGS.batch_size-np.sum(sequence_len))
        true_count += eval_correct_
        num_examples += np.sum(sequence_len)
    precision = float(true_count) / num_examples

    # Print status to stdout.
    # print('  Num correct: %d ' % true_count)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %f ' % (num_examples, true_count, precision))


# def evaluation(data_type):
#
#     with tf.Graph().as_default():
#
#         # Extract data from tfrecords
#
#         images, labels, rows = ctt.inputs(data_set_type=data_type,
#                                           batch_size=tl.FLAGS.batch_size,
#                                           num_epochs=tl.FLAGS.num_epochs)
#         images = tf.sparse_to_dense(images.indices, images.shape, images.values)
#
#         inputs_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size, None, ctt.FLAGS.feature_col])
#         rows_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size])
#         labels_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size, None, tl.num_class])
#
#         eval_correct = do_eval_batch(inputs_placeholder,
#                                      labels_placeholder,
#                                      rows_placeholder)
#
#         # summary = tf.merge_all_summaries()
#
#         # The op for initializing the variables.
#         init_op = tf.group(tf.initialize_all_variables(),
#                            tf.initialize_local_variables())
#
#         sess = tf.Session()
#
#         # Initialize the variables (the trained variables and the
#         # epoch counter).
#
#         # Instantiate a SummaryWriter to output summaries and the Graph.
#         # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
#
#         sess.run(init_op)
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         try:
#             step = 0
#             true_count = 0  # Counts the number of correct predictions.
#             num_examples = 0
#             while not coord.should_stop():
#
#                 feature_data, labels_data, sequence_len = sess.run([images, labels, rows])
#
#                 labels_data = tl.one_hot(labels_data, sequence_len)
#                 eval_correct_ = sess.run(eval_correct,
#                                          feed_dict={inputs_placeholder: feature_data,
#                                                     rows_placeholder: sequence_len,
#                                                     labels_placeholder: labels_data})
#                 eval_correct_ -= (max(sequence_len) * tl.FLAGS.batch_size-np.sum(sequence_len))
#                 true_count += eval_correct_
#                 num_examples += np.sum(sequence_len)
#                 precision = float(true_count) / num_examples
#
#                 step += 1
#
#         except tf.errors.OutOfRangeError:
#             # Print status to stdout.
#             # print('  Num correct: %d ' % true_count)
#             print('  Num examples: %d  Num correct: %d  Precision @ 1: %f ' % (num_examples, true_count, precision))
#             print('Done training for %d epochs, %d steps.' % (tl.FLAGS.num_epochs, step))
#         finally:
#             coord.request_stop()
#
#         coord.join(threads)
#         sess.close()
#
#
# def main(_):
#     evaluation('validation')
#
#
# if __name__ == '__main__':
#     tf.app.run()
