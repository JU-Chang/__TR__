
import tensorflow as tf
import convert_to_tfrecords as ctt
import tourism_lstm as tl


def do_eval(outputs, labels_placeholder, softmax_w, softmax_b):

    batch_correct = 0
    b = []
    for example in range(tl.FLAGS.batch_size):
        logits = tl.inference_of_one_example(outputs, example, softmax_w, softmax_b)

        # for situation: rows<max_time, the filter_mat keep the logits[0:rows, :] ,
        # but turn logits[rows+1:, :] to zero, and the label[rows+1:, :] is also zero-padding.
        filter_mat = tf.expand_dims(tf.reduce_sum(labels_placeholder[example, :, :], 1), 1)
        a = tf.mul(logits, filter_mat)
        logits_mean = tf.argmax(tf.reduce_mean(tf.mul(logits, filter_mat), 0), 0)
        b.append(a)
        label = tf.argmax(labels_placeholder[example, :, :], 1)[0]

        # a = tf.argmax(tf.mul(logits, filter_mat), 1)
        # b = tf.argmax(labels_placeholder[example, :, :], 1)

        correct = tf.equal(logits_mean, label)
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        batch_correct += eval_correct
    return batch_correct, correct, b, filter_mat


def evaluation(test_step, data_type, data_sets, sess):
    with tf.name_scope('evaluation'):
        # Extract data from tfrecords
        inputs_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size, None, ctt.FLAGS.feature_col])
        rows_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size])
        labels_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size, None, tl.num_class])
    with tf.variable_scope("output_layer", reuse=True):
        softmax_w = tf.get_variable(name='weights', shape=[tl.num_units, tl.num_class], dtype=tf.float32)
        softmax_b = tf.get_variable(name='biases', shape=[tl.num_class], initializer=tf.constant_initializer(0.0))

    tf.get_variable_scope().reuse_variables()
    outputs = tl.lstm(inputs_placeholder, rows_placeholder, False)

    eval_correct, correct, b, filter_mat = do_eval(outputs, labels_placeholder, softmax_w, softmax_b)

    true_count = 0  # Counts the number of correct predictions.
    num_examples = 0
    for step in range(test_step):

        if data_type == 'validation':
            feature_data, labels_data, sequence_len = sess.run(data_sets[1])
        else:
            feature_data, labels_data, sequence_len = sess.run(data_sets[2])

        labels_data = tl.one_hot(labels_data, sequence_len)
        eval_correct_, c, b_, f = sess.run([eval_correct, correct, b, filter_mat],
                                 feed_dict={inputs_placeholder: feature_data,
                                            rows_placeholder: sequence_len,
                                            labels_placeholder: labels_data})

        # eval_correct_ -= (max(sequence_len) * tl.FLAGS.batch_size-np.sum(sequence_len))
        true_count += eval_correct_
        # num_examples += np.sum(sequence_len)
        num_examples += tl.FLAGS.batch_size
    precision = float(true_count) / num_examples

    # Print status to stdout.
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %f ' % (num_examples, true_count, precision))
