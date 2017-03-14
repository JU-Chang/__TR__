import time
import tensorflow as tf
import convert_to_tfrecords as ctt
import collections
import numpy as np
import math


tf.app.flags.DEFINE_integer('batch_size', 60,
                            'Batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
FLAGS = tf.app.flags.FLAGS


num_class = 156
num_units = 200
num_layers = 1
learning_rate = 0.001


# def do_eval(sess,
#             eval_correct,
#             inputs_placeholder,
#             rows_placeholder,
#             labels_placeholder,
#             data_sets,
#             data_type):
#     """Runs one evaluation against the full epoch of data.
#
#     Args:
#     sess: The session in which the model has been trained.
#     eval_correct: The Tensor that returns the number of correct predictions.
#     images_placeholder: The images placeholder.
#     labels_placeholder: The labels placeholder.
#     data_set: The set of images and labels to evaluate, from
#       input_data.read_data_sets().
#     """
#     # coord_eval = tf.train.Coordinator()
#     # threads = tf.train.start_queue_runners(sess=sess, coord=coord_eval)
#     #
#     # try:
#     step = 0
#     true_count = 0  # Counts the number of correct predictions.
#     #     while not coord_eval.should_stop():
#     for step in range(13):
#         feature_data, labels_data, sequence_len = sess.run(data_sets[data_type])
#         true_count += sess.run(eval_correct,
#                                feed_dict={inputs_placeholder: feature_data,
#                                           rows_placeholder: sequence_len,
#                                           labels_placeholder: labels_data})
#         step += 1
#     num_examples = (60*(step+1))
#     precision = float(true_count) / num_examples
#
#     # Print status to stdout.
#     print('  Num examples: %d  Num correct: %d  Precision @ 1: %f ' % (num_examples, true_count, precision))
#
#     # except tf.errors.OutOfRangeError:
#     #     print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
#     # finally:
#     #     # When done, ask the threads to stop.
#     #     coord_eval.request_stop()
#     # coord_eval.join(threads)


def run_training():

    with tf.Graph().as_default():

        # Extract data from tfrecords
        # data_type_placeholder = tf.placeholder("string")
        data_set_type = ['train', 'validation', 'test']
        data_sets = []
        for i in range(3):
            images, labels, rows = ctt.inputs(data_set_type=data_set_type[i],
                                              batch_size=FLAGS.batch_size,
                                              num_epochs=FLAGS.num_epochs)
            images = tf.sparse_to_dense(images.indices, images.shape, images.values)
            data_sets.append([images, labels, rows])

        # LSTM
        inputs_placeholder = tf.placeholder("float32", [FLAGS.batch_size, None, ctt.FLAGS.feature_col])
        rows_placeholder = tf.placeholder("float32", [FLAGS.batch_size])
        labels_placeholder = tf.placeholder("int32", [FLAGS.batch_size, None])
        max_time_placeholder = tf.placeholder("float32", [])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units,
                                                 forget_bias=1.0,
                                                 state_is_tuple=False)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=False)

        initial_state = cell.zero_state(FLAGS.batch_size, dtype=tf.float32)

        outputs, output_states = tf.nn.dynamic_rnn(cell=cell,
                                                   inputs=inputs_placeholder,
                                                   sequence_length=rows_placeholder,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   swap_memory=False,
                                                   time_major=False,
                                                   scope=None)

        # softmax_w , shape=[num_units, num_class]
        # softmax_w = tf.get_variable("softmax_w", [num_units, num_class], dtype=tf.float32)
        # softmax_w = tf.Variable(tf.truncated_normal([num_units, num_class],
        #                                             stddev=1.0 / math.sqrt(float(num_units))),
        #                         name='output_weights')
        softmax_w = tf.Variable(tf.zeros([num_units, num_class]), name='output_weights')
        softmax_b = tf.Variable(tf.zeros([num_class]), name='output_biases')
        # softmax_b = tf.get_variable("softmax_b", [num_class], dtype=tf.float32)

        # extra_thing
        padding_vec = tf.zeros([1, FLAGS.batch_size], dtype=tf.float32)
        padding_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(padding_vec, [0])

        one_example_loss = 0.0
        for batch in range(FLAGS.batch_size):
            # for batch in range(5):
            # output Layer
            logits = tf.matmul(outputs[batch, :, :], softmax_w) + softmax_b

            # Add to the Graph the loss calculation.
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, labels_placeholder[batch, :], name='xentropy')
            pad_loss_sum = padding_loss * (max_time_placeholder - rows_placeholder[batch]+1)
            one_example_loss += (tf.reduce_sum(cross_entropy, name='slot_loss') - pad_loss_sum) / rows_placeholder[batch]
        loss = one_example_loss / FLAGS.batch_size

        # logits = tf.matmul(outputs[:, -1, :], softmax_w) + softmax_b
        #
        # # Add to the Graph the loss calculation.
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits, labels_placeholder, name='xentropy')
        # loss = tf.reduce_mean(cross_entropy, name='slot_loss')

        # Add a scalar summary for the snapshot loss.
        # tf.scalar_summary(loss.op.name, loss)

        # Add to the Graph operations that train the model.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        # evaluation
        correct = tf.nn.in_top_k(logits, labels, 1)
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        # summary = tf.merge_all_summaries()

        # The op for initializing the variables.
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize the variables (the trained variables and the
        # epoch counter).

        # Instantiate a SummaryWriter to output summaries and the Graph.
        # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.

                # feature_data, labels_data, sequence_len = sess.run([images, labels, rows],
                #                                                    feed_dict={data_type_placeholder: 'train'})
                feature_data, labels_data, sequence_len = sess.run(data_sets[0])
                # labels_data = tf.matmul(labels_data, np.ones(max(sequence_len), dtype=tf.float64))
                labels_data = np.array(labels_data).reshape(1, FLAGS.batch_size)
                labels_data = labels_data.repeat(max(sequence_len), axis=0).transpose()
                for index in range(FLAGS.batch_size):
                    labels_data[index, sequence_len[index]:] = 0
                # _, loss_value, outputs_value, \
                # softmax_w_value = sess.run([train_op, loss, outputs, softmax_w],
                #                            feed_dict={inputs_placeholder: feature_data,
                #                                       rows_placeholder: sequence_len,
                #                                       labels_placeholder: labels_data})

                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={inputs_placeholder: feature_data,
                                                    rows_placeholder: sequence_len,
                                                    labels_placeholder: labels_data,
                                                    max_time_placeholder: max(sequence_len)})

                # data_sets_value = sess.run(data_sets)
                # _, loss_value = sess.run([train_op, loss],
                #                          feed_dict={inputs_placeholder: data_sets_value.data,
                #                                     rows_placeholder: data_sets_value.rows,
                #                                     labels_placeholder: data_sets_value.target})

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                # if (step % 10 == 0) or (step > 80):
                #  if step < 10:
                if step % 10 == 0:
                    # Print status to stdout.
                    # print labels_data, sequence_len
                    print('Step %d: loss = %.2f(%.3f sec)' % (step, loss_value, duration))
                    # print(outputs_value, softmax_w_value)
                    # summary_writer.add_summary(summary)
                    # summary_writer.flush()

                    # print('Validation Data Eval:')
                    # do_eval(sess,
                    #         eval_correct,
                    #         inputs_placeholder,
                    #         rows_placeholder,
                    #         labels_placeholder,
                    #         data_sets,
                    #         1)
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
