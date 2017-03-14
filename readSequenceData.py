# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import sys
reload( sys )

sys.setdefaultencoding('utf8')

# from struct import pack
#a = [1, 2, 3]
#b = [2, 3, 4]
#c = [3, 2, 1]
#sequences = [[a, b, c], [c, b]]
#label_sequences = [[0, 1, 0], [1, 0, 0]]
feature_path = '/deep/tourismDatasetClassification/tourismChosedTextfile/train/'

def create_example(sequences, labels):
    example = tf.train.SequenceExample()
    
    # A non-sequential feature of our example
    sequence_length = len(sequences)
    print(sequence_length)
    example.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = example.feature_lists.feature_list["tokens"]
    fl_labels = example.feature_lists.feature_list["labels"]
    for token, label in zip(sequences, labels):
        fl_tokens.feature.add().float_list.value.extend(token)
        fl_labels.feature.add().int64_list.value.append(label)    

    return example
    
def create_tfrecords(feature_path, save_name, type_string):
    # build writer
    file_name = feature_path + type_string + save_name
    if os.path.exists(file_name):
        print("Exists!")
    else:
        writer = tf.python_io.TFRecordWriter(file_name)
        
        # read sequences and label_sequences
        txt_name = type_string + 'List.txt'
        seq_lists = open(os.path.join(feature_path, txt_name))
        for seq_list in seq_lists:
            seq_list = seq_list.split()
            sequence = np.load(seq_list[0])
            sequence = sequence.tolist()
            seq_length = len(sequence)
            label_sequence = list([int(seq_list[1])]) * seq_length
            # print(sequence, label_sequence)
            print(seq_list[0], int(seq_list[1]))
            print(label_sequence)
            
            # build data for writing into tfrecords file
            example = create_example(sequence, label_sequence)
            writer.write(example.SerializeToString())
        writer.close()
    # return file_name
    return file_name
    
    
def decode_tfrecords(tfrecords_file):
    print(tfrecords_file)
    filename_queue = tf.train.string_input_producer([tfrecords_file], num_epochs = None)
   
    #  TFRecords reader
    reader = tf.TFRecordReader()
    
    # read filename into queue
    _, serialized_example = reader.read(filename_queue)
    
    # context feature
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }    
    
    # sequence feature and sequence label
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([4096], dtype = tf.float32),
        "labels" : tf.FixedLenSequenceFeature([], dtype = tf.int64)
    }    
    
    # tf.parse_single_sequence_example()
    # parse single sequence data
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized = serialized_example,
        context_features = context_features,
        sequence_features = sequence_features
    )
    print(type(context_parsed), type(sequence_parsed))
    print(sequence_parsed.get("tokens"), sequence_parsed.get("labels"))
    print(context_parsed.get("length"))
    return context_parsed, sequence_parsed
    
if __name__=='__main__':
     type_string = 'train'   
     file_name = '.tfrecords'
     tfrecords_path = create_tfrecords(feature_path, file_name, type_string)
     context, sequence = decode_tfrecords(tfrecords_path)
     print("OK!")
     # batch
     batch_data = tf.train.batch(
         tensors = sequence,
         batch_size = 60,
         dynamic_pad = True,
         name = 'seq_batch'
     )
#    # print("OK---------------------------------------------")
#     init = tf.initialize_all_variables()
#     
#     with tf.Session() as sess:
#         sess.run(init)      
#        # start queue
#         threads = tf.train.start_queue_runners(sess=sess)
#         for i in range(10):
#             val, l= sess.run([batch_data])
#             print(val.shape, l)
#     # res = tf.contrib.learn.run_n({"labels": sequence, "tokens": sequence}, n = 1, feed_dict = None)
     res = tf.contrib.learn.run_n({"sequence": batch_data}, feed_dict=None, 
                                  restore_checkpoint_path=None, n = 1)
     print("Batch shape: {}".format(res[0]["sequence"]))
     print(res[0]["sequence"])
