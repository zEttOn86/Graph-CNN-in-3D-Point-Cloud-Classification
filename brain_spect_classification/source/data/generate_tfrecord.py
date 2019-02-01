# coding:utf-8
import os, sys, time
import argparse
import tensorflow as tf
import glob
import pickle
import numpy as np
"""
https://knowledge.sakura.ad.jp/13152/
https://github.com/Silver-L/TFrecord_2/blob/master/make_tfrecord_ver2.py
"""
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', type=str, default= '',
                        help='Directory to input ')
parser.add_argument('--output_dir', '-o', type=str, default= '',
                    help='Directory to output the file path list')

parser.add_argument('--num_groups', '-ng', type=int, default=6,
                    help='Number of groups')
args = parser.parse_args()

for group in range(args.num_groups):
    print('----- Group: {}'.format(group))
    with open('{}/grouped_graph_{}'.format(args.input_dir, group), 'rb') as handle:
        batch_laplacian = pickle.load(handle)

    with open('{}/grouped_intensity_{}'.format(args.input_dir, group), 'rb') as handle:
        batch_intensity = pickle.load(handle)

    with open('{}/grouped_label_{}'.format(args.input_dir, group), 'rb') as handle:
        batch_label = pickle.load(handle)

    filename = '{}/record_file_{}'.format(args.output_dir, group)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(filename, options=options)

    for index in range(len(batch_label)):
        flatten_laplacian = batch_laplacian.tocsr()[index].todense()
        flatten_laplacian = np.array(flatten_laplacian).flatten()
        label = batch_label[index]
        flatten_intensity = batch_intensity[index]

        example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                            'num_point' : tf.train.Feature(int64_list=tf.train.Int64List(value=[len(flatten_intensity)])),
                            'laplacian' : tf.train.Feature(float_list=tf.train.FloatList(value=flatten_laplacian)),
                            'intensity' : tf.train.Feature(float_list=tf.train.FloatList(value=flatten_intensity)),
                        }
                    )
                )

        writer.write(example.SerializeToString())

    writer.close()
