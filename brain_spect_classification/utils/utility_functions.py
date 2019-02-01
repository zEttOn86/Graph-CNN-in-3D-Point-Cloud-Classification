#codgin:utf-8
import os, sys, time
import tensorflow as tf
import numpy as np

def _parse_function(record, num_point):
    keys_to_features = {
        'label' : tf.FixedLenFeature([], tf.int64),
        'num_point' : tf.FixedLenFeature([], tf.int64),
        'laplacian' : tf.FixedLenFeature(num_point*num_point, tf.float32),
        'intensity' : tf.FixedLenFeature(num_point, tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    #label = tf.cast(parsed_features['label'], tf.int32)
    #label = tf.cast(parsed_features['label'], tf.float32)
    label = parsed_features['label']
    flatten_laplacian = parsed_features['laplacian']
    flatten_intensity = tf.reshape(parsed_features['intensity'], [num_point, 1])

    return label, flatten_laplacian, flatten_intensity

def gpu_config(index = "0"):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=index , # specify GPU number
            allow_growth=True
        )
    )
    return config

# calculate total parameters
def calc_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return print('Total params: %d ' % total_parameters)
