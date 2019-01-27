# coding:utf-8
import os, sys, time
import tensorflow as tf

def weight_variables(shape, name):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.05)
    return tf.Variable(initial, name=name)

def global_pooling(gcn_output):
    """
    """
    mean, var = tf.nn.moments(gcn_output, axes=[1])
    max_f = tf.reduce_max(gcn_output, axis=[1])
    pooling_output = tf.concat([max_f, var], axis=1)
    #return max_f
    return pooling_output

def fully_connected(features, input_feature_n, output_feature_n):
    """
    # fully connected layer without relu activation
    """
    weight_fc = weight_variables([input_feature_n, output_feature_n], name='weight_fc')
    bias_fc = weight_variables([output_feature_n], name='bias_fc')
    output_fc = tf.matmul(features, weight_fc) + bias_fc
    return output_fc

def chebyshev_coefficient(chebyshev_order, input_number, output_number):
    chebyshev_weights = dict()
    for i in range(chebyshev_order):
        initial = tf.truncated_normal(shape=[input_number, output_number], mean=0, stddev=0.05)
        chebyshev_weights['w_{}'.format(i)] = tf.Variable(initial)
    return chebyshev_weights

def graph_conv(input_p, scaled_laplacian, point_number,
                    input_feature_n, output_feature_n, chebyshev_order):
    bias_weight = weight_variables([output_feature_n], name='bias_w')
    chebyshev_coeff = chebyshev_coefficient(chebyshev_order, input_feature_n, output_feature_n)
    cheby_poly = []
    cheby_k_minus1 = tf.matmul(scaled_laplacian, input_p)
    cheby_k_minus2 = input_p

    cheby_poly.append(cheby_k_minus2)
    cheby_poly.append(cheby_k_minus1)
    for i in range(2, chebyshev_order):
        cheby_k = 2 * tf.matmul(scaled_laplacian, cheby_k_minus1) - cheby_k_minus2
        cheby_poly.append(cheby_k)
        cheby_k_minus2 = cheby_k_minus1
        cheby_k_minus1 = cheby_k

    cheby_output = []
    for i in range(chebyshev_order):
        weights = chebyshev_coeff['w_{}'.format(i)]
        cheby_poly_reshape = tf.reshape(cheby_poly[i], [-1, input_feature_n])
        output = tf.matmul(cheby_poly_reshape, weights)
        output = tf.reshape(output, [-1, point_number, output_feature_n])
        cheby_output.append(output)

    gcn_output = tf.add_n(cheby_output) + bias_weight
    gcn_output = tf.nn.relu(gcn_output)
    return gcn_output
