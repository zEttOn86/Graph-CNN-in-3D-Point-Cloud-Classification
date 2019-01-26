# coding;utf-8
import os, sys, time
import tensorflow as tf

from utils.layers.common_layers import fully_connected, global_pooling, graph_conv

def model(para):
    input_p = tf.placeholder(tf.float32, [None, para.point_num, para.feature_num])
    input_graph = tf.placeholder(tf.float32, [None, para.point_number * para.point_number])
    output_label = tf.placeholder(tf.float32, [None, para.output_class_n])

    scaled_laplacian = tf.reshape(input_graph, [-1, para.point_number, para.point_number])

    weights = tf.placeholder(tf.float32, [None])
    lr = tf.placeholder(tf.float32)
    keep_prob_1 = tf.placeholder(tf.float32)
    keep_prob_2 = tf.placeholder(tf.float32)

    # gcn layer 1
    gcn_1 = graph_conv(input_p, scaled_laplacian,
                     point_number=para.point_number,
                     input_feature_n=para.feature_num,
                     output_feature_n=para.gcn_1_filter_n,
                     chebyshev_order=para.chebyshev_1_order)
    gcn_1_output = tf.nn.dropout(gcn_1, keep_prob=keep_prob_1)
    gcn_1_pooling = global_pooling(gcn_1_output)
    print("The output of the first gcn layer is {}".format(gcn_1_pooling))
    print (gcn_1_pooling)

    # gcn_layer_2
    gcn_2 = graph_conv(gcn_1_output, scaledLaplacian,
                     point_number=para.pointNumber,
                     input_feature_n=para.gcn_1_filter_n,
                     output_feature_n=para.gcn_2_filter_n,
                     chebyshev_order=para.chebyshev_2_Order)
    gcn_2_output = tf.nn.dropout(gcn_2, keep_prob=keep_prob_1)
    gcn_2_pooling = global_pooling(gcn_2_output)
    print("The output of the second gcn layer is {}".format(gcn_2_pooling))

    # concatenate global features
    #globalFeatures = gcn_3_pooling
    global_features = tf.concat([gcn_1_pooling, gcn_2_pooling], axis=1)
    global_features = tf.nn.dropout(global_features, keep_prob=keep_prob_2)
    print("The global feature is {}".format(global_features))
    #globalFeatureN = para.gcn_2_filter_n*2
    global_feature_n = (para.gcn_1_filter_n + para.gcn_2_filter_n)*2

    # fully connected layer 1
    fc_layer_1 = fully_connected(global_features, input_feature_n=global_feature_n, output_feature_n=para.fc_1_n)
    fc_layer_1 = tf.nn.relu(fc_layer_1)
    fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=keep_prob_2)
    print("The output of the first fc layer is {}".format(fc_layer_1))

    # fully connected layer 2
    fc_layer_2 = fully_connected(fc_layer_1, input_feature_n=para.fc_1_n, output_feature_n=para.outputClassN)
    print("The output of the second fc layer is {}".format(fc_layer_2))

    return fc_layer_2
