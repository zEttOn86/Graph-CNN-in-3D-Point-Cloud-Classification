# coding;utf-8
import os, sys, time
import tensorflow as tf

from utils.layers.common_layers import fully_connected, global_pooling, graph_conv

class GraphCNN(object):
    def __init__(self, sess, output_dir, batch_size,
                num_sampling_point, num_feature, is_training=True, learning_rate=1e-4):
        self._sess = sess
        self._output_dir = output_dir
        self._learning_rate = learning_rate
        self._is_training = is_training
        self._batch_size = batch_size

        self._num_sampling_point = num_sampling_point
        self._num_feature = num_feature

        #
        self._output_class_n = 4
        self._gcn_1_filter_n = 1000
        self._gcn_2_filter_n = 1000
        self._fc_1_n = 600
        self._chebyshev_1_order = 4
        self._chebyshev_2_order = 3
        self._keep_prob_1 = 0.9 #0.9 original
        self._keep_prob_2 = 0.55

        self._train_operation = self._build_graph()
        self.saver = tf.train.Saver(max_to_keep=None)
        init = tf.initializers.global_variables()
        self._sess.run(init)

    def _build_graph(self):
        input_feature = tf.placeholder(tf.float32, [None, self._num_sampling_point, self._num_feature])
        input_graph = tf.placeholder(tf.float32, [None, self._num_sampling_point * self._num_sampling_point])
        output_label = tf.placeholder(tf.float32, [None, self._output_class_n])

        scaled_laplacian = tf.reshape(input_graph, [-1, self._num_sampling_point, self._num_sampling_point])

        weights = tf.placeholder(tf.float32, [None])
        lr = tf.placeholder(tf.float32)
        keep_prob_1 = tf.placeholder(tf.float32)
        keep_prob_2 = tf.placeholder(tf.float32)

        # gcn layer 1
        gcn_1 = graph_conv(input_feature, scaled_laplacian,
                         point_number=self._num_sampling_point,
                         input_feature_n=self._num_feature,
                         output_feature_n=self._gcn_1_filter_n,
                         chebyshev_order=self._chebyshev_1_order)
        gcn_1_output = tf.nn.dropout(gcn_1, keep_prob=keep_prob_1)
        gcn_1_pooling = global_pooling(gcn_1_output)
        print("The output of the first gcn layer is {}".format(gcn_1_pooling))
        print(gcn_1_pooling)

        # gcn layer 2
        gcn_2 = graph_conv(gcn_1_output, scaled_laplacian,
            point_number=self._num_sampling_point,
            input_feature_n=self._gcn_1_filter_n,
            output_feature_n=self._gcn_2_filter_n,
            chebyshev_order=self._chebyshev_2_order)
        gcn_2_output = tf.nn.dropout(gcn_2, keep_prob=keep_prob_1)
        gcn_2_pooling = global_pooling(gcn_2_output)
        print("The output of the second gcn layer is {}".format(gcn_2_pooling))

        global_features = tf.concat([gcn_1_pooling, gcn_2_pooling], axis=1)
        global_features = tf.nn.dropout(global_features, keep_prob=keep_prob_2)
        print("The global feature is {}".format(global_features))
        global_feature_n = (self._gcn_1_filter_n + self._gcn_2_filter_n)*2

        # fully connected layer 1
        fc_layer_1 = fully_connected(global_features,
                           input_feature_n=global_feature_n,
                           output_feature_n=self._fc_1_n)
        fc_layer_1 = tf.nn.relu(fc_layer_1)
        fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=keep_prob_2)

        # fully connected layer 2
        fc_layer_2 = fully_connected(fc_layer_1,
                                    input_feature_n=self._fc_1_n,
                                    output_feature_n=self._output_class_n)
        print("The output of the second fc layer is {}".format(fc_layer_2))

        # Define loss
        predict_softMax = tf.nn.softmax(fc_layer_2)
        predict_labels = tf.argmax(predict_softMax, axis=1)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer_2, labels=output_label)
        loss = tf.multiply(loss, weights)
        loss = tf.reduce_mean(loss)

        vars = tf.trainable_variables()
        loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 8e-6  # best: 8 #last: 10
        loss_total = loss + loss_reg

        correct_prediction = tf.equal(predict_labels, tf.argmax(output_label, axis=1))
        acc = tf.cast(correct_prediction, tf.float32)
        acc = tf.reduce_mean(acc)

        train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_total)

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print('Total parameters number is {}'.format(total_parameters))

        train_operaion = {'train': train,
                            'loss_total':loss_total,
                            'loss': loss,
                            'acc': acc,
                            'loss_reg': loss_reg,
                            'input_feature': input_feature,
                            'input_graph': input_graph,
                            'output_label': output_label,
                            'weights': weights,
                            'predict_labels': predict_labels,
                            'keep_prob_1': keep_prob_1,
                            'keep_prob_2': keep_prob_2,
                            'lr': lr}

        return train_operaion

    def update(self, input_feature, input_graph, label, batch_weight):
        feed_dict = {self._train_operation['input_feature']:input_feature,
                     self._train_operation['input_graph']: input_graph,
                     self._train_operation['output_label']:label,
                     self._train_operation['weights']: batch_weight,
                     self._train_operation['lr']: self._learning_rate,
                     self._train_operation['keep_prob_1']: self._keep_prob_1,
                     self._train_operation['keep_prob_2']: self._keep_prob_2}

        opt, loss_train, acc_train, loss_reg = self._sess.run([self._train_operation['train'],
                        self._train_operation['loss_total'],
                        self._train_operation['acc'],
                        self._train_operation['loss_reg']],
                        feed_dict=feed_dict
                        )
        return loss_train, loss_reg, acc_train

    def save_model(self, index):
        save = self.saver.save(self._sess, os.path.join(self._output_dir, 'model_{}'.format(index)))
        return save
