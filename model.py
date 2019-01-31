# coding;utf-8
"""
@auther tozawa
"""
import os, sys, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

from utils.layers.global_pooling import global_pooling
from utils.layers.graph_convolution import GraphConvolution

class GraphCNN(chainer.Chain):
    def __init__(self, in_channels=3, hidden_channels=1000, out_channels=40):
        """
        This model is based on https://github.com/maggie0106/Graph-CNN-in-3D-Point-Cloud-Classification
        """
        initializer = chainer.initializers.HeNormal()
        super(GraphCNN, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.dropout_ratio1 = 0.9
        self.dropout_ratio2 = 0.55
        with self.init_scope():
            self.gc1 = GraphConvolution(in_channels=in_channels, out_channels=hidden_channels, chebyshev_order=4)
            self.gc2 = GraphConvolution(in_channels=hidden_channels, out_channels=hidden_channels, chebyshev_order=3)
            self.fc1 = L.Linear(None, 600, initialW=initializer)
            self.fc2 = L.Linear(None, out_channels, initialW=initializer)

    def forward(self, x, laplacian):
        """
        x: shape = (batchsize, N, in_channels)
        laplacian: shape = (batchsize, N, N)
        """

        h1 = self.gc1(x, laplacian)
        h1 = F.dropout(h1, ratio=self.dropout_ratio1)
        h1_pool = global_pooling(h1)
        h2 = self.gc2(h1, laplacian)
        h2 = F.dropout(h2, ratio=self.dropout_ratio1)
        h2_pool = global_pooling(h2)
        h = F.dropout(F.concat([h1_pool, h2_pool], axis=1), ratio=self.dropout_ratio2)
        #del h1, h2, h1_pool, h2_pool
        h = F.dropout(F.relu(self.fc1(h)), ratio=self.dropout_ratio2)
        h = self.fc2(h)

        return h
