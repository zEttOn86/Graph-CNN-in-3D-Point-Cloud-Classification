#coding:utf-8
import os, sys, time
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

class GraphConvolution(chainer.Chain):
    def __init__(self, in_channels, out_channels, chebyshev_order,
                    nobias=False, initialW=None, initial_bias=None):
        """
        @in_channels: number of input channels
        @out_channels: number of output channels
        @chebyshev_order: chebyshev order

        input tensor shape = (batchsize, N (number of vertices), in_channels)
        laplacian shape = (batchsize, N, N)
        output tensor shape = (batchsize, N (number of vertices), out_channels)

        ref:
        https://github.com/meliketoy/graph-cnn.pytorch/blob/master/layers.py
        https://github.com/zEttOn86/Graph-CNN-in-3D-Point-Cloud-Classification/blob/master/global_pooling_model/layers.py
        https://github.com/musyoku/chainer-speech-recognition/blob/master/asr/nn/convolution_1d.py
        """
        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.initialW = initialW
        with self.init_scope():
            self.chebyshev_coeff = self.chebyshev_coefficient(chebyshev_order)

            if nobias:
                self.bias = None
            else:
                if initial_bias is None:
                    initial_bias = chainer.initializers.Normal(scale=0.05)

                initial_bias = chainer.initializers._get_initializer(initial_bias)
                self.bias = chainer.Parameter(initial_bias, out_channels)

    def chebyshev_coefficient(self, chebyshev_order):
        self.chebyshev_order = chebyshev_order

        initialw = chainer.initializers.Normal(scale=0.05)
        weight = chainer.Parameter(initializer=chainer.initializers._get_initializer(initialw),
                                    name='chebyshev_coeff_{}'.format(chebyshev_order))
        weight.initialize((chebyshev_order, self.in_channels, self.out_channels))

        return weight

    def forward(self, x, scaled_laplacian):
        """
        x: (batchsize, N, in_channels)
        scaled_laplacian: (batchsize, N, N)
        output: (batchsize, N, out_channels)
        """
        batchsize, N, _ = x.shape

        chebyshev_poly = [] # cheby_poly = (batchsize, N, in_channels)

        cheby_k_minus1 = F.matmul(scaled_laplacian, x) # (batchsize, N, in_channels)
        cheby_k_minus2 = x                             # (batchsize, N, in_channels)

        chebyshev_poly.append(cheby_k_minus2)
        chebyshev_poly.append(cheby_k_minus1)
        for i in range(2, self.chebyshev_order):
            cheby_k = 2 * F.matmul(scaled_laplacian, cheby_k_minus1) - cheby_k_minus2
            chebyshev_poly.append(cheby_k)
            cheby_k_minus2 = cheby_k_minus1
            cheby_k_minus1 = cheby_k

        # chebyshev loop
        for j, (chebyshev, cheby_weight) in enumerate(zip(chebyshev_poly, self.chebyshev_coeff)):
            chebyshev = F.reshape(chebyshev, (-1, self.in_channels))
            output = F.matmul(chebyshev, cheby_weight)
            output = F.reshape(output, (-1, N, self.out_channels))
            if j==0:
                y = output
            else:
                y = F.bias(y, output, axis=0)

        y = F.bias(y, self.bias, axis=2)
        y = F.relu(y)

        return y
