# coding:utf-8
import os, sys, time
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import function

def global_pooling(x):
    """
    x: shape = (batchsize, N, in_channels)
    """
    mean = F.mean(x, axis = 1, keepdims=True)
    var = F.mean(F.square(x-mean), axis=1)
    max_f = F.max(x, axis=1)
    pooling_output = F.concat([max_f, var], axis=1)
    return pooling_output
