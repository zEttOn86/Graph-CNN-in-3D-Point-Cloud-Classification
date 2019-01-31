#coding:utf-8
import os, sys, time, random
import argparse
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import scipy
import cupy as cp

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
from utils.layers.global_pooling import global_pooling
import utils.ioFunctions as IO

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', type=str, default= '',
                        help='Directory to input ')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

xp = np
if args.gpu >=0:
    xp = cp

random.seed(0)
np.random.seed(0)
if chainer.backends.cuda.available:
    chainer.backends.cuda.cupy.random.seed(0)

print('----- Generate input -----')
x = chainer.Variable(xp.array(xp.random.rand(2,3,5), dtype=xp.float32))
print(x.shape)
print(x)


print('----- apply -----')
y = global_pooling(x)
print(y.shape)
