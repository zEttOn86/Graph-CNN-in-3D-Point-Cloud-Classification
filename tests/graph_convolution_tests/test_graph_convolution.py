#coding:utf-8
"""
https://groups.google.com/forum/#!topic/chainer-jp/QzprFJet2eo
"""
import os, sys, time
import argparse
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import scipy

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
from utils.layers.graph_convolution import GraphConvolution
import utils.ioFunctions as IO

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', type=str, default= '',
                        help='Directory to input ')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

print('----- Make model -----')
test = GraphConvolution(in_channels=1, out_channels=1000, chebyshev_order=4)
if args.gpu >= 0:
    chainer.backends.cuda.set_max_workspace_size(1024 * 1024 * 1024) # 1GB
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    test.to_gpu()
xp = test.xp
optimizer = chainer.optimizers.Adam()
optimizer.setup(test)

print('----- Read data -----')
batch_laplacian, batch_intensity, batch_label = IO.read_pickle_data(args.input_dir, group_num=0)
if not isinstance(batch_laplacian, scipy.sparse.coo.coo_matrix):
    raise NotImplementedError()
flatten_laplacian = batch_laplacian.tocsr()[:5].todense()
flatten_laplacian = np.array(flatten_laplacian).reshape(-1, 1024, 1024)
flatten_intensity = batch_intensity[:5].reshape(-1, 1024, 1)

flatten_laplacian = chainer.Variable(xp.array(flatten_laplacian, dtype=xp.float32))
flatten_intensity = chainer.Variable(xp.array(flatten_intensity, dtype=xp.float32))

# print(flatten_laplacian.shape)
# print(flatten_intensity.shape)
# print(batch_label)


print('----- Apply -----')
print('bias mean:{}'.format(test.bias.array.mean()))
print('bias min:{}'.format(test.bias.array.min()))
print('bias max:{}'.format(test.bias.array.max()))
print('chebyshev_coeff mean:{}'.format(test.chebyshev_coeff.array.mean()))
print('chebyshev_coeff min:{}'.format(test.chebyshev_coeff.array.min()))
print('chebyshev_coeff max:{}'.format(test.chebyshev_coeff.array.max()))
x = test(flatten_intensity, flatten_laplacian)

print('----- Calc loss -----')
t = xp.zeros(x.shape, dtype='f')
loss = F.mean_squared_error(x, t)
import chainer.computational_graph as c
# https://docs.chainer.org/en/stable/reference/graph.html
g = c.build_computational_graph([loss])
with open('{}/test_graph_conv.dot'.format(os.path.dirname(os.path.abspath(__file__))), 'w') as o:
    o.write(g.dump())
test.cleargrads()
loss.backward()
optimizer.update()
print('bias mean:{}'.format(test.bias.array.mean()))
print('bias min:{}'.format(test.bias.array.min()))
print('bias max:{}'.format(test.bias.array.max()))
print('chebyshev_coeff mean:{}'.format(test.chebyshev_coeff.array.mean()))
print('chebyshev_coeff min:{}'.format(test.chebyshev_coeff.array.min()))
print('chebyshev_coeff max:{}'.format(test.chebyshev_coeff.array.max()))
