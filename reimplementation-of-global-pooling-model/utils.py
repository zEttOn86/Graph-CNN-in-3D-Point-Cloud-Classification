# coding:utf-8
import os, sys, time
import h5py
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:] # e.g. shape = (420, 2048, 3) : (case, num_point, coordinate)
    label = f['label'][:] # e.g. shape = (420, 1)
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

##############################
# Mathematical utilties
##############################
def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape # #sampling point, kNN
    assert M, k == idx.shape
    assert dist.min() >= 0
    # Weights.
    sigma2 = np.mean(dist[:, -1]) ** 2
    # print(sigma2)
    dist = np.exp(- dist ** 2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k) # 各要素をk回繰り返す
    J = idx.reshape(M * k) # cols
    V = dist.reshape(M * k) # rows
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.coo_matrix.html
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W

def scaled_laplacian(adj): # future work
    adj_normalized = normalize_adj(adj)
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = scipy.sparse.linalg.eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian

def normalize_adj(adj):
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
