# coding:utf-8
import os, sys, time
import h5py
import numpy as np

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
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W

def scaled_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = scipy.sparse.linalg.eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian
