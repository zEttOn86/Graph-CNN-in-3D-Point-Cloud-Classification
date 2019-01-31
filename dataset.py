# coding:utf-8
import os, sys, time
import numpy as np
import chainer
import pickle
import scipy
from scipy.spatial import cKDTree

import utils.ioFunctions as IO
from utils.mathematical_functions import adjacency, scaled_laplacian

class PointCloudDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, list_path, num_vertices, num_nearest_neighbor, xp, data_type='train', augmentation=False):
        """
        @root: input file directory
        @list_path: data file name list
        @num_vertices: # vertices
        @num_nearest_neighbor: # nearest neighbor
        """
        self.xp = xp
        self.augmentation = augmentation

        print('***** Read data *****')
        print('Note that this is farthest sampling')
        file_paths = IO.read_data_list(list_path)
        input_data = np.empty((0, num_vertices, 3))
        label_data = np.array([], dtype=np.int32)
        for index, file_path in enumerate(file_paths):
            current_data, current_label = IO.read_h5(os.path.join(root, file_path))
            current_data = current_data[:,0:num_vertices,:]
            current_label = np.squeeze(current_label)
            current_label = np.int_(current_label)
            input_data = np.append(input_data, current_data, axis=0)
            label_data = np.append(label_data, current_label)

        output_dir = '{}/data/graph/ModelNet40_{}_pn_{}_nn_{}'.format(root, data_type, num_vertices, num_nearest_neighbor)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print('::::: Calculate the graph :::::')
            start = time.time()
            for i, point_data in enumerate(input_data):
                print('::: Data number: {}'.format(i))
                tree = cKDTree(point_data)
                dd, ii = tree.query(point_data, k=num_nearest_neighbor)
                A = adjacency(dd, ii)
                laplacian = scaled_laplacian(A)
                flatten_laplacian = laplacian.tolil().reshape((1, num_vertices*num_vertices))
                if i==0:
                    batch_flatten_laplacian = flatten_laplacian
                else:
                    batch_flatten_laplacian = scipy.sparse.vstack([batch_flatten_laplacian, flatten_laplacian])

            with open('{}/flatten_laplacian_graph'.format(output_dir), 'wb') as handle:
                pickle.dump(batch_flatten_laplacian, handle)
            print(':: Calculation done: {:.3f} [s]'.format(time.time() - start))

        else:
            print('>>>>> Load the graph data >>>>>')
            with open('{}/flatten_laplacian_graph'.format(output_dir), 'rb') as handle:
                batch_flatten_laplacian = pickle.load(handle)

        self.laplacian_dataset = batch_flatten_laplacian
        self.coordinate_dataset = input_data
        self.label_dataset = label_data
        self.num_vertices = num_vertices
        self.class_weights = self.weight_fc(label_data)

    def __len__(self):
        return len(self.label_dataset)

    def add_noise(self, x, sigma=0.008, clip=0.02):
        N, in_ch = x.shape
        noise = np.clip(sigma*np.random.randn(N, in_ch), -1*clip, clip).astype(np.float32)
        return noise + x

    def weight_fc(self, train_labels):
        class_number = len(np.unique(train_labels))
        label_list = [i for i in train_labels]
        from sklearn.preprocessing import label_binarize
        labels = label_binarize(label_list, classes=[i for i in range(class_number)])
        class_distribution = np.sum(labels, axis=0)
        class_distribution = [float(i) for i in class_distribution]
        class_distribution /= np.sum(class_distribution)
        inverse_dist = 1 / class_distribution
        norm_inv_dist = inverse_dist / np.sum(inverse_dist)
        weights = norm_inv_dist * class_number + 1
        weights = self.xp.array(weights, dtype = self.xp.float32)
        return weights

    def get_example(self, i):
        laplacian = self.laplacian_dataset.tocsr()[i].todense()
        laplacian = np.array(laplacian, dtype=np.float32).reshape(self.num_vertices, self.num_vertices)
        coordinates = self.coordinate_dataset[i].astype(np.float32)
        label = self.label_dataset[i]
        if self.augmentation:
            return label, self.add_noise(coordinates), laplacian

        return label, coordinates, laplacian
