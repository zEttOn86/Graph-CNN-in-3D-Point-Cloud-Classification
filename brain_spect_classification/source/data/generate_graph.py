# coding:utf-8
"""
@auther tzw
"""
import os, sys, time
import argparse
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree
import pickle

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import utils.ioFunctions as IO
from utils.sampling_method import uniform_subsampling
from utils.mathematical_functions import adjacency, scaled_laplacian

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_file', '-i', type=str, default= '',
                            help='Directory path you want to know ')
    parser.add_argument('--output_dir', '-o', type=str, default= '',
                        help='Directory to output the file path list')

    parser.add_argument('--random_seed', '-r', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_groups', '-ng', type=int, default=6,
                        help='Number of groups')
    parser.add_argument('--num_sampling_point', '-nsp', type=int, default=1024,
                        help='Number of sampling point')
    parser.add_argument('--nearest_neighbor_num', '-nnn', type=int, default=3,
                        help='Nearest neighbor number (k-NN k)')

    parser.add_argument('--root', '-R', type=str, default= '',
                            help='Directory path contained data file ')
    args = parser.parse_args()

    print('----- Read data -----')
    df = pd.read_csv(args.input_csv_file)

    num_sampling_point = args.num_sampling_point
    nearest_neighbor_num = args.nearest_neighbor_num
    output_dir = '{}/graph/_pn_{}_nn_{}'.format(args.output_dir, num_sampling_point, nearest_neighbor_num)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    IO.save_args(output_dir, args)

    for group in range(args.num_groups):
        print('----- Group: {}'.format(group))
        group_df = df[df['group']==group]
        scaledLaplacianDict = dict()
        intensityDict = dict()
        label_list = []

        for case_num, (case, label) in enumerate(zip(df['Case'], df['label'])):
            print('---- Case, Label: {}, {}'.format(case, label))
            filepath = os.path.join(args.root, case)
            data_df = IO.read_dat(filepath)[:-1] # Remove last row
            assert(len(data_df) == 15964)

            # Subsampling
            subsampling_data_df = uniform_subsampling(data_df, num_sampling_point, args.random_seed)
            assert(len(subsampling_data_df) == num_sampling_point)
            flattenIntensity = subsampling_data_df['intensity'].values.reshape(1,-1)
            label_list.append(label)

            # Generate graph
            point_coorfinates = subsampling_data_df[['x', 'y', 'z']].values
            tree = cKDTree(point_coorfinates)
            dd, ii = tree.query(point_coorfinates, k = nearest_neighbor_num)
            A = adjacency(dd, ii)
            scaledLaplacian = scaled_laplacian(A)
            flattenLaplacian = scaledLaplacian.tolil().reshape((1, num_sampling_point*num_sampling_point))
            if case_num == 0:
                batchFlattenLaplacian = flattenLaplacian
                batchFlattenIntensity = flattenIntensity
            else:
                batchFlattenLaplacian = scipy.sparse.vstack([batchFlattenLaplacian, flattenLaplacian])
                batchFlattenIntensity = np.vstack([batchFlattenIntensity, flattenIntensity])

        scaledLaplacianDict.update({group: batchFlattenLaplacian})
        intensityDict.update({group: batchFlattenIntensity})
        label_list = np.asarray(label_list, dtype=np.int8)

        with open('{}/grouped_graph_{}'.format(output_dir, group), 'wb') as handle:
            pickle.dump(batchFlattenLaplacian, handle)

        with open('{}/grouped_intensity_{}'.format(output_dir, group), 'wb') as handle:
            pickle.dump(batchFlattenIntensity, handle)

        with open('{}/grouped_label_{}'.format(output_dir, group), 'wb') as handle:
            pickle.dump(label_list, handle)

        print ("Saving the graph data group {} is done ".format(group))


if __name__ == '__main__':
    main()
