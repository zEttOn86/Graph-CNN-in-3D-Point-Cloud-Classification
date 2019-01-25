# coding:utf-8
"""
This program is to generate data group for 3 fold validation.
Label:
  A: 0
  D: 1
  F: 2
  N: 3

Procedure:
  1. Each label group is separated to 6 groups and then assigned.
  2. Each group is grouped into validation, test, train set.
  3.
"""
import os, sys, time
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv_file', '-i', type=str, default= '',
                        help='Directory path you want to know ')
parser.add_argument('--output_dir', '-o', type=str, default= '',
                    help='Directory to output the file path list')

parser.add_argument('--random_seed', '-r', type=int, default=0,
                    help='Random seed')
parser.add_argument('--num_groups', '-ng', type=int, default=6,
                    help='Number of groups')
args = parser.parse_args()

df = pd.read_csv(args.input_csv_file)

def grouping_each_label(dataframe, label='A', label_num=0, num_groups=6, random_seed=0):
    """
    return dataframe
    """
    label_df = dataframe[dataframe['Case'].str.startswith(label)]
    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    label_df = label_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    chunk_size = len(label_df) // num_groups
    groups = np.arange(0, num_groups).repeat(chunk_size)

    chunk_size = len(label_df) // num_groups
    groups = np.arange(0, num_groups).repeat(chunk_size)

    if not len(label_df) - len(groups) < 0:
        for i in range(len(label_df) - len(groups)):
            groups = np.append(groups, i)

    assert(len(groups) == len(label_df))

    label_df['group'] = groups
    label_df = label_df.sort_values(['group']).reset_index(drop=True)

    label_df['label'] = label_num

    return label_df

random_seed = args.random_seed
num_groups = args.num_groups

A_df = grouping_each_label(df, 'A', 0, num_groups=num_groups, random_seed=random_seed)
D_df = grouping_each_label(df, 'D', 1, num_groups=num_groups, random_seed=random_seed)
F_df = grouping_each_label(df, 'F', 2, num_groups=num_groups, random_seed=random_seed)
N_df = grouping_each_label(df, 'N', 3, num_groups=num_groups, random_seed=random_seed)

# print(A_df)
# print(D_df)
# print(F_df)
# print(N_df)

result_df = pd.concat([A_df, D_df, F_df, N_df], axis=0)
result_df = result_df.sort_values(['group']).reset_index(drop=True)
result_dir = args.output_dir
input_filename = os.path.basename(args.input_csv_file)
result_df.to_csv('{}/{}'.format(result_dir, input_filename), index=False, encoding='utf-8', mode='w')
