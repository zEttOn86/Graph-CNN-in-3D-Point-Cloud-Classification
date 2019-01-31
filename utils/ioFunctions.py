# coding:utf-8
"""
@auther tzw
"""
import os, sys, time
import numpy as np
import pandas as pd
import yaml
import pickle
import h5py

def save_args(output_dir, args):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open('{}/config_{}.yml'.format(output_dir, time.strftime('%Y-%m-%d_%H-%M-%S')), 'w') as f:
        f.write(yaml.dump(vars(args), default_flow_style=False))

def read_data_list(path):
    return [line.rstrip() for line in open(path)]

def read_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)
