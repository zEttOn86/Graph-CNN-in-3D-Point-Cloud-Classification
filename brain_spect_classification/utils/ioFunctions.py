# coding:utf-8
"""
@auther tzw
"""
import os, sys, time
import numpy as np
import pandas as pd
import yaml

def read_dat(path):
    """
    @path: file path
    @return: pandas dataframe
    """
    root, ext = os.path.splitext(path)
    if not ext == '.dat':
        raise NotImplementedError()

    df = pd.read_csv(path, names=('x', 'y', 'z','intensity'))

    return df

def save_args(output_dir, args):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open('{}/config_{}.yml'.format(output_dir, time.strftime('%Y-%m-%d_%H-%M-%S')), 'w') as f:
        f.write(yaml.dump(vars(args), default_flow_style=False))
