# coding:utf-8
import os, sys, time
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser(description='Get file list')
parser.add_argument('--input_dir', '-i', type=str, default= '',
                        help='Directory path you want to know ')
parser.add_argument('--output_dir', '-o', type=str, default= '',
                    help='Directory to output the file path list')
args = parser.parse_args()

'''http://staffblog.amelieff.jp/entry/2018/04/24/103006'''
path_list = [os.path.basename(i) for i in glob.glob("{}/**/*.dat".format(args.input_dir), recursive=True)]
#print(path_list)

df = pd.DataFrame({'Case': path_list})
df.to_csv('{}/file_list.csv'.format(args.output_dir), index=False, encoding='utf-8', mode='w')
