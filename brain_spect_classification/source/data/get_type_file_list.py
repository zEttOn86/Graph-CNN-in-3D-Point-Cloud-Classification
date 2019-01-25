# coding:utf-8
import os, sys, time
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv_file', '-i', type=str, default= '',
                        help='Directory path you want to know ')
parser.add_argument('--output_dir', '-o', type=str, default= '',
                    help='Directory to output the file path list')
args = parser.parse_args()

df = pd.read_csv(args.input_csv_file)

result_dir = args.output_dir

"""
Normalize method
0: nomark (no normalization)
1: THLZSFM
2: PNSZSFM
3: GLBZSFM
4: CBLZSFM
Ref. :
https://note.nkmk.me/python-pandas-str-contains-match/
"""
thlzsfm_binary_df = df['Case'].str.contains('THLZSFM')
pnszsfm_binary_df = df['Case'].str.contains('PNSZSFM')
glbzsfm_binary_df = df['Case'].str.contains('GLBZSFM')
cblzsfm_binary_df = df['Case'].str.contains('CBLZSFM')
nomark_binary_df =  ~(thlzsfm_binary_df | pnszsfm_binary_df | glbzsfm_binary_df | cblzsfm_binary_df)

thlzsfm_df = df[thlzsfm_binary_df]
thlzsfm_df.to_csv('{}/thlzsfm_list.csv'.format(result_dir), index=False, encoding='utf-8', mode='w')

pnszsfm_df = df[pnszsfm_binary_df]
pnszsfm_df.to_csv('{}/pnszsfm_list.csv'.format(result_dir), index=False, encoding='utf-8', mode='w')

glbzsfm_df = df[glbzsfm_binary_df]
glbzsfm_df.to_csv('{}/glbzsfm_list.csv'.format(result_dir), index=False, encoding='utf-8', mode='w')

cblzsfm_df = df[cblzsfm_binary_df]
cblzsfm_df.to_csv('{}/cblzsfm_list.csv'.format(result_dir), index=False, encoding='utf-8', mode='w')

nomark_df = df[nomark_binary_df]
nomark_df.to_csv('{}/nomark_list.csv'.format(result_dir), index=False, encoding='utf-8', mode='w')
