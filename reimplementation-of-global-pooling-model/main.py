# coding:utf-8
"""
For python3
"""
import os, sys, time
import numpy as np

from Parameters import Parameters
from read_data import load_data, prepareData

print('----- Read parameters -----')
start_time = time.time()
para = Parameters()
print('Dataset {}'.format(para.dataset))
print('')

print('----- Read dataset -----')
pointNumber = para.pointNumber
neighborNumber = para.neighborNumber
samplingType = para.samplingType
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if para.dataset == 'ModelNet40':
    inputTrain, trainLabel, inputTest, testLabel = load_data(pointNumber, samplingType, BASE_DIR)
elif para.dataset == 'ModelNet10':
    ModelNet10_dir = '/raid60/yingxue.zhang2/ICASSP_code/data/'
    with open(ModelNet10_dir+'input_data','rb') as handle:
        a = pickle.load(handle)
    inputTrain, trainLabel, inputTest, testLabel = a
else:
    print("Please enter a valid dataset")

scaledLaplacianTrain, scaledLaplacianTest = prepareData(inputTrain, inputTest,
                                                        neighborNumber, pointNumber, BASE_DIR, para.dataset)
