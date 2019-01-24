# coding:utf-8
import os, sys, time
import numpy as np
from scipy.spatial import cKDTree

import utils

def load_data(NUM_POINT, sampleType, BASE_DIR):
    """
    return dict
    """
    TRAIN_FILES = utils.getDataFiles( \
        os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/train_files.txt'))
    TEST_FILES = utils.getDataFiles(\
        os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/test_files.txt'))

    if sampleType == 'farthest_sampling':
        inputTrainFarthest, inputTrainLabel = farthestSampling(TRAIN_FILES, NUM_POINT, BASE_DIR)
        inputTestFathest, inputTestLabel = farthestSampling(TEST_FILES, NUM_POINT, BASE_DIR)
        return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel

    elif sampleType == 'uniform_sampling':
        inputTrainFarthest, inputTrainLabel = uniformSampling(TRAIN_FILES, NUM_POINT, BASE_DIR)
        inputTestFathest, inputTestLabel = uniformSampling(TEST_FILES, NUM_POINT, BASE_DIR)

    return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel

def farthestSampling(file_names, NUM_POINT, base_dir):
    print('***** Farthest sampling *****')
    print('***** # sampling point {}'.format(NUM_POINT))
    file_indexs = np.arange(0, len(file_names))
    inputData = dict()
    inputLabel = dict()
    # Line loop
    for index in range (len(file_indexs)):
        current_data, current_label = utils.loadDataFile( \
                                        os.path.join(base_dir,
                                                '../', file_names[file_indexs[index]]))
        current_data = current_data[:,0:NUM_POINT,:] # shape = (420, 1024, 3)
        current_label = np.squeeze(current_label)
        current_label= np.int_(current_label)
        inputData.update({index : current_data}) # index ごとにarrayにぶち込む
        inputLabel.update({index : current_label})

    return inputData, inputLabel

def uniformSampling(file_names, NUM_POINT, base_dir):
    file_indexs = np.arange(0, len(file_names))
    inputData = dict()
    inputLabel = dict()
    for index in range (len(file_indexs)):
        current_data, current_label = utils.loadDataFile(\
                                        os.path.join(base_dir,
                                                '../', file_names[file_indexs[index]]))
        current_label = np.squeeze(current_label)
        current_label= np.int_(current_label)
        output = np.zeros((len(current_data), NUM_POINT, 3))
        for i,object_xyz in enumerate (current_data):
            samples_index=np.random.choice(2048, NUM_POINT, replace=False)
            output[i] = object_xyz[samples_index]
        inputData.update({index : output})
        inputLabel.update({index : current_label})
    return inputData, inputLabel

def prepareData(inputTrain, inputTest, neighborNumber, pointNumber, base_dir, dataset):
    scaledLaplacianTrain = prepareGraph(inputTrain, neighborNumber, pointNumber, 'train', base_dir, dataset)
    scaledLaplacianTest = prepareGraph(inputTest, neighborNumber, pointNumber, 'test', base_dir, dataset)
    return scaledLaplacianTrain, scaledLaplacianTest

##############################################
# Graph utilties
##############################################

#generate graph structure and store in the system
def prepareGraph(inputData, neighborNumber, pointNumber, dataType, base_dir, dataset):
    scaledLaplacianDict = dict()
    baseDir = base_dir
    #baseDir ='/raid60/yingxue.zhang2/ICASSP_code'
    #baseDir= os.path.abspath(os.path.dirname(os.getcwd()))
    if dataset == 'ModelNet40':
        fileDir =  baseDir+ '/graph/' + dataType+'_pn_'+str(pointNumber)+'_nn_'+str(neighborNumber)
    elif dataset == 'ModelNet10':
        fileDir =  baseDir+ '/graph_ModelNet10/' + dataType+'_pn_'+str(pointNumber)+'_nn_'+str(neighborNumber)
    else:
        print("Please enter a valid dataset")

    if (not os.path.isdir(fileDir)):
        print("calculating the graph data")
        os.makedirs(fileDir)
        # batchIndex: number of .h5 files
        for batchIndex in range(len(inputData)):
            batchInput = inputData[batchIndex]
            for i in range(len(batchInput)):
                print('Batch Index: {}, Case Num: {}'.format(batchIndex, i))
                pcCoordinates = batchInput[i] # shape = (1024, 3)
                tree = cKDTree(pcCoordinates)
                dd, ii = tree.query(pcCoordinates, k = neighborNumber) # shape = (1024, 40): (# sampling point, nearestNeighbor)
                A = utils.adjacency(dd, ii)
                scaledLaplacian = utils.scaled_laplacian(A)
                flattenLaplacian = scaledLaplacian.tolil().reshape((1, pointNumber*pointNumber))
                if i ==0:
                    batchFlattenLaplacian = flattenLaplacian
                else:
                    batchFlattenLaplacian = scipy.sparse.vstack([batchFlattenLaplacian, flattenLaplacian])
            scaledLaplacianDict.update({batchIndex: batchFlattenLaplacian})
            with open(fileDir+'/batchGraph_'+str(batchIndex), 'wb') as handle:
                pickle.dump(batchFlattenLaplacian, handle)
            print("Saving the graph data batch"+str(batchIndex))

    else:
        print("Loading the graph data from "+dataType+'Data')
        scaledLaplacianDict = loadGraph(inputData, neighborNumber, pointNumber, fileDir)
    return scaledLaplacianDict
