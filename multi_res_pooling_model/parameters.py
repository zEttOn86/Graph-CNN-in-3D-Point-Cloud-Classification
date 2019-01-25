# coding :utf-8
import os
class parameters():
    def __init__(self):
        #address
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.modelDir = BASE_DIR +'/multi_res_random/model/'
        self.logDir = BASE_DIR +'/multi_res_random/log/'
        self.fileName = '0221_40nn_cheby_2_2_w_55_52_multi_res'

        #fix parameters
        self.neighborNumber = 40
        self.pointNumber = 1024
        self.outputClassN = 40
        self.chebyshev_1_Order = 3
        self.chebyshev_2_Order = 3
        self.keep_prob_1 = 0.8
        self.keep_prob_2 = 0.55
        self.batchSize = 28
        self.testBatchSize = 1
        self.learningRate = 12e-4
        self.weight_scaler = 40  #40
	self.weighting_scheme = 'weighted'  #uniform weighted,uniform
        self.max_epoch = 210

        self.gcn_1_filter_n = 1000 # filter number of the first gcn layer
        self.gcn_2_filter_n = 1000 # filter number of the second gcn layer
        self.fc_1_n = 300 #fully connected layer dimension

        #multi res parameters
        self.clusterNumberL1 = 55 #layer one convolutional layer's cluster number
        self.nearestNeighborL1 =52 #nearest neighbor number of each centroid points when performing max pooling in first gcn

        self.clusterNumberL2 = 4  #layer two convolutional layer's cluster number
        self.nearestNeighborL2 = 10 ##nearest neighbor number of each centroid points when performing max pooling in second gcn layer
