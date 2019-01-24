# coding:utf-8

class Parameters(object):
    def __init__(self):

        self.neighborNumber = 40
        self.pointNumber = 1024

        self.max_epoch = 260
        self.learningRate = 12e-4
        self.dataset = 'ModelNet40'
        self.samplingType = 'farthest_sampling'
