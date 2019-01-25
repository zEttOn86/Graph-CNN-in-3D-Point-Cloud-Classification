# coding:utf-8
import os, sys, time
import tensorflow as tf
import numpy as np
import pickle

from read_data import load_data, prepareData
from parameters import parameters
from model_multi_res import model_architecture, trainOneEpoch, evaluateOneEpoch
from utils import weight_dict_fc
from sklearn.metrics import confusion_matrix

print('----- Read parameters -----')
start_time = time.time()
para = parameters()

print('----- Build model -----')
trainOperaion, sess = model_architecture(para)

print('----- Load data -----')
pointNumber = para.pointNumber
neighborNumber = para.neighborNumber
samplingType = para.samplingType
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
inputTrain, trainLabel, inputTest, testLabel = load_data(pointNumber, samplingType, BASE_DIR)
scaledLaplacianTrain, scaledLaplacianTest = prepareData(inputTrain, inputTest, neighborNumber, pointNumber)

print('----- Train model -----')

saver = tf.train.Saver()
learningRate = para.learningRate

modelDir = para.modelDir
if not os.path.exists(modelDir):
    os.makedirs(modelDir)
save_model_path = modelDir + "model_" + para.fileName
weight_dict = weight_dict_fc(trainLabel, para)

testLabelWhole = []
for i in range(len(testLabel)):
    labels = testLabel[i]
    [testLabelWhole.append(j) for j in labels]
testLabelWhole = np.asarray(testLabelWhole)

test_acc_record = []
test_mean_acc_record = []

for epoch in range(para.max_epoch):
    print('===========================epoch {}===================='.format(epoch))
    if (epoch % 20 == 0):
        learningRate = learningRate / 2#1.7
    learningRate = np.max([learningRate, 1e-5])
    print(learningRate)
    #training step
    train_average_loss, train_average_acc, loss_reg_average = trainOneEpoch(inputTrain, scaledLaplacianTrain, trainLabel,
                                                                            para, sess, trainOperaion,
                                                                            weight_dict, learningRate)

    save = saver.save(sess, save_model_path)
    print('=============average loss, l2 loss, acc  for this epoch is {} {} and {}======'.format(train_average_loss,
                                                                                                 loss_reg_average,
                                                                                                 train_average_acc))
    #validating step
    eval_start_time = time.time()
    test_average_loss, test_average_acc, test_predict = evaluateOneEpoch(inputTest, scaledLaplacianTest,
                                                                         testLabel, para, sess, trainOperaion)
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    print("The forward inference time is {} second".format(eval_time))
    # calculate mean class accuracy
    test_predict = np.asarray(test_predict)
    test_predict = test_predict.flatten()
    confusion_mat = confusion_matrix(testLabelWhole[0:len(test_predict)], test_predict)
    normalized_confusion = confusion_mat.astype('float') / confusion_mat.sum(axis=1)
    class_acc = np.diag(normalized_confusion)
    mean_class_acc = np.mean(class_acc)

    # save log
    log_Dir = para.logDir
    if not os.path.exists(log_Dir):
        os.makedirs(log_Dir)
    fileName = para.fileName
    with open(log_Dir + 'confusion_mat_' + fileName, 'wb') as handle:
        pickle.dump(confusion_mat, handle)
    print('the average acc among 4 class is:{}'.format(mean_class_acc))
    print(
        '===========average loss and acc for this epoch is {} and {}======='.format(test_average_loss,
                                                                                    test_average_acc))
    test_acc_record.append(test_average_acc)
    test_mean_acc_record.append(mean_class_acc)

    with open(log_Dir + 'overall_acc_record_' + fileName, 'wb') as handle:
        pickle.dump(test_acc_record, handle)
    with open(log_Dir + 'mean_class_acc_record_' + fileName, 'wb') as handle:
        pickle.dump(test_mean_acc_record, handle)
