#coding:utf-8

import os, time, sys
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class GraphCnnUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gcnn = kwargs.pop('model')
        self.class_weights = kwargs.pop('class_weights')
        super(GraphCnnUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('gcnn')

        batch = self.get_iterator('main').next()
        label, coordinates, laplacian = self.converter(batch, self.device)

        gcnn = self.gcnn

        y = gcnn(coordinates, laplacian)

        loss = F.softmax_cross_entropy(y, label, class_weight=self.class_weights)
        gcnn.cleargrads()
        loss.backward()
        optimizer.update()
        acc = F.accuracy(F.softmax(y), label)
        chainer.reporter.report({'train/loss':loss, 'train/acc':acc})
