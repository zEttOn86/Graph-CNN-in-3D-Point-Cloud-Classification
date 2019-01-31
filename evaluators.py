#coding:utf-8
import os, time, sys, random
import argparse, yaml, shutil, math
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class GraphCnnEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, gcnn, class_weights,
                    converter=chainer.dataset.concat_examples,
                    device=None, eval_hook=None):
        if isinstance(iterator, chainer.dataset.Iterator):
            iterator = {"main":iterator}

        self._iterators = iterator
        self._class_weights = class_weights
        self._targets = {"gcnn" : gcnn}
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    # def calc_acc(self, prediction, gt_label):
    #     predict = F.softmax(prediction)
    #     predict_label = F.argmax(predict, axis=1)
    #     correct_prediction =

    def evaluate(self):
        iterator = self._iterators["main"]
        gen = self._targets["gcnn"]

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator) #shallow copy

        summary = chainer.reporter.DictSummary()

        for batch in it:
            observation ={}
            with chainer.reporter.report_scope(observation):
                label, intensity, laplacian = self.converter(batch, self.device)
                with chainer.using_config("train", False), chainer.using_config('enable_backprop', False):
                        y = gen(intensity, laplacian)
                loss = F.softmax_cross_entropy(y, label, class_weight=self._class_weights)
                acc = F.accuracy(F.softmax(y), label)

                observation["val/loss"] = loss
                observation['val/acc'] = acc

            summary.add(observation)

        return summary.compute_mean()
