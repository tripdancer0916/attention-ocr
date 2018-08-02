# !/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import pickle
import time
import argparse
import numpy as np
from chainer.training import extensions
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, training, serializers, utils, iterators
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import math
import matplotlib.pyplot as plt
import cv2


class ResBlock(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1, ksize=1):
        w = math.sqrt(2)
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_in, n_out, 3, stride, 1, False)
            self.bn1 = L.BatchNormalization(n_out)
            self.conv2 = L.Convolution2D(n_out, n_out, 3, 1, 1, False)
            self.bn2 = L.BatchNormalization(n_out)

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = self.bn2(self.conv2(h), test=not train)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=not train)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        return F.relu(h + x)


class ResNet(chainer.Chain):
    def __init__(self, block_class, n=18):
        super(ResNet, self).__init__()
        w = math.sqrt(2)
        links = [('conv1', L.Convolution2D(3, 16, 3, 1, 0))]
        links += [('bn1', L.BatchNormalization(16))]
        for i in range(n):
            links += [('res{}'.format(len(links)), block_class(16, 16))]
        for i in range(n):
            links += [('res{}'.format(len(links)),
                       block_class(32 if i > 0 else 16, 32,
                                   1 if i > 0 else 2))]
        for i in range(n):
            links += [('res{}'.format(len(links)),
                       block_class(64 if i > 0 else 32, 64,
                                   1 if i > 0 else 2))]
        links += [('_apool{}'.format(len(links)),
                   F.average_pooling_2d(6, 1, 0))]
        links += [('fc{}'.format(len(links)),
                   L.Linear(64, 10))]
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
        self.train = True

    def __call__(self, x, t):
        for name, f in self.forward:
            if 'res' in name:
                x = f(x, self.train)
            else:
                x = f(x)
        if self.train:
            self.loss = F.softmax_cross_entropy(x, t)
            self.accuracy = F.accuracy(x, t)
            return self.loss
        else:
            return x

