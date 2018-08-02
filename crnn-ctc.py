# !/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import pickle
import time
import os
import argparse
import numpy as np
from chainer.training import extensions
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, training, serializers, utils, configuration
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import cv2
import unicodedata
import json

parser = argparse.ArgumentParser(description='Chainer example: CRNN-CTC')
parser.add_argument('--gpu', '-gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

# GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

train_loss = []
train_acc = []
test_loss = []
test_acc = []
N = 50000
N_test = 10000
batch_size = 100


def str_to_label(x):
    CLASSES = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    label = []
    for char in x:
        # label.append(np.identity(len(CLASSES))[CLASSES.index(char)])
        label.append(CLASSES.index(char))
    return label


img_train = []
label_train = []
train_imgs = os.listdir('./IIIT5K/train')
# print(train_imgs)
for i in range(len(train_imgs)):
    if train_imgs[i][-1] == 'g':
        train_img = cv2.imread('./IIIT5K/train/' + train_imgs[i])
        train_img = cv2.resize(train_img, (128, 32))
        img_train.append(train_img)
        json_file_path = './IIIT5K/train/' + train_imgs[i][:-4] + '.json'
        with open(json_file_path, 'r') as f:
            answer_string = unicodedata.normalize('NFKC', json.load(f)[u'answer']).strip()
        label = str_to_label(answer_string)
        # print(answer_string, label)
        label_train.append(label)

img_test = []
label_test = []
test_imgs = os.listdir('./IIIT5K/test')
# print(train_imgs)
for i in range(len(test_imgs)):
    if test_imgs[i][-1] == 'g':
        test_img = cv2.imread('./IIIT5K/test/' + test_imgs[i])
        test_img = cv2.resize(test_img, (128, 32))
        img_test.append(test_img)
        json_file_path = './IIIT5K/test/' + test_imgs[i][:-4] + '.json'
        with open(json_file_path, 'r') as f:
            answer_string = unicodedata.normalize('NFKC', json.load(f)[u'answer']).strip()
        label = str_to_label(answer_string)
        # print(answer_string, label)
        label_test.append(label)


img_train = np.array(img_train)
img_test = np.array(img_test)
# print(img_train)
img_train = img_train/255.
img_test = img_test/255.
label_train = np.array(label_train)
label_test = np.array(label_test)
img_train = img_train.astype(np.float32)
img_test = img_test.astype(np.float32)
print(label_train)
label_train = label_train.astype(np.int32)
label_test = label_test.astype(np.int32)

train = chainer.datasets.TupleDataset(img_train, label_train)
test = chainer.datasets.TupleDataset(img_test, label_test)

print(img_train.shape)
print(label_train.shape)

"""
# 畳み込み６層
class Cifar10Model(chainer.Chain):

    def __init__(self):
        super(Cifar10Model, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 3, pad=1)
            self.conv2 = L.Convolution2D(32, 32, 3, pad=1)
            self.conv3 = L.Convolution2D(32, 32, 3, pad=1)
            self.conv4 = L.Convolution2D(32, 32, 3, pad=1)
            self.conv5 = L.Convolution2D(32, 32, 3, pad=1)
            self.conv6 = L.Convolution2D(32, 32, 3, pad=1)
            self.l1 = L.Linear(512, 512)
            self.l2 = L.Linear(512, 10)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2)
        print(h.shape)
        h = F.dropout(F.relu(self.l1(h)))
        return self.l2(h)


model = Cifar10Model()
# model = L.Classifier(model)
# GPU使用のときはGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# model = L.Classifier(Cifar10Model())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

n_epochs = 10
batchsize = 100

for epoch in range(n_epochs):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x = x_train[perm[i:i + batchsize]]
        y = y_train[perm[i:i + batchsize]]
        if args.gpu >= 0:
            x = cuda.to_gpu(x)
            y = cuda.to_gpu(y)
        x = Variable(x)
        y = Variable(y)
        y_ = model(x)
        model.cleargrads()
        loss, acc = F.softmax_cross_entropy(y_, y), F.accuracy(y_, y)
        loss.backward()
        optimizer.update()
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
        del loss

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    with configuration.using_config('train', False):
        sum_accuracy = 0
        sum_loss = 0
        for i in range(0, N_test, batchsize):
            x = x_test[i:i + batchsize]
            y = y_test[i:i + batchsize]
            if args.gpu >= 0:
                x = cuda.to_gpu(x)
                y = cuda.to_gpu(y)
            y_ = model(x)
            loss, acc = F.softmax_cross_entropy(y_, y), F.accuracy(y_, y)
            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))
"""