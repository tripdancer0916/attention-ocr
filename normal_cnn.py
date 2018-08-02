# !/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import pickle
import time
import argparse
import numpy as np
from chainer.training import extensions
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, training, serializers, utils, configuration
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Chainer example: CIFAR-10')
parser.add_argument('--gpu', '-gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

# GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


def unpickle(file):
    global data
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


train_loss = []
train_acc = []
test_loss = []
test_acc = []
N = 50000
N_test = 10000
batch_size = 100

x_train = None
y_train = []
for i in range(1, 6):
    data_dic = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
    if i == 1:
        x_train = data_dic['data']
    else:
        x_train = np.vstack((x_train, data_dic['data']))
    y_train += data_dic['labels']

test_data_dic = unpickle("cifar-10-batches-py/test_batch")
x_test = test_data_dic['data']
x_test = x_test.reshape(len(x_test), 3, 32, 32)
y_test = np.array(test_data_dic['labels'])
x_train = x_train.reshape((len(x_train), 3, 32, 32))
y_train = np.array(y_train)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

train = chainer.datasets.TupleDataset(x_train, y_train)
test = chainer.datasets.TupleDataset(x_test, y_test)

print(x_train.shape)
print(y_train.shape)


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


