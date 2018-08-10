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
from seq2seq import Seq2seq, Decoder, AttentionModule
from utils import PAD, EOS
from utils import get_subsequence_before_eos


parser = argparse.ArgumentParser(description='Chainer example: CRNN-attention')
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
N = 2000
N_test = 3000
batch_size = 100


def str_to_label(x):
    CLASSES = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    label = []
    for char in x:
        # label.append(CLASSES.index(char))
        label.append(np.identity(len(CLASSES))[CLASSES.index(char)])
    return label


img_train = []
label_train = []
train_imgs = os.listdir('./IIIT5K/train')
for i in range(len(train_imgs)):
    if train_imgs[i][-1] == 'g':
        train_img = cv2.imread('./IIIT5K/train/' + train_imgs[i])
        train_img = cv2.resize(train_img, (128, 32))
        img_train.append(train_img)
        json_file_path = './IIIT5K/train/' + train_imgs[i][:-4] + '.json'
        with open(json_file_path, 'r') as f:
            answer_string = unicodedata.normalize('NFKC', json.load(f)[u'answer']).strip()
        label = str_to_label(answer_string)
        label_train.append(label)

img_test = []
label_test = []
test_imgs = os.listdir('./IIIT5K/test')
for i in range(len(test_imgs)):
    if test_imgs[i][-1] == 'g':
        test_img = cv2.imread('./IIIT5K/test/' + test_imgs[i])
        test_img = cv2.resize(test_img, (128, 32))
        img_test.append(test_img)
        json_file_path = './IIIT5K/test/' + test_imgs[i][:-4] + '.json'
        with open(json_file_path, 'r') as f:
            answer_string = unicodedata.normalize('NFKC', json.load(f)[u'answer']).strip()
        label = str_to_label(answer_string)
        label_test.append(label)

img_train = np.array(img_train)
img_test = np.array(img_test)
img_train = img_train / 255.
img_test = img_test / 255.
label_train = np.array(label_train)
label_test = np.array(label_test)
img_train = img_train.astype(np.float32)
img_train = img_train.transpose([0, 3, 1, 2])
img_test = img_test.astype(np.float32)
img_test = img_test.transpose([0, 3, 1, 2])

train = chainer.datasets.TupleDataset(img_train, label_train)
test = chainer.datasets.TupleDataset(img_test, label_test)

# print(label_train[0])


class CRNN(chainer.Chain):
    def __init__(self):
        super(CRNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 3, pad=1)
            self.conv2 = L.Convolution2D(32, 32, 3, pad=1)
            self.conv3 = L.Convolution2D(32, 32, 3, pad=1)
            self.conv4 = L.Convolution2D(32, 64, 3, pad=1)
            self.conv5 = L.Convolution2D(64, 128, 3, pad=1)
            self.conv6 = L.Convolution2D(128, 128, 3, pad=1)
            self.rnn = L.NStepBiGRU(2, in_size=512, out_size=512, dropout=0.2)
            self.embedding = L.Linear(512 * 2, 63)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        h = F.relu(self.conv5(h))
        conv_feature = F.max_pooling_2d(F.relu(self.conv6(h)), 2)
        conv_feature = F.transpose(conv_feature, axes=(0, 3, 1, 2))
        rnn_input = F.reshape(conv_feature, (batch_size, 16, 512))
        rnn_input = F.transpose(rnn_input, (1, 0, 2))
        xs = [rnn_input[i] for i in range(16)]
        _, rnn_output = self.rnn(None, xs)
        output = [self.embedding(rnn_output[i]) for i in range(16)]
        return output


class CRNNAttention(chainer.Chain):
    def __init__(self, n_target_vocab, n_decoder_units, n_attention_units, n_encoder_units, n_maxout_units):
        super(CRNNAttention, self).__init__()
        with self.init_scope():
            self.crnn = CRNN()
            self.decoder = Decoder(
                n_target_vocab,
                n_decoder_units,
                n_attention_units,
                n_encoder_units * 2,  # because of bi-directional lstm
                n_maxout_units,
            )

    def __call__(self, xs, ys):
        recurrent_output = self.crnn(xs)
        output = self.decoder(ys, recurrent_output)

        concatenated_os = F.concat(output, axis=0)
        concatenated_ys = F.flatten(ys.T)
        n_words = len(self.xp.where(concatenated_ys.data != PAD)[0])

        loss = F.sum(
            F.softmax_cross_entropy(
                concatenated_os, concatenated_ys, reduce='no', ignore_label=PAD
            )
        )
        loss = loss / n_words
        chainer.report({'loss': loss.data}, self)
        perp = self.xp.exp(loss.data * batch_size / n_words)
        chainer.report({'perp': perp}, self)

        return loss

    def translate(self, xs, max_length=100):
        """Generate sentences based on xs.

        Args:
            xs: Source sentences.

        Returns:
            ys: Generated target sentences.

        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            hxs = self.crnn(xs)
            ys = self.decoder.translate(hxs, max_length)
        return ys


model = CRNNAttention(63, 256, 256, 1024, 256)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

