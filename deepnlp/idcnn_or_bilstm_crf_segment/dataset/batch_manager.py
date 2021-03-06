# -*- coding:utf-8 -*-

import math
import random

class BatchManager(object):

    def __init__(self, data,  batch_size, sort=True):
        if sort:
            self.batch_data = self.sort_and_pad(data, batch_size)
        else:
            self.batch_data = self.pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    def pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        chars = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            char, target = line
            padding = [0] * (max_length - len(char))
            chars.append(char + padding)
            targets.append(target + padding)
        return [chars, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]