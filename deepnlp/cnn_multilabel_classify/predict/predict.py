# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import codecs

from deepnlp.cnn_multilabel_classify import cnn_model, cnn_config
from deepnlp.cnn_multilabel_classify.dataset import rawdata


class ModelLoader(object):
    def __init__(self, data_path, ckpt_path):
        self.data_path = data_path
        self.ckpt_path = ckpt_path  # the path of the ckpt file, e.g. ./ckpt/zh/pos.ckpt
        print("Starting new Tensorflow session...")
        self.session = tf.Session()

        print("Starting new Tensorflow session...")
        self.config = cnn_config.CnnConfig()
        self.config.batch_size = 1
        self.word_idx_map = rawdata.get_idx_from_file(data_path)
        print("Initializing multi label classifer...")
        self.model = self._init_model(self.session, self.ckpt_path)

    def _init_model(self, session, ckpt_path):
        model = cnn_model.CNNModel(config=self.config)
        if tf.gfile.IsDirectory(ckpt_path):
            print("Loading model from checkpoint: %s" % ckpt_path)
            checkpoint_path = tf.train.latest_checkpoint(ckpt_path)
            if not checkpoint_path:
                raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
            tf.train.Saver().restore(session, checkpoint_path)
            print("Successfully loaded checkpoint: %s" % checkpoint_path)
        else:
            print("Model not found, created with fresh parameters.")
            session.run(tf.global_variables_initializer())

        return model

    def predict(self, line):
        line_ids = rawdata.get_idx_from_sent(line, self.word_idx_map, self.config.sentence_length)
        x = np.zeros([1, self.config.sentence_length], dtype=np.int64)
        x[0] = line_ids
        y = np.zeros([1, self.config.class_num], dtype=np.int64)
        fetches = [self.model.predict_y]

        feed_dict = {}
        feed_dict[self.model.x_in] = x
        feed_dict[self.model.y_in] = y
        feed_dict[self.model.keep_prob] = 1.0

        predict_y = self.session.run(fetches, feed_dict)

        return predict_y


def predict(data_path, ckpt_path, predict_file, output_file):
    model = ModelLoader(data_path, ckpt_path)
    with codecs.open(predict_file, 'r', 'utf-8') as predict,\
            codecs.open(output_file, 'w', 'utf-8') as output:
        lines = predict.readlines()
        for line in lines:
            tagging = model.predict(line)
            output.write(line)
            output.write(",".join(str(element) for element in tagging[0][0]))
            output.write('\n')