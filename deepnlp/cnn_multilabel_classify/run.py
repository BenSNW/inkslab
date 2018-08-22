# -*- coding:utf-8 -*-

import tensorflow as tf

from deepnlp.cnn_multilabel_classify.train import train
from deepnlp.cnn_multilabel_classify.predict import predict

r'''
Example usage:
    ./run \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=train or predict
'''
flags = tf.app.flags

flags.DEFINE_string('train_dir', '/data/multi_label_classify/ckpt', 'Directory to save the checkpoints and training classify.')
flags.DEFINE_string('data_dir', '/data/multi_label_classify/data', 'Directory to save the train and test data.')
flags.DEFINE_string('predict_file', '', 'File to classify.')
flags.DEFINE_string('output_file', '', 'File to save results of predict_file.')

flags.DEFINE_string('process', 'train', 'process to train or predict')

FLAGS = flags.FLAGS

def main(_):
    if FLAGS.process == tf.estimator.ModeKeys.TRAIN:
        train.train_cnn_classfier(FLAGS.train_dir, FLAGS.data_dir)
    elif FLAGS.process == tf.estimator.ModeKeys.PREDICT:
        predict.predict(FLAGS.data_dir, FLAGS.train_dir, FLAGS.predict_file, FLAGS.output_file)
    else:
        raise Exception("cannot support this process:" + FLAGS.process)

if __name__ == '__main__':
    tf.app.run()
