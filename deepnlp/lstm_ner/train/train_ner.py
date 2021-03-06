# -*- coding:utf-8 -*-

import tensorflow as tf
import os
from deepnlp.lstm_ner.dataset import dataset, rawdata
from deepnlp.lstm_ner import ner_model, ner_model_bilstm


def train_lstm(args):
    if not args.data_dir:
        raise ValueError("No data files found in 'data_path' folder")

    if not os.path.isdir(args.utils_dir):
        os.mkdir(args.utils_dir)

    if not os.path.isdir(args.train_dir):
        os.mkdir(args.train_dir)

    raw_data = rawdata.load_data(args.data_dir, args.utils_dir, args.seq_length)
    train_word, train_tag, dev_word, dev_tag, vocab_size, tag_size = raw_data

    train_dataset = dataset.Dataset(train_word, train_tag)
    valid_dataset = dataset.Dataset(dev_word, dev_tag)
    args.vocab_size = vocab_size
    args.tag_size = tag_size

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_normal_initializer(-args.init_scale, args.init_scale)
        with tf.variable_scope('ner_var_scope', reuse=None, initializer=initializer):
            m = ner_model.NERTagger(is_training=True, config=args)
        with tf.variable_scope('ner_var_scope', reuse=True, initializer=initializer):
            valid_m = ner_model.NERTagger(is_training=False, config=args)

        sess.run(tf.global_variables_initializer())

        for i in range(args.num_epochs):
            lr_decay = args.lr_decay ** max(float(i - args.max_epoch), 0.0)
            m.assign_lr(sess, args.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))

            train_perplexity = ner_model.run(sess, m, train_dataset, m.train_op,
                                             ner_train_dir=args.train_dir, epoch=i)

            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            valid_perplexity = ner_model.run(sess, valid_m, valid_dataset, tf.no_op(),
                                             ner_train_dir=args.train_dir, epoch=i)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            train_dataset.reset()
            valid_dataset.reset()


def train_bilstm(args):
    if not args.data_dir:
        raise ValueError("No data files found in 'data_path' folder")

    if not os.path.isdir(args.utils_dir):
        os.mkdir(args.utils_dir)

    if not os.path.isdir(args.train_dir):
        os.mkdir(args.train_dir)

    raw_data = rawdata.load_data(args.data_dir, args.utils_dir, args.seq_length)
    train_word, train_tag, dev_word, dev_tag, vocab_size, tag_size = raw_data

    train_dataset = dataset.Dataset(train_word, train_tag)
    valid_dataset = dataset.Dataset(dev_word, dev_tag)

    args.vocab_size = vocab_size
    args.tag_size = tag_size

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_normal_initializer(-args.init_scale, args.init_scale)
        with tf.variable_scope('ner_var_scope', reuse=None, initializer=initializer):
            m = ner_model_bilstm.NERTagger(is_training=True, config=args)
        with tf.variable_scope('ner_var_scope', reuse=True, initializer=initializer):
            valid_m = ner_model_bilstm.NERTagger(is_training=False, config=args)

        sess.run(tf.global_variables_initializer())

        for i in range(args.num_epochs):
            lr_decay = args.lr_decay ** max(float(i - args.max_epoch), 0.0)
            m.assign_lr(sess, args.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
            train_perplexity = ner_model_bilstm.run(sess, m, train_dataset, m.train_op,
                                                    ner_train_dir=args.train_dir, epoch=i)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = ner_model_bilstm.run(sess, valid_m, valid_dataset, tf.no_op(),
                                                    ner_train_dir=args.train_dir, epoch=i)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            train_dataset.reset()
            valid_dataset.reset()
