# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
from deepnlp.glove.train import train


def main(args):
    if args.process == tf.estimator.ModeKeys.TRAIN:
        train(args)
    else:
        raise Exception("cannot support this process:" + args.process)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/glove')
    parser.add_argument('--log_dir', type=str, default='data/glove/log')

    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--context_size', type=int, default=10)
    parser.add_argument('--max_vocab_size', type=int, default=100000)
    parser.add_argument('--min_occurrences', type=int, default=1)

    parser.add_argument('--scaling_factor', type=float, default=0.75)
    parser.add_argument('--cooccurrence_cap', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--tsne_epoch_interval', type=int, default=50)

    parser.add_argument('--result_file', type=str, default='data/glove/out.txt')
    parser.add_argument('--process', type=str, default='train')

    args = parser.parse_args()

    # import os
    # import sys
    # print(os.getcwd())
    # print(sys.path)
    # print(os.path.join(args.data_dir, "train_little.txt"))
    # print(os.path.abspath(os.path.join(args.data_dir, "train_little.txt")))
    # print(os.path.normpath(os.path.join(args.data_dir, "train_little.txt")))

    # open(r'data/glove/train_little.txt')
    # open(os.path.abspath(os.path.join(args.data_dir, "train_little.txt")))

    main(args)
