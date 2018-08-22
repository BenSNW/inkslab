# -*- coding:utf-8 -*-

import tensorflow as tf
from deepnlp.lstm_ner.train import train_ner
from deepnlp.lstm_ner.predict import predict_ner
import argparse

r'''
Example usage:
    ./run \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=train or infer \
        --method=lstm or bilstm
'''


def main(args):
    if args.process == tf.estimator.ModeKeys.TRAIN:
        if args.model == "lstm":
            train_ner.train_lstm(args)
        else:
            train_ner.train_bilstm(args)
    elif args.process == tf.estimator.ModeKeys.PREDICT:
        if args.model == "lstm":
            predict_ner.predict(args)
        else:
            predict_ner.predict(args)
    else:
        raise Exception("cannot support this process:" + args.process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, type=str, default='data/lstm_ner/ckpt')
    parser.add_argument('--data_dir', required=True, type=str, default='data/lstm_ner/data')
    parser.add_argument('--utils_dir', required=True, type=str, default='data/lstm_ner/utils')

    parser.add_argument('--model', type=str, default='bilstm')

    parser.add_argument('--max_grad_norm', type=int, default=10)

    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=14)

    parser.add_argument('--init_scale', type=float, default=0.04)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=float, default=0.9)

    parser.add_argument('--predict_file', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--process', type=str, default='infer')

    args = parser.parse_args()

    main(args)
