# -*- coding:utf-8 -*-

import argparse

from deepnlp.idcnn_or_bilstm_crf_segment.distributed.train import train
from deepnlp.idcnn_or_bilstm_crf_segment.distributed.predict import predict

def main(args):
    if args.train:
        train(args)
    else:
        predict(args)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', type=bool, default=False, help="Whether train the model")
        parser.add_argument('--clean', type=bool, default=True, help="Whether clean the model")

        parser.add_argument('--train_data_path', type=str, default="ckpt", help="Path to save model")
        parser.add_argument('--test_data_path', type=str, default="train.log", help="File for log")
        parser.add_argument('--raw_data_path', type=str, default="train.log", help="File for log")
        parser.add_argument('--vocab_path', type=str, default="train.log", help="File for log")

        parser.add_argument('--log_dir', type=str, default="vocab.json", help="Path to vocab file")
        parser.add_argument('--word2vec_path', type=str, default="config_file", help="File for config")
        parser.add_argument('--max_sentence_len', type=int, default=100, help="evaluation script")
        parser.add_argument('--embedding_size', type=int, default=100, help="Path to result")
        parser.add_argument('--num_tags', type=int, default=4, help="Path for pre_trained embedding")
        parser.add_argument('--num_hidden', type=int, default=100, help="Path for train data")
        parser.add_argument('--batch_size', type=int, default=100, help="Path for dev data")
        parser.add_argument('--train_steps', type=int, default=300, help="Path for test data")
        parser.add_argument('--learning_rate', type=float, default=0.001, help="Path for predict data")

        parser.add_argument('--use_idcnn', type=bool, default=True, help="Model type, can be idcnn or bilstm")

        parser.add_argument('--track_history', type=int, default=6,
                            help="Embedding size for segmentation, 0 if not used")

        args = parser.parse_args()
        main(args)
