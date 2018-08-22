# -*- coding: utf-8 -*-

import argparse
import os
from inkslab.python.pos_tagging.train import train
from inkslab.python.pos_tagging.tag_model import TfTagModel


def main():
    if args.process == "train":
        train(args)
    elif args.process == "predict":
        tag_model = TfTagModel()
        model_file = os.path.join(args.model_dir, args.model_name)
        tag_model.load_model(model_file, args.vocab_path, args.user_dict_path)
        tag_model.tagging_file(args.raw_data_path, args.result_data_path)
    else:
        raise Exception("not support this process: %s" % args.process)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--process', type=str, default="train", help="Whether train the model")

        parser.add_argument('--workers', type=str)
        parser.add_argument('--parameter_servers', type=str)
        parser.add_argument('--job_name', type=str, default='ps')
        parser.add_argument('--task_index', type=int, default=0)

        parser.add_argument('--train_data_path', type=str, default="train_data_path", help="File to train")
        parser.add_argument('--test_data_path', type=str, default="test_data_path", help="File to test")
        parser.add_argument('--dev_data_path', type=str, default="test_data_path", help="File to test")
        parser.add_argument('--raw_data_path', type=str, default="raw_data_path", help="Raw file to predict")
        parser.add_argument('--result_data_path', type=str, default="result_data_path", help="Result file")
        # 用户自定义词典
        parser.add_argument('--user_dict_path', type=str, default="user_dict_path", help="User dict path")

        parser.add_argument('--vocab_path', type=str, default="vocab_path", help="Vocab file path")

        parser.add_argument('--model_dir', type=str, default="model_dir", help="model dir")
        parser.add_argument('--model_name', type=str, default="tagging_model.pbtxt", help="model name")

        parser.add_argument('--word2vec_path', type=str, default="vec.txt", help="word2vec pretrained file")
        parser.add_argument('--max_sentence_len', type=int, default=100, help="max sentence len")
        parser.add_argument('--embedding_size', type=int, default=100, help="embedding size")
        parser.add_argument('--num_hidden', type=int, default=100, help="the num of hidden size")
        parser.add_argument('--batch_size', type=int, default=100, help="batch size")
        parser.add_argument('--train_steps', type=int, default=1000, help="Train step")
        parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate ")

        parser.add_argument('--model_method', type=str, default="idcnn", help="Model type, can be idcnn or bilstm")

        args = parser.parse_args()
        main()
