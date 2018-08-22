# -*- coding: utf-8 -*-

import argparse
import os
from inkslab.python.segments.seg_model import TfSegModel
from inkslab.python.segments.train import train


def main():
    if args.process == "train":
        train(args)
    elif args.process == "predict":
        seg_model = TfSegModel()
        model_file = os.path.join(args.model_dir, args.model_name)
        seg_model.load_model(model_file, args.vocab_path, args.user_dict_path)
        seg_model.segment_file(args.raw_data_path, args.result_data_path)
    else:
        raise Exception("not support this process: %s" % args.process)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 训练或者预测
    parser.add_argument('--process', type=str, default="train", help="Whether train the model")
    parser.add_argument('--workers', type=str, default='localhost:2220')
    parser.add_argument('--parameter_servers', type=str, default='localhost:2221')
    # parser.add_argument('--job_name', type=str, default='ps')
    parser.add_argument('--job_name', type=str, default='worker')
    parser.add_argument('--task_index', type=int, default=0)

    parser.add_argument('--train_data_path', type=str, default="datas/segment/train.txt", help="File to train")
    parser.add_argument('--test_data_path', type=str, default="datas/segment/test.txt", help="File to test")
    parser.add_argument('--dev_data_path', type=str, default="datas/segment/dev.txt", help="File to val")
    # 待分词的文件
    parser.add_argument('--raw_data_path', type=str, default="datas/segment/raw.txt", help="Raw file to predict")
    parser.add_argument('--result_data_path', type=str, default="datas/segment/seg_result.txt", help="Result file")
    # 用户自定义词典
    parser.add_argument('--user_dict_path', type=str, default="datas/common/ud_dict.txt", help="User dict path")
    # 词典路径
    parser.add_argument('--vocab_path', type=str, default="datas/segment/segment_vocab", help="Vocab file path")
    # 模型存储路径
    parser.add_argument('--model_dir', type=str, default="/Document/pycharm/inkslab/models/bilstm/", help="model dir")
    # 模型名字
    parser.add_argument('--model_name', type=str, default="segment_model.pbtxt", help="model name")
    # 训练好的词向量
    parser.add_argument('--word2vec_path', type=str, default="datas/common/char_vec.txt", help="pretrained word2vec")
    parser.add_argument('--max_sentence_len', type=int, default=100, help="max sentence len")
    parser.add_argument('--embedding_size', type=int, default=100, help="embedding size")
    parser.add_argument('--num_hidden', type=int, default=100, help="the num of hidden size")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--train_steps', type=int, default=500, help="Train step")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate ")
    # 使用 idcnn 或者 bilstm 模型
    parser.add_argument('--model_method', type=str, default="bilstm", help="Model type, idcnn or bilstm")

    args = parser.parse_args()
    main()
