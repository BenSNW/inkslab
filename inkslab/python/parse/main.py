# -*- coding: utf-8 -*-

import argparse
from inkslab.python.parse.train import train
from inkslab.python.parse.freeze_predict import predict


def main():
    if args.process == "train":
        train(args)
    elif args.process == "predict":
        predict(args)
    else:
        raise Exception("not support this process: %s" % args.process)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--process', type=str, default="train", help="Whether train the model")

        parser.add_argument('--workers', type=str)
        parser.add_argument('--parameter_servers', type=str)
        parser.add_argument('--job_name', type=str, default='ps')
        parser.add_argument('--task_index', type=int, default=0)

        parser.add_argument('--parse_data_path', type=str, default="parse_data_path", help="File to train")

        parser.add_argument('--raw_data_path', type=str, default="raw_data_path", help="Raw file to predict")
        parser.add_argument('--result_data_path', type=str, default="result_data_path", help="Result file")

        parser.add_argument('--model_dir', type=str, default="model_dir", help="model dir")
        parser.add_argument('--model_name', type=str, default="tagging_model.pbtxt", help="model name")

        parser.add_argument('--init_scale', type=float, default=0.04, help="Init scale")
        parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
        parser.add_argument('--max_grad_norm', type=int, default=10, help="Max grad norm")
        parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dim")
        parser.add_argument('--hidden_size', type=int, default=256, help="Hidden size")
        parser.add_argument('--max_epoch', type=int, default=2, help="Max epoch")
        parser.add_argument('--max_max_epoch', type=int, default=5, help="Max max epoch")
        parser.add_argument('--lr_decay', type=float, default=0.90, help="Learning rate ")
        parser.add_argument('--pos_size', type=int, default=23, help="Pos size ")
        parser.add_argument('--label_size', type=int, default=70, help="Label size ")
        parser.add_argument('--feature_num', type=int, default=48, help="Feature num")
        parser.add_argument('--feature_word_num', type=int, default=18, help="Feature word num")
        parser.add_argument('--feature_pos_num', type=int, default=18, help="Feature pos num")
        parser.add_argument('--feature_label_num', type=int, default=12, help="Feature label num")

        parser.add_argument('--batch_size', type=int, default=1, help="batch size")
        parser.add_argument('--vocab_size', type=int, default=15000, help="Train step")

        args = parser.parse_args()
        main()
