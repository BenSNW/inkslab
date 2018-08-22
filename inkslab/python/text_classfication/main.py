# -*- coding: utf-8 -*-

import argparse
from inkslab.python.text_classfication.train import trainer
from inkslab.python.text_classfication.predict import predictor


def main():
    if args.process == "train":
        trainer(args)
    elif args.process == "predict":
        predictor(args)
    else:
        raise Exception('Only support train and predict')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, default="train", choices=["train", "predict"])
    parser.add_argument('--workers', type=str)
    parser.add_argument('--parameter_servers', type=str)
    parser.add_argument('--job_name', type=str, default='ps')
    parser.add_argument('--task_index', type=int, default=0)

    parser.add_argument('--log_dir', default='data/log')
    parser.add_argument('--train_path', default='data/train/output.tfrecord')
    parser.add_argument('--val_path', default='data/train/val.pkl')
    parser.add_argument('--test_path', default='data/train/test.pkl')
    parser.add_argument('--vocab_info_path', type=str, default='data/vocab_info.pkl')
    parser.add_argument("--use_pre_trained_vec", type=bool, default=False)
    parser.add_argument("--pre_trained_vec_path", type=str, default='data/word2vec/vec.txt')
    parser.add_argument('--valid_words', type=str)

    parser.add_argument('--training_steps', type=int, default=20000)
    parser.add_argument('--max_to_keep', type=int, default=10)

    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sentence_length', type=int, default=20)
    parser.add_argument('--num_iter_decay', type=int, default=200)
    parser.add_argument('--l2_reg', type=float, default=0.01)
    parser.add_argument('--model', type=str, default='RNN')
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--vector_size', type=int, default=100)
    parser.add_argument('--rnn_cell', type=str, default='gru')
    parser.add_argument('--rnn_size', type=float, default=100)
    parser.add_argument('--rnn_mlp_hidden_dim', type=int, default=200)
    parser.add_argument('--num_rnn_layers', type=int, default=1)

    parser.add_argument('--filter_hs', type=list, default=[1, 2, 3, 5])
    parser.add_argument('--num_filters', type=float, default=128)
    parser.add_argument('--img_h', type=int, default=20)
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--filter_w', type=int, default=100)

    parser.add_argument('--external_info', type=str, default='')
    args = parser.parse_args()

    main()
