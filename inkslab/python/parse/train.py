# -*- coding: utf-8 -*-

import numpy as np
from inkslab.python.parse import transition_system
from inkslab.python.models.parse_model import NNParserModel
from inkslab.python.common.tf_utils import *


def run_epoch(session, model, eval_op, dataset, batch_size, is_train=True):
    costs = 0.0
    iters = 0
    las_correct_num = 0
    las_total_num = 0
    for step, (x, y) in enumerate(dataset):
        fetches = [model.loss, model.predict_label, model.correct_predict_num, eval_op]

        feed_dict = dict()
        feed_dict[model.X] = x
        feed_dict[model.Y] = y
        if is_train:
            loss, predict_label, correct_predict_num, _ = session.run(fetches, feed_dict)
        else:
            loss, predict_label, correct_predict_num = session.run(fetches, feed_dict)

        costs += loss
        iters += 1
        # Update UAS and LAS score
        las_correct_num += np.sum(correct_predict_num)
        las_total_num += np.sum(batch_size)
        if step % 10000 == 10:
            print("step %.3f perplexity: %.3f examples Accuracy %.3f " % (
                step * 1.0, np.exp(costs / iters), las_correct_num * 1.0 / las_total_num))

    return np.exp(costs / iters)


def _train(args, server):
        m = NNParserModel(config=args)

        with get_monitored_training_session(server, args.task_index, args.model_dir) as sess:
            for i in range(args.max_max_epoch):
                if sess.should_stop():
                    break
                lr_decay = args.lr_decay ** max(i - args.max_epoch, 0.0)
                m.assign_lr(sess, args.learning_rate * lr_decay)
                print("Epoch: %d Learning rate: %.4f" % (i + 1, sess.run(m.lr)))
                # Training: Check training set perplexity

                train_x, train_y = transition_system.get_examples(args.parse_data_path, is_train=True)
                train_iter = transition_system.iter_examples(train_x, train_y, args.batch_size)
                train_perplexity = run_epoch(sess, m, m.train_op, train_iter, args.batch_size)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                # Dev: Check development set perplexity
                # dev_x, dev_y = transition_system.get_examples(args.parse_data_path, is_train=False)
                # dev_iter = transition_system.iter_examples(dev_x, dev_y, m.batch_size)
                # dev_perplexity = run_epoch(sess, m, tf.no_op(), dev_iter, is_train=False)
                # print("Epoch: %d Dev Perplexity: %.3f" % (i + 1, dev_perplexity))


def train(args):
    distributed_run(args, _train)


