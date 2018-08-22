# -*- coding:utf-8 -*-

import pickle
from inkslab.python.models.segment_model import Model
from inkslab.python.dataset.dataset_utils import generate_batch_data, do_load_data
from inkslab.python.evaluate import evaluate_seg
from inkslab.python.common.tf_utils import *


def _train(args, server):
    train_data_path = args.train_data_path
    with open(args.vocab_path, 'rb') as f:
        _, _, tag_to_id, _ = pickle.load(f)

    args.num_tags = len(tag_to_id)

    model = Model(args)
    print("train data path:", train_data_path)
    # 使用多线程的方式读取训练数据集
    X, Y = generate_batch_data(args.batch_size, train_data_path, args.max_sentence_len)
    # 直接读取测试数据集
    tX, tY = do_load_data(args.test_data_path, args.max_sentence_len)
    model.run(X, Y)
    test_unary_score, test_sequence_length = model.test_unary_score()

    saver = get_saver()

    with get_monitored_training_session(server, args.task_index, args.model_dir) as sess:
        tf.train.write_graph(sess.graph, args.model_dir, 'model_graph.pbtxt')
        # actual training loop
        training_steps = args.train_steps
        best_acc = 0
        for step in range(training_steps):
            if sess.should_stop():
                break
            try:
                _, trains_matrix, global_step = \
                    sess.run([model.train_op, model.transition_params, model.global_step])

                if (step + 1) % 100 == 0:
                    print("global step: %d step: %d loss: %r" % (global_step, step + 1, sess.run(model.loss)))
                if (step + 1) % 500 == 0 or step == 0:
                    acc = evaluate_seg.evaluate(sess, args.batch_size, test_unary_score,
                                                test_sequence_length, trains_matrix, model.inp, tX, tY)
                    if acc > best_acc and args.task_index == 0:
                        if step:
                            saver.save(get_session(sess), args.model_dir + '/best_model')
                        best_acc = acc
            except KeyboardInterrupt as e:
                raise e


def train(args):
    distributed_run(args, _train)


