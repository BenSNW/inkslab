# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from deepnlp.idcnn_or_bilstm_crf_segment.distributed.model import Model
from deepnlp.idcnn_or_bilstm_crf_segment.dataset.dataset import load_raw, do_load_data

def predict(args):
    char_to_id = dict()
    id_to_char = dict()
    with open(args.vocab_path, "r") as f:
        for line in f.readlines():
            id_word = line.split("\t")
            char_to_id[id_word[1].strip()]=id_word[0]
            id_to_char[id_word[0]]=id_word[1].strip()

    graph = tf.Graph()
    with graph.as_default():
        tX, tY = do_load_data(args.raw_data_path, args.max_sentence_len)  # 直接读取测试数据集

        model = Model(args)
        # model.run(tX, tY)
        test_unary_score, test_sequence_length = model.test_unary_score()
        sv = tf.train.Supervisor(graph=graph, logdir=args.log_dir)
        with sv.managed_session(master='') as sess:
            # trainsMatrix = sess.run(
            #     [model.transition_params])
            acc = evaluate(sess, args.batch_size, test_unary_score,
                           test_sequence_length, model.trains_params,
                           model.inp, tX, tY)
            print(acc)




    # saver = tf.train.Saver(tf.global_variables())
    # with tf.Session() as sess:
    #     ckpt = tf.train.get_checkpoint_state(args.log_dir)
    #     if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    #         print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #
    #     model.run(tX, tY)
    #     test_unary_score, test_sequence_length = model.test_unary_score()
    #     predict_path(sess, tX, tY, 1, test_unary_score, test_sequence_length,
    #                             model.trains_params, model.inp)

    # graph = tf.Graph()
    # with graph.as_default():
    #     tX = load_raw1(args.raw_data_path, args.max_sentence_len, char_to_id)  # 直接读取测试数据集
    #     model = Model(args)
    #     test_unary_score, test_sequence_length = model.test_unary_score()
    #
    #     sv = tf.train.Supervisor(graph=graph, logdir=args.log_dir)
    #     with sv.managed_session(master='') as sess:
    #         sequence = predict_path(sess, tX, 1, test_unary_score, test_sequence_length,
    #                      model.trains_params, model.inp)


def evaluate(sess, batch_size, unary_score, test_sequence_length, transMatrix, inp,
                  tX, tY):
    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] - 1) / batch_size) + 1
    correct_labels = 0
    total_labels = 0
    for i in range(numBatch):
        endOff = (i + 1) * batch_size
        if endOff > totalLen:
            endOff = totalLen
        y = tY[i * batch_size:endOff]
        feed_dict = {inp: tX[i * batch_size:endOff]}
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)
        for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_sequence_length_val):
            # print("seg len:%d" % (sequence_length_))
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            # print(tf_unary_scores_.shape)
            # print(transMatrix.shape)

            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, transMatrix)
            # Evaluate word-level accuracy.
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += sequence_length_
    accuracy = 100.0 * correct_labels / float(total_labels)
    print("Accuracy: %.3f%%" % accuracy)
    return accuracy

