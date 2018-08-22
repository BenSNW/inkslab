# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def evaluate(sess, batch_size, unary_score, test_sequence_length, trans_matrix, inp,
                  tx, ty):
    total_len = tx.shape[0]
    num_batch = int((tx.shape[0] - 1) / batch_size) + 1
    correct_labels = 0
    total_labels = 0
    for i in range(num_batch):
        end_off = (i + 1) * batch_size
        if end_off > total_len:
            end_off = total_len
        y = ty[i * batch_size:end_off]
        feed_dict = {inp: tx[i * batch_size:end_off]}
        unary_score_val, test_sequence_length_val = sess.run([unary_score, test_sequence_length], feed_dict)

        for tf_unary_scores_, y_, sequence_length_ in zip(unary_score_val, y, test_sequence_length_val):
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, trans_matrix)
            # Evaluate word-level accuracy.
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += sequence_length_
    accuracy = 100.0 * correct_labels / float(total_labels)
    print("Accuracy: %.3f%%" % accuracy)
    return accuracy


def predict_evaluate(sess, batch_size, trans_matrix, input_data, scores_tensor, tx, ty):
    total_len = tx.shape[0]
    num_batch = int((tx.shape[0] - 1) / batch_size) + 1
    correct_labels = 0
    total_labels = 0
    for i in range(num_batch):
        end_off = (i + 1) * batch_size
        if end_off > total_len:
            end_off = total_len
        y = ty[i * batch_size:end_off]

        unary_score_val = sess.run(scores_tensor, feed_dict={input_data: tx[i * batch_size:end_off]})

        unary_score_val = np.asarray(unary_score_val).reshape(-1, 100, 4)

        for tf_unary_scores_, y_ in zip(unary_score_val, y):
            predict_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, trans_matrix)

            correct_labels += np.sum(np.equal(predict_sequence, y_))
            total_labels += 100
    accuracy = 100.0 * correct_labels / float(total_labels)
    print("Accuracy: %.3f%%" % accuracy)
