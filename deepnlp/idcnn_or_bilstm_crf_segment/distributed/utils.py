# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

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

