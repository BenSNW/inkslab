# -*- coding:utf-8 -*-

import os
import datetime
import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score
from deepnlp.relation_extraction.model import AttGRUModel


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


def test(args):
    model_path = os.path.join(args.train_dir, "ATT_GRU_model-100")

    word_embedding = np.load(os.path.join(args.data_dir, 'vec.npy'))

    # args.vocab_size = 16693
    # args.num_classes = 12
    # args.batch_size = 5561

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            print("read model")
            with tf.variable_scope("model"):
                mtest = AttGRUModel(is_training=False, word_embeddings=word_embedding, args=args)

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):
                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            # ckpt = tf.train.get_checkpoint_state(model_path)
            # saver = tf.train.Saver(tf.global_variables())
            # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            #     saver.restore(sess, ckpt.model_checkpoint_path)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            print(names_to_vars)
            saver = tf.train.Saver(names_to_vars)

            # testlist = range(1000, 1800, 100)
            # testlist = [9000]

            # for model_iter in testlist:
            # for compatibility purposes only, name key changes from tf 0.x to 1.x, compat_layer
            print("restore %s" % model_path)
            saver.restore(sess, model_path)

            print(datetime.datetime.now().isoformat())
            print('Evaluating all test data and save data for PR curve')

            test_y = np.load(os.path.join(args.data_dir, 'testall_y.npy'))
            test_word = np.load(os.path.join(args.data_dir, 'testall_word.npy'))
            test_pos1 = np.load(os.path.join(args.data_dir, 'testall_pos1.npy'))
            test_pos2 = np.load(os.path.join(args.data_dir, 'testall_pos2.npy'))
            allprob = []
            acc = []
            for i in range(int(len(test_word) / float(args.batch_size))):
                prob, accuracy = test_step(test_word[i * args.batch_size:(i + 1) * args.batch_size],
                                           test_pos1[i * args.batch_size:(i + 1) * args.batch_size],
                                           test_pos2[i * args.batch_size:(i + 1) * args.batch_size],
                                           test_y[i * args.batch_size:(i + 1) * args.batch_size])
                acc.append(np.mean(np.reshape(np.array(accuracy), args.batch_size)))
                prob = np.reshape(np.array(prob), (args.batch_size, args.num_classes))
                for single_prob in prob:
                    allprob.append(single_prob[1:])
            allprob = np.reshape(np.array(allprob), -1)
            # order = np.argsort(-allprob)

            print('saving all test result...')
            # current_step = model_iter

            np.save(os.path.join(args.data_dir, 'allprob_iter.npy'), allprob)
            allans = np.load(os.path.join(args.data_dir, 'allans.npy'))

            # caculate the pr curve area
            average_precision = average_precision_score(allans[:len(allprob)], allprob)
            print('PR curve area:' + str(average_precision))
