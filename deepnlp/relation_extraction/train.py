# -*- coding:utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import datetime
from deepnlp.relation_extraction.model import AttGRUModel
from deepnlp.relation_extraction.dataset.utils import handle_data


def train(args):
    handle_data(args.data_dir)

    print('reading word_embedding')
    word_embedding = np.load(os.path.join(args.data_dir, 'vec.npy'))

    print('reading training data')
    train_y = np.load(os.path.join(args.data_dir, 'train_y.npy'))
    train_word = np.load(os.path.join(args.data_dir, 'train_word.npy'))
    train_pos1 = np.load(os.path.join(args.data_dir, 'train_pos1.npy'))
    train_pos2 = np.load(os.path.join(args.data_dir, 'train_pos2.npy'))

    args.vocab_size = len(word_embedding)
    args.num_classes = len(train_y[0])

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = AttGRUModel(is_training=True, word_embeddings=word_embedding, args=args)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.0005)

            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)

            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(args.summary_dir + '/train_loss', sess.graph)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, batch_size):
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

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                acc = np.mean(np.reshape(np.array(accuracy), batch_size))
                summary_writer.add_summary(summary, step)

                if step % 50 == 0:
                    print("{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc))

            for epoch in range(args.num_epochs):
                print("epoch num %d" % epoch)
                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)
                for i in range(int(len(temp_order) / float(args.batch_size))):
                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    temp_input = temp_order[i * args.batch_size:(i + 1) * args.batch_size]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        print('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)

                    train_step(temp_word, temp_pos1, temp_pos2, temp_y, args.batch_size)

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % 100 == 0:
                        print('saving model')
                        path = saver.save(sess, os.path.join(args.train_dir, 'ATT_GRU_model'), global_step=current_step)
                        print('saved model to ' + path)
