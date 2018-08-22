# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from deepnlp.cnn_multilabel_classify.dataset import rawdata
from deepnlp.cnn_multilabel_classify.cnn_config import CnnConfig
from deepnlp.cnn_multilabel_classify import cnn_model


def train_cnn_classfier(train_path, data_dir):
    config = CnnConfig()
    train_revs, test_revs, idx_word_map, word_idx_map = rawdata.get_data(data_dir)

    n_batches = len(train_revs) / config.batch_size

    # 开始定义模型============================================
    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        # 构建模型
        model = cnn_model.CNNModel(config=config)

        # 训练模型========================================
        checkpoint_dir = train_path
        checkpoint_prefix = os.path.join(checkpoint_dir, "classify")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        batch_x_test, batch_y_test = rawdata.get_test_batch(test_revs, word_idx_map, config.class_num)

        for i in range(config.num_epoch):
            print("Start epoch:" + str(i))
            cnn_model.run(sess, model, train_revs, n_batches, word_idx_map, config.class_num)
            test_accuracy = model.accuracy.eval(
                feed_dict={model.x_in: batch_x_test, model.y_in: batch_y_test, model.keep_prob: 1.0})
            current_step = tf.train.global_step(sess, model.global_step)
            print("Update step %d, test accuracy %g" % (current_step, test_accuracy))
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

    return model.embeddings, sess, idx_word_map