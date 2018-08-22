# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import numpy as np
from inkslab.python.common.tf_utils import *
from inkslab.python.dataset.dataset_utils import read_tfrecord_data, load_pickle_data
from inkslab.python.models.text_classification_model import CNNModel
from inkslab.python.models.text_classification_model import RNNModel


class TrainCNNClassfier(object):
    def __init__(self, opts):
        self._options = opts
        # load_data
        self.tX, self.tY = load_pickle_data(opts.val_path)
        self.tX = np.array(self.tX)
        self.tY = np.array(self.tY)

        self.word2id, self.id2word, self.label2id, self.id2label, self.word2vec \
            = load_pickle_data(opts.vocab_info_path)

        self.vocab_size = len(self.word2id)
        print('The shape of word2vec is {}'.format(self.word2vec.shape))
        # Add information into opts to construct model
        self._options.vocab_size = self.vocab_size
        self._options.label_size = len(self.label2id)
        # Get data from tfrecord file
        examples, labels = read_tfrecord_data(opts.batch_size, opts.train_path, opts.sentence_length)
        self.model = model = self.forward(examples, labels)
        self._lr, self._loss, self._accuracy, self._train_op, self.global_step = \
            model.learning_rate, model.loss, model.accuracy, model.optimizer, model.global_step
        self.sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                          device_filters=["/job:ps", "/job:worker/task:%d" % opts.task_index])

    def forward(self, examples, labels):
        config = self._options
        if config.model == 'CNN':
            model = CNNModel(config, x_in=examples, y_in=labels, word2vec=self.word2vec)
        elif config.model == 'RNN':
            model = RNNModel(config, x_in=examples, y_in=labels, word2vec=self.word2vec)
        else:
            raise Exception('{} is not supported!'.format(config.model))
        return model

    def train(self, sess, saver):
        opts = self._options
        # sv = self.sv
        best_accuracy, val_acc_list = 0.0, []
        tf.train.write_graph(sess.graph, opts.log_dir, opts.external_info+'best_model.pbtxt')
        for step in range(1, opts.training_steps):
            if sess.should_stop():
                break
            try:
                sess.run([self._train_op])
                if step % 100 == 0:
                    (loss, lr, accuracy) = sess.run(
                        [self._loss, self._lr, self._accuracy])
                    rate = step / float(opts.training_steps)
                    stamp = datetime.now().strftime('%F %T')
                    print("Training: %s Step = %5d, lr = %5.4f, accuracy = %6.3f, loss = %6.3f, samples_rate = %6.3s\r" %
                          (stamp, step, lr, accuracy, loss, rate))
                    sys.stdout.flush()
                    if step % 500 == 0:
                        val_accuracy, val_loss = self.evaluate(sess=sess, tX=self.tX, tY=self.tY)
                        val_acc_list.append((step, val_accuracy))
                        print("Validation: %s Step %6d: lr = %5.4f, accuracy = %6.3f, loss = %6.3f" %
                              (stamp, step, lr, val_accuracy, val_loss))
                        sys.stdout.flush()
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            saver.save(get_session(sess), os.path.join(opts.log_dir, opts.external_info+'best_model'))
                            print('Model epoch-{} has been saved!'.format(step))
                        else:
                            print('Current step is not good enough!')
            except KeyboardInterrupt as e:
                saver.save(get_session(sess), save_path=os.path.join(opts.log_dir, 'Interrupt_%d' % step))
                raise e
        return val_acc_list

    def evaluate(self, sess, tX, tY):
        opts = self._options
        batch_size = opts.batch_size
        total_len = tX.shape[0]
        num_batch = int((tX.shape[0] - 1) / batch_size) + 1
        batch_size_list = []
        accuracy_list = []
        loss_list = []
        for i in range(num_batch):
            end_off = (i + 1) * batch_size
            if end_off > total_len:
                end_off = total_len
            sample_num = end_off - i * batch_size
            currX = tX[i * batch_size: end_off]
            currY = tY[i * batch_size: end_off]
            loss_value, accuracy_value = self.model.accuracy_loss(sess=sess, x=currX, y=currY)

            batch_size_list.append(sample_num)
            accuracy_list.append(accuracy_value * sample_num)
            loss_list.append(loss_value * sample_num)

        accuracy_value = sum(accuracy_list) / max(sum(batch_size_list), 1)
        loss_value = sum(loss_list) / max(sum(batch_size_list), 1)
        return accuracy_value, loss_value


def _train(args, server):
    model = TrainCNNClassfier(opts=args)
    saver = get_saver()
    with get_monitored_training_session(server, args.task_index, args.log_dir) as sess:
        val_acc_list = model.train(sess, saver)
        # import matplotlib.pyplot as plt
        # e_list, acc_list = [], []
        # with open(os.path.join(args.log_dir, args.model + '_val_correct_rate.txt'), 'w') as f:
        #     for e, acc in val_acc_list:
        #         f.write(str(e) + ',' + str(acc) + '\n')
        #         e_list.append(e)
        #         acc_list.append(acc)
        # f.close()
        # plt.plot(e_list, acc_list)
        # plt.xlabel('epoch')
        # plt.ylabel('val_correct_rate')
        # plt.show()


def trainer(args):
    distributed_run(args, _train)
