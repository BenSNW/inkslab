# -*- coding: utf-8 -*-


import tensorflow as tf
from deepnlp.idcnn_or_bilstm_crf_segment.distributed.model import Model
from deepnlp.idcnn_or_bilstm_crf_segment.dataset.dataset import get_batch_data, do_load_data
from deepnlp.idcnn_or_bilstm_crf_segment.distributed.utils import evaluate

def train(args):
    trainDataPath = args.train_data_path
    graph = tf.Graph()
    with graph.as_default():
        model = Model(args)
        print("train data path:", trainDataPath)
        X, Y = get_batch_data(args.batch_size, trainDataPath, args.max_sentence_len) # 使用多线程的方式读取训练数据集
        tX, tY = do_load_data(args.test_data_path, args.max_sentence_len) # 直接读取测试数据集
        model.run(X, Y)
        test_unary_score, test_sequence_length = model.test_unary_score()
        sv = tf.train.Supervisor(graph=graph, logdir=args.log_dir)
        with sv.managed_session(master='') as sess:
            # actual training loop
            training_steps = args.train_steps
            trackHist = 0
            bestAcc = 0
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    _, trainsMatrix = sess.run(
                        [model.train_op, model.transition_params])
                    # for debugging and learning purposes, see how the loss
                    # gets decremented thru training steps
                    if (step + 1) % 100 == 0:
                        print("[%d] loss: [%r]" %
                              (step + 1, sess.run(model.loss)))
                    if (step + 1) % 1000 == 0 or step == 0:
                        acc = evaluate(sess, args.batch_size, test_unary_score,
                                            test_sequence_length, trainsMatrix,
                                            model.inp, tX, tY)
                        if acc > bestAcc:
                            if step:
                                sv.saver.save(
                                    sess, args.log_dir + '/best_model')
                            bestAcc = acc
                            trackHist = 0
                        elif trackHist > args.track_history:
                            print(
                                "always not good enough in last %d histories, best accuracy:%.3f"
                                % (trackHist, bestAcc))
                            break
                        else:
                            trackHist += 1
                except KeyboardInterrupt as e:
                    sv.saver.save(sess,
                                  args.log_dir + '/model',
                                  global_step=(step + 1))
                    raise e
            sv.saver.save(sess, args.log_dir + '/finnal-model')