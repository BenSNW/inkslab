# -*- coding:utf-8 -*-

import pickle

import tensorflow as tf
from deepnlp.idcnn_or_bilstm_crf_segment.stand_alone.model import Model
from deepnlp.idcnn_or_bilstm_crf_segment.stand_alone.loader import load_data
from deepnlp.idcnn_or_bilstm_crf_segment.dataset.batch_manager import BatchManager
from deepnlp.idcnn_or_bilstm_crf_segment.dataset.data_utils import *
from deepnlp.idcnn_or_bilstm_crf_segment.stand_alone.utils import *


def evaluate(sess, model, name, data, id_to_tag, result_path):
    print("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_segment(ner_results, result_path)
    for line in eval_lines:
        print(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            print("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            print("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train(args):
    if args.clean:
        clean_and_make_path(args)

    train_data = load_data(args.train_file)
    dev_data = load_data(args.dev_file)
    test_data = load_data(args.test_file)

    with open(args.vocab_path, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    args.num_chars =  len(char_to_id)
    args.num_tags = len(tag_to_id)

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, args.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data

    with tf.Session(config=tf_config) as sess:
        model = Model(args)

        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
            if args.pre_emb:
                emb_weights = sess.run(model.char_lookup.read_value())
                emb_weights = load_word2vec(args.emb_file, id_to_char, args.char_dim, emb_weights)
                sess.run(model.char_lookup.assign(emb_weights))
                print("Load pre-trained embedding.")

        print("start training")
        loss = []
        for i in range(args.max_epoch):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % args.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    print("iteration:{} step:{}/{}, "
                                "Segment loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, args.result_path)
            if best:
                checkpoint_path = os.path.join(args.ckpt_path, args.model_type + ".segment.ckpt")
                model.saver.save(sess, checkpoint_path)
                print("model saved")
            evaluate(sess, model, "test", test_manager, id_to_tag, args.result_path)