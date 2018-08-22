# -*- coding:utf-8 -*-

import pickle
import tensorflow as tf
from deepnlp.idcnn_or_bilstm_crf_segment.stand_alone.model import Model
from deepnlp.idcnn_or_bilstm_crf_segment.dataset.data_utils import input_from_line
from deepnlp.idcnn_or_bilstm_crf_segment.dataset.batch_manager import BatchManager


def predict(args):
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(args.vocab_path, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    args.num_chars = len(char_to_id)
    args.num_tags = len(tag_to_id)

    with tf.Session(config=tf_config) as sess:
        model = Model(args)

        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No model parameters read.")

        with open(args.raw_file, "r", encoding='utf-8') as f:
            for line in f.readlines():
                inputs = input_from_line(line, char_to_id)
                results = []
                for sentence in inputs:
                    result = model.predict_line(sess, sentence, id_to_tag)
                    results.extend(result)
                print(" ".join(results))

def predict_batch(args):
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(args.vocab_path, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    args.num_chars = len(char_to_id)
    args.num_tags = len(tag_to_id)

    with tf.Session(config=tf_config) as sess:
        model = Model(args)

        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No model parameters read.")

        raw_in = open(args.raw_file, "r", encoding='utf-8')
        lines = raw_in.readlines()
        data = []
        for line in lines:
            trip_line = line.strip().replace(' ','')
            ids = [char_to_id[c] if c in char_to_id else char_to_id["<UNK>"] for c in trip_line]
            tags= [0 for _ in trip_line]
            data.append([ids, tags])

        predict_manager = BatchManager(data, 5, False)

        results = model.predict_batch(sess, predict_manager, id_to_tag, id_to_char)

        with open(args.predict_out_file, 'w') as write:
            for result in results:
                write.write(" ".join(result))
                write.write("\n")
        # print(results)







