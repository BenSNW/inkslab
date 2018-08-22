# -*- coding: utf-8 -*-

import pickle
import numpy as np
import tensorflow as tf
from inkslab.python.tf_model import TfModel
from inkslab.python.base.ac_scanner import AcScanner, TaggingScanReporter
from inkslab.python.dataset.dataset_utils import result_to_sentence


class TfTagModel(object):
    def __init__(self):
        self.__model_ = TfModel()
        self.__num_tags = 0
        self.__transitions_ = None
        self.__scanner_ = AcScanner()

    def load_model(self, model_path, vocab_path, user_dic_path=None):
        load_suc = self.__model_.load(model_path)
        if not load_suc:
            print("Could not load model from: " + model_path)
            return False

        self.__transitions_ = self.__model_.eval("transitions:0")
        self.__char_to_id, _, self.__tag_to_id, self.__id_to_tag = self.load_vocab(vocab_path)
        self.__num_tags = len(self.__id_to_tag)
        if user_dic_path is not None:
            self.load_user_dic(user_dic_path)

    @staticmethod
    def load_vocab(vocab_path):
        with open(vocab_path, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

        return char_to_id, id_to_char, tag_to_id, id_to_tag

    def load_user_dic(self, user_dic_path):
        with open(user_dic_path, 'r') as f:
            lines = f.readlines()
            tn = 0
            for line in lines:
                word_weight = line.split(" ")
                word = word_weight[0]
                weight = int(word_weight[1])
                tag = word_weight[2]
                self.__scanner_.push_node(word, weight, tag)  # 词和词的权重添加到scanner_中
                tn += 1
        if tn > 1:
            self.__scanner_.build_fail_node()  # 构建fail node

        return True

    def tagging(self, sentence):
        result = None
        sentence_ids = [self.__char_to_id[c] if c in self.__char_to_id else self.__char_to_id['PAD'] for c in sentence]
        in_data = np.asarray(sentence_ids).reshape(1, len(sentence_ids))
        results = self.__model_.eval_with_input(in_data, "scores_1:0")
        unary_score_val = np.asarray(results).reshape(-1, len(sentence_ids), self.__num_tags)

        if self.__scanner_.num_item() > 0:
            # 启用用户自定义词典
            report = TaggingScanReporter(sentence, self.__tag_to_id)
            self.__scanner_.do_scan(sentence, report)
            report.fake_predication(unary_score_val, 0)  # 调整权重

        for unary_score_val_line in unary_score_val:
            predict_sequence, _ = tf.contrib.crf.viterbi_decode(
                unary_score_val_line, self.__transitions_)

            tag_sequence = [self.__id_to_tag[id] for id in predict_sequence]
            result = result_to_sentence(sentence, tag_sequence, "tagging")
            print(result)

        return result

    def tagging_file(self, predict_file, result_file):
        with open(predict_file, 'r') as f, open(result_file, 'w') as w:
            sentences = f.readlines()
            for sentence in sentences:
                sentence = sentence.strip()
                result = self.tagging(sentence)
                w.write(" ".join(result))
                w.write("\n")
