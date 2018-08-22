# -*- coding: utf-8 -*-

from collections import deque
import pickle
import numpy as np
import tensorflow as tf
from inkslab.python.tf_model import TfModel
from inkslab.python.dataset.dataset_utils import result_to_sentence
from inkslab.python.dataset.constants import PAD, UNK

from sklearn.utils.extmath import softmax

class TfSegModel(object):
    def __init__(self):
        self.model = TfModel()
        self.num_tags = 0
        self.transitions = None
        # self.scanner = AcScanner()
        self.user_dict = {}

    def load_model(self, model_path, vocab_path, user_dic_path=None):
        load_suc = self.model.load(model_path)
        if not load_suc:
            print("Could not load model from: " + model_path)
            return False

        self.transitions = self.model.eval("transitions:0")
        self.transitions = softmax(self.transitions, copy=False)
        self.transitions[0, 2:4] = -100
        self.transitions[1, 0:2] = -100
        self.transitions[2, 0:2] = -100
        self.transitions[3, 2:4] = -100

        self.char2id, _, _, self.id2tag = self.load_vocab(vocab_path)
        self.tag2id = {v: k for k, v in self.id2tag.items()}

        self.num_tags = len(self.id2tag)

        if user_dic_path is not None:
            self.load_user_dic(user_dic_path)

    @staticmethod
    def load_vocab(vocab_path):
        with open(vocab_path, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

        return char_to_id, id_to_char, tag_to_id, id_to_tag

    def load_user_dic(self, user_dic_path):
        with open(user_dic_path, 'r', encoding='u8') as f:
            lines = f.readlines()
            for line in lines:
                word_weight = line.split(" ")
                word = word_weight[0]
                weight = float(word_weight[1])
                self.user_dict[word] = weight
        return True

    def segment(self, sentence: str = '', user_dict={}):
        sentence_ids = [self.char2id[c] if c in self.char2id else self.char2id[UNK] for c in sentence]
        in_data = np.asarray(sentence_ids).reshape(1, len(sentence_ids))
        results = self.model.eval_with_input(in_data, "scores_1:0")
        unary_score_val = np.asarray(results).reshape(len(sentence_ids), self.num_tags)

        if len(user_dict) > 0:
            ac = ACAutomaton(user_dict)
            for idx, word, weight in ac.get_keywords_found(sentence):
                if len(word) == 1:
                    unary_score_val[idx, self.tag2id['S']] += weight
                else:
                    unary_score_val[idx, self.tag2id['B']] += weight
                    for i in range(idx + 1, idx + len(word) - 1):
                        unary_score_val[i, self.tag2id['M']] += weight
                    idx += len(word) - 1
                    unary_score_val[idx, self.tag2id['E']] += weight

        unary_score_val = softmax(unary_score_val, copy=False)
        viterbi_prediction, _ = tf.contrib.crf.viterbi_decode(unary_score_val, self.transitions)
        tag_sequence = [self.id2tag[idx] for idx in viterbi_prediction]
        return result_to_sentence(sentence, tag_sequence)

    def segment_file(self, predict_file, result_file):
        with open(predict_file, 'r', encoding='u8') as f, open(result_file, 'w', encoding='u8') as w:
            sentences = f.readlines()
            for sentence in sentences:
                sentence = sentence.strip()
                result = self.segment(sentence, self.user_dict)
                w.write(" ".join(result))
                w.write("\n")


class ACAutomaton(object):

    def __init__(self, keywords: dict):
        """ creates a trie of from iterator of tuples, then sets fail transitions """

        # initialize the root of the trie
        self.AdjList = list()
        self.AdjList.append({'value': '', 'next_states': [], 'fail_state': 0, 'output': []})

        self.add_keywords_and_values(keywords.items())
        self.set_fail_transitions()

    def add_keywords(self, keywords):
        """ add all keywords in list of keywords """
        for keyword in keywords:
            self.add_keyword(keyword, None)

    def add_keywords_and_values(self, kvs):
        """ add all keywords and values in list of (k,v) """
        for k, v in kvs:
            self.add_keyword(k, v)

    def find_next_state(self, current_state, value):
        for node in self.AdjList[current_state]["next_states"]:
            if self.AdjList[node]["value"] == value:
                return node
        return None

    def add_keyword(self, keyword, value):
        """ add a keyword to the trie and mark output at the last node """
        current_state = 0
        j = 0
        keyword = keyword.lower()
        child = self.find_next_state(current_state, keyword[j])
        while child is not None:
            current_state = child
            j = j + 1
            if j < len(keyword):
                child = self.find_next_state(current_state, keyword[j])
            else:
                break
        for i in range(j, len(keyword)):
            node = {'value': keyword[i], 'next_states': [], 'fail_state': 0, 'output': []}
            self.AdjList.append(node)
            self.AdjList[current_state]["next_states"].append(len(self.AdjList) - 1)
            current_state = len(self.AdjList) - 1
        self.AdjList[current_state]["output"].append((keyword, value))

    def set_fail_transitions(self):
        q = deque()
        for node in self.AdjList[0]["next_states"]:
            q.append(node)
            self.AdjList[node]["fail_state"] = 0
        while q:
            r = q.popleft()
            for child in self.AdjList[r]["next_states"]:
                q.append(child)
                state = self.AdjList[r]["fail_state"]
                while self.find_next_state(state, self.AdjList[child]["value"]) is None and state != 0:
                    state = self.AdjList[state]["fail_state"]
                self.AdjList[child]["fail_state"] = self.find_next_state(state, self.AdjList[child]["value"])
                if self.AdjList[child]["fail_state"] is None:
                    self.AdjList[child]["fail_state"] = 0
                self.AdjList[child]["output"] = self.AdjList[child]["output"] + \
                                                self.AdjList[self.AdjList[child]["fail_state"]]["output"]

    def get_keywords_found(self, line):
        """ returns true if line contains any keywords in trie, format: (start_idx,kw,value) """
        line = line.lower()
        current_state = 0
        keywords_found = []

        for i in range(len(line)):
            while self.find_next_state(current_state, line[i]) is None and current_state != 0:
                current_state = self.AdjList[current_state]["fail_state"]
            current_state = self.find_next_state(current_state, line[i])
            if current_state is None:
                current_state = 0
            else:
                for k, v in self.AdjList[current_state]["output"]:
                    keywords_found.append((i - len(k) + 1, k, v))

        return keywords_found

