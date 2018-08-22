# -*- coding: utf-8 -*-

import queue
import math
from abc import abstractmethod


class AcScanner(object):
    def __init__(self):
        self.__num_node_ = 1
        self.__root_ = TrieNode()
        self.__root_.fail_node_ = self.__root_

    def push_node(self, str, data, tag=None):
        cur = self.__root_
        nn = len(str)
        if nn == 0:
            return  # 词为空
        for i in range(nn):
            wc = str[i]
            prev = cur
            it = cur.transimition_.get(wc)
            if it is None:
                new_node = TrieNode()
                cur = new_node
                prev.transimition_[wc] = cur
                self.__num_node_ += 1
            else:
                cur = it

        if cur.is_leaf_:
            print("duplicated string found!")
            return
        else:
            cur.is_leaf_ = True
            cur.data_ = data
            cur.tag_ = tag
            cur.len_ = nn

    def build_fail_node(self):
        todos = queue.Queue()
        for (k, v) in self.__root_.transimition_.items():
            p_node = v
            p_node.fail_node_ = self.__root_
            todos.put(item=p_node)

        while not todos.empty():
            parent = todos.get()  # 从队首元素开始
            for (k, v) in parent.transimition_.items():
                wc = k
                cur = v
                parent_fail_node = parent.fail_node_
                it2 = parent_fail_node.transimition_.get(wc)
                while parent_fail_node != self.__root_ and it2 is None:
                    parent_fail_node = parent_fail_node.fail_node_
                    it2 = parent_fail_node.transimition_.get(wc)
                if it2 is None:
                    cur.fail_node_ = self.__root_
                else:
                    cur.fail_node_ = v
                todos.put(cur)  # already processed cur， 添加到队列中

    def do_scan(self, word, reporter):
        nn = len(word)
        if nn == 0:
            return True
        cur = self.__root_
        prev_found = None
        prev_pos = 0
        i = 0
        while i < nn:
            wc = word[i]
            parent = cur
            it = cur.transimition_.get(wc)
            if it is not None:
                cur = it
            else:
                cur = None
            if cur is None:
                if prev_found:
                    if reporter.callback(prev_pos, prev_found.data_, prev_found.len_, prev_found.tag_):
                        return True
                    # 从上一次匹配开头的下一个位置开始
                    i = prev_pos - prev_found.len_ + 2
                    prev_found = None
                    cur = self.__root_
                    continue
                else:
                    cur = parent.fail_node_

            if cur.is_leaf_:
                prev_found = cur
                prev_pos = i
            i = i + 1

        if prev_found:
            if reporter.callback(prev_pos, prev_found.data_, prev_found.len_, prev_found.tag_):
                return True
        return False

    def num_item(self):
        return self.__num_node_ - 1


class TrieNode(object):
    def __init__(self, is_leaf_=False, len_=0):
        self.data_ = None
        self.tag_ = None
        self.is_leaf_ = is_leaf_
        self.len_ = len_
        self.fail_node_ = None
        self.transimition_ = dict()


class ScanReporter(object):
    @abstractmethod
    def callback(self, pos, data, len, tag=None):
        pass


class FakeEmitInfo(object):
    def __init__(self, len):
        self.need_fake = False
        self.tag_ids = []
        self.weights = [1.0] * len
        self.total_weight = len


class SegmentScanReporter(ScanReporter):
    def __init__(self, sentence):
        self.sentence = sentence
        self.emit_info = list()
        for i in range(len(sentence)):
            self.emit_info.append(FakeEmitInfo(4))

    def callback(self, pos, weight, len, tag=None):
        if len == 1:
            self.emit_info[pos].need_fake = True
            self.emit_info[pos].weights[0] += weight  # 'S'
        else:
            for i in range(len):
                p = pos - len + 1 + i
                self.emit_info[p].need_fake = True
                self.emit_info[p].total_weight += weight
                if i == 0:
                    self.emit_info[p].weights[1] += weight  # 'B'
                elif i == len - 1:
                    self.emit_info[p].weights[3] += weight  # 'E'
                else:
                    self.emit_info[p].weights[2] += weight  # 'M'
        return False

    def fake_predication(self, predictions, sentence_id):
        s_len = len(self.sentence)
        for i in range(s_len):
            if self.emit_info[i].need_fake:
                predictions[sentence_id][i][0] =\
                    math.log(self.emit_info[i].weights[0] / self.emit_info[i].total_weight)
                predictions[sentence_id][i][1] = \
                    math.log(self.emit_info[i].weights[1] / self.emit_info[i].total_weight)
                predictions[sentence_id][i][2] = \
                    math.log(self.emit_info[i].weights[2] / self.emit_info[i].total_weight)
                predictions[sentence_id][i][3] = \
                    math.log(self.emit_info[i].weights[3] / self.emit_info[i].total_weight)
        return predictions


class TaggingScanReporter(ScanReporter):
    def __init__(self, ustr, tag_to_id):
        self.__sentence_ = ustr
        self.__emit_infos_ = list()
        self.__tag_to_id = tag_to_id
        for i in range(len(ustr)):
            self.__emit_infos_.append(FakeEmitInfo(len(tag_to_id)))

    def callback(self, pos, weight, len, tag=None):
        s_tag_id = self.__tag_to_id["S-" + tag]
        b_tag_id = self.__tag_to_id["B-" + tag]
        m_tag_id = self.__tag_to_id["M-" + tag]
        e_tag_id = self.__tag_to_id["E-" + tag]
        if len == 1:
            self.__emit_infos_[pos].need_fake = True
            self.__emit_infos_[pos].tag_ids.append(s_tag_id)
            self.__emit_infos_[pos].weights[s_tag_id] += weight  # 'S'
        else:
            for i in range(len):
                p = pos - len + 1 + i
                self.__emit_infos_[p].need_fake = True
                self.__emit_infos_[p].total_weight += weight
                if i == 0:
                    self.__emit_infos_[p].weights[b_tag_id] += weight  # 'B'
                    self.__emit_infos_[p].tag_ids.append(b_tag_id)
                elif i == len - 1:
                    self.__emit_infos_[p].weights[e_tag_id] += weight  # 'E'
                    self.__emit_infos_[p].tag_ids.append(e_tag_id)
                else:
                    self.__emit_infos_[p].weights[m_tag_id] += weight  # 'M'
                    self.__emit_infos_[p].tag_ids.append(m_tag_id)
        return False

    def fake_predication(self, predictions, sentence_id):
        s_len = len(self.__sentence_)
        for i in range(s_len):
            if self.__emit_infos_[i].need_fake:
                for tag_id in range(len(self.__emit_infos_[i].weights)):
                    predictions[sentence_id][i][tag_id] =\
                        math.log(self.__emit_infos_[i].weights[tag_id] / self.__emit_infos_[i].total_weight)

        return predictions


