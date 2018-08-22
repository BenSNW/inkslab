# -*- coding: utf-8 -*-

import os
import _pickle as pickle
import argparse
from inkslab.python.dataset.constants import UNK, PAD


class Sentence(object):
    def __init__(self):
        self.tokens = []  # 词集合
        self.tags = []
        self.chars = 0
        self.x = []
        self.y = []

    def add_token(self, t):
        self.chars += len(t)
        self.tokens.append(t)

    def add_tag(self, t):
        self.tags.append(t)

    def clear(self):
        self.tokens = []
        self.tags = []
        self.x = []
        self.y = []
        self.chars = 0

    def generate_tr_line(self, vob):
        for t in self.tokens:
            if len(t) == 1:
                self.x.append(vob[str(t[0])])
                self.y.append(0)  # 'S'，表示单字为词
            else:
                nn = len(t)
                for i in range(nn):
                    self.x.append(vob[str(t[i])])
                    if i == 0:
                        self.y.append(1)  # 'B', 表示词的首字
                    elif i == (nn - 1):
                        self.y.append(3)  # 'E'，表示词的尾字
                    else:
                        self.y.append(2)  # 'M'，表示词的中间

    def generate_line(self, task, tag_to_id):
        for i, t in enumerate(self.tokens):
            if task == "segment":
                if len(t) == 1:
                    self.x.append(str(t[0]))
                    self.y.append('S')  # 'S'，表示单字为词
                else:
                    nn = len(t)
                    for j in range(nn):
                        self.x.append(str(t[j]))
                        if j == 0:
                            self.y.append('B')  # 'B', 表示词的首字
                        elif j == (nn - 1):
                            self.y.append('E')  # 'E'，表示词的尾字
                        else:
                            self.y.append('M')  # 'M'，表示词的中间
            elif task == "tagging":
                if len(t) == 1:
                    self.x.append(str(t[0]))
                    self.y.append('S-' + self.tags[i])  # 'S'，表示单字为词
                else:
                    nn = len(t)
                    for j in range(nn):
                        self.x.append(str(t[j]))
                        if j == 0:
                            self.y.append('B-' + self.tags[i])  # 'B', 表示词的首字
                        elif j == (nn - 1):
                            self.y.append('E-' + self.tags[i])  # 'E'，表示词的尾字
                        else:
                            self.y.append('M-' + self.tags[i])  # 'M'，表示词的中间
            elif task == "ner":
                if len(t) == 1:
                    self.x.append(str(t[0]))
                    self.y.append('S-' + self.tags[i] if 'S-' + self.tags[i] in tag_to_id else 'S-o')  # 'S'，表示单字为词
                else:
                    nn = len(t)
                    for j in range(nn):
                        self.x.append(str(t[j]))
                        if j == 0:  # 'B', 表示词的首字
                            self.y.append('B-' + self.tags[i] if 'B-' + self.tags[i] in tag_to_id else 'B-o')
                        elif j == (nn - 1): # 'E'，表示词的尾字
                            self.y.append('E-' + self.tags[i] if 'E-' + self.tags[i] in tag_to_id else 'E-o')
                        else:  # 'M'，表示词的中间
                            self.y.append('M-' + self.tags[i] if 'M-' + self.tags[i] in tag_to_id else 'M-o')
            else:
                raise Exception("Unsupported this task: %s" % task)


class HandlerPeopleDaily(object):
    def __init__(self, args):
        self._UNK, self._PAD = UNK, PAD
        self.max_sentence_len = args.max_sentence_len
        self.vocab = set()
        self.data_dir = args.data_dir
        self.task_type = args.task_type
        self.tag_to_id = dict()
        self.id_to_tag = dict()
        self.tag_file = args.tag_file
        self.sentence_info = []
        self.data_output_file = args.data_output_file
        self.process_tag()

    def process_token(self, token, sentence=None, end=None):
        nn = len(token)
        while nn > 0 and token[nn - 1] != '/':
            nn = nn - 1

        word = token[:nn - 1].strip()
        tag = token[nn:]
        self.vocab.update(word)

        sentence.add_token(word)
        sentence.add_tag(tag)

        if token == '。' or end:
            sentence.generate_line(self.task_type, self.tag_to_id)
            nn = len(sentence.x)
            line = ''
            for i in range(nn):
                if i > 0:
                    line += " "
                line += str(sentence.x[i])
            for j in range(nn):
                line += " " + str(sentence.y[j])
            if len(line) > 3:
                self.sentence_info.append(line)
                # data_out.write("%s\n" % line)
            sentence.clear()

    # 处理单行文本
    def process_line(self, line):
        line = line.strip()
        nn = len(line)
        see_left_b = False
        start = 0
        sentence = Sentence()
        try:
            for i in range(nn):  # 去掉中括号
                if line[i] == ' ':
                    if not see_left_b:
                        token = line[start:i]
                        if token.startswith('['):
                            token_len = len(token)
                            while token_len > 0 and token[token_len - 1] != ']':
                                token_len = token_len - 1
                            token = token[1:token_len - 1]
                            ss = token.split(' ')
                            for s in ss:
                                self.process_token(s, sentence, False)
                        else:
                            self.process_token(token, sentence, False)
                        start = i + 1
                elif line[i] == '[':
                    see_left_b = True
                elif line[i] == ']':
                    see_left_b = False
            if start < nn:
                token = line[start:]
                if token.startswith('['):
                    token_len = len(token)
                    while token_len > 0 and token[token_len - 1] != ']':
                        token_len = token_len - 1
                    token = token[1:token_len - 1]
                    ss = token.split(' ')
                    ns = len(ss)
                    for i in range(ns - 1):
                        self.process_token(ss[i], sentence, False)
                        self.process_token(ss[-1], sentence, True)
                else:
                    self.process_token(token, sentence, True)
        except Exception as e:
            print("处理sentence出错: " + str(e))
            pass

    def process_tag(self):
        with open(self.tag_file, encoding='u8') as tf:
            lines = tf.readlines()
            if self.task_type == "segment":
                for i, line in enumerate(lines):
                    tag = line.strip()
                    self.tag_to_id[tag] = i
                    self.id_to_tag[i] = tag
            elif self.task_type == "tagging" or self.task_type == "ner":
                for i in range(len(lines) * 4)[::4]:
                    tag = lines[i // 4].strip()
                    self.tag_to_id["S-" + tag] = i
                    self.tag_to_id["B-" + tag] = i + 1
                    self.tag_to_id["M-" + tag] = i + 2
                    self.tag_to_id["E-" + tag] = i + 3
                    self.id_to_tag[i] = "S-" + tag
                    self.id_to_tag[i + 1] = "B-" + tag
                    self.id_to_tag[i + 2] = "M-" + tag
                    self.id_to_tag[i + 3] = "E-" + tag
            else:
                raise Exception("Unsupport this task: %s" % self.task_type)

    def process(self):
        for dirName, subdirList, fileList in os.walk(self.data_dir):
            for subdir in subdirList:
                level1_dir = os.path.join(dirName, subdir)
                print(level1_dir)
                for inDirName, inSubdirList, inFileList in os.walk(level1_dir):
                    for file in inFileList:
                        if file.endswith(".txt"):
                            cur_file = os.path.join(level1_dir, file)
                            fp = open(cur_file, encoding='u8')
                            for line in fp.readlines():
                                line = line.strip()
                                self.process_line(line)
                            fp.close()


class BuildVocab(HandlerPeopleDaily):
    def __init__(self, args):
        HandlerPeopleDaily.__init__(self, args)

        self.id_to_char = dict()
        self.char_to_id = dict()

        self.char_to_id[self._PAD] = 0
        self.char_to_id[self._UNK] = 1
        self.id_to_char[0] = self._PAD
        self.id_to_char[1] = self._UNK
        self.vocab_file = args.vocab_file

    def build_vocab(self):
        self.process()
        for i, c in enumerate(self.vocab):
            self.id_to_char[i + 2] = c
            self.char_to_id[c] = i + 2

        with open(self.vocab_file, 'wb') as out:
            pickle.dump([self.char_to_id, self.id_to_char,
                         self.tag_to_id, self.id_to_tag], out)

    def generate(self):
        out = open(self.data_output_file, "w")
        data = [line.split() for line in self.sentence_info]

        for j, sen in enumerate(data):
            sen_len = len(sen) // 2
            try:
                chars = sen[:sen_len]
                tags = sen[sen_len:]
                char_ids = [self.char_to_id[c] for c in chars]
                tag_ids = [self.tag_to_id[t] for t in tags]
            except:
                print("line %d handle error" % j)
            else:
                if sen_len < self.max_sentence_len:
                    pad_len = self.max_sentence_len - sen_len
                    for i in range(pad_len):
                        char_ids.append(self.char_to_id[self._PAD])
                        tag_ids.append(0)

                line = ''
                for i in range(self.max_sentence_len):
                    if i > 0:
                        line += " "
                    line += str(char_ids[i])
                for j in range(self.max_sentence_len):
                    line += " " + str(tag_ids[j])
                out.write("%s\n" % line)


def main(args):
    build_vocab = BuildVocab(args)
    build_vocab.build_vocab()
    build_vocab.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datas/common/2014",
        help="The path of people daily source file include"
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="datas/segment/segment_vocab",
        help="The file to save vocab info"
    )
    parser.add_argument(
        "--tag_file",
        type=str,
        default="datas/segment/tags.txt",
        help="The file that contain all tags"
    )
    parser.add_argument(
        "--data_output_file",
        type=str,
        default="datas/segment/full_id.txt",
        help="The file to save "
    )
    parser.add_argument(
        "--max_sentence_len",
        type=int,
        default=100,
        help="Max sentence length"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="segment",
        choices=["segment", "tagging", "ner"],
        help="The data file that used to test"
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
