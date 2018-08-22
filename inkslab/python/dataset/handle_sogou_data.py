# -*- coding: utf-8 -*-

from collections import Counter
import os
import argparse
import csv
import numpy as np

from inkslab.python.dataset.dataset_utils import save_to_pickle_file
from inkslab.python.common.tf_utils import save_tfrecord
from inkslab.python.dataset.constants import UNK, PAD


class BuildVocab(object):
    def __init__(self, opts):
        self._options = opts
        self._UNK, self._PAD = UNK, PAD
        self.spec_words = [self._PAD, self._UNK]
        # 1. load raw data set
        train_sentence_list, train_label_list = self.read_csv(opts.train_data_path)
        print(len(train_sentence_list), len(train_label_list))
        # 2. get the frequency of words and remove word that appears too few times
        freq_words = self.get_freq_word(train_sentence_list)
        # 3. load external word2vec and identify the final freq_word
        self.id2word, self.word2id, self.wordid2vec = self.load_word_info(freq_words)
        # 4. construct the type and label mapping
        self.id2label, self.label2id = self.get_label_info(train_label_list)
        # 5. interpreter
        self.interpreter('train.tfrecord', path=opts.train_data_path, tfrecord=True)
        self.interpreter('val.pkl', opts.val_data_path)
        self.interpreter('test.pkl', opts.test_data_path)
        # 6. save vocob_info
        save_to_pickle_file([self.id2word, self.word2id, self.id2label, self.label2id, self.wordid2vec],
                              os.path.join(opts.train_dir, 'vocab_info.pkl'))

    def read_csv(self, path):
        opts = self._options
        sentence_list, label_list = [], []
        min_length = opts.min_sentence_length
        with open(path, 'r', encoding='u8') as f:
            reader = csv.reader(f)
            index = 0
            for line in reader:
                if index == 0:
                    index += 1
                    continue
                else:
                    sentence, label = line
                    if len(sentence) < min_length:
                        continue
                    sentence_list.append(list(sentence))
                    label_list.append(label)
        f.close()
        print('The number of records in {}'.format(len(label_list)))
        return sentence_list, label_list

    def get_freq_word(self, question_list_list):
        opts = self._options
        raw_word_freq = Counter()
        for question_list in question_list_list:
            for question in question_list:
                raw_word_freq.update(question)
        freq_words = [word for word, freq in raw_word_freq.items() if freq >= opts.min_count]
        print('The total fo word in data set: {}, the number of word: {}, that of removed ones: {}'.
              format(sum(raw_word_freq.values()), len(freq_words), len(raw_word_freq) - len(freq_words)))
        return freq_words

    def load_word_info(self, freq_words):
        # load raw word2vec
        opts = self._options
        raw_word2vec = dict()
        if opts.pre_trained_vec_path is not None:
            with open(opts.pre_trained_vec_path, 'r') as f:
                lines = f.readlines()
                invalid_num = 0
                for line in lines:
                    v = line.strip().split(" ")
                    if len(v) != opts.emb_dim + 1:
                        invalid_num += 1
                    else:
                        raw_word2vec[v[0]] = np.array([float(nv) for nv in v[1:]])
                if invalid_num > 1:
                    print('The number of invalid word vec in {} is {}'.
                          format(opts.pre_trained_vec_path, invalid_num - 1))
            f.close()
            # get id2vec, and word2id and id2word
            common_id2word, wordid2vec, invalid_num = [], [], 0
            for word in freq_words:
                if raw_word2vec.get(word) is not None:
                    common_id2word.append(word)
                    wordid2vec.append(raw_word2vec[word])
                else:
                    invalid_num += 1
            wordid2vec = np.array(wordid2vec)
            print('Invalid number: {}, the shape of id2vec: {}'.
                  format(invalid_num, wordid2vec.shape))
        else:
            common_id2word = freq_words
            wordid2vec = None
        # construct id2word, word2id, word2id, vec_mat
        id2word = self.spec_words + common_id2word
        id2word = {i: w for i, w in enumerate(id2word)}
        word2id = {w: i for i, w in id2word.items()}

        print('The length of word2id: {}'.format(len(word2id)))
        return id2word, word2id, wordid2vec

    @staticmethod
    def get_label_info(label_list):
        label_freq = Counter()
        for label in label_list:
            label_freq.update([label])
        id2label = [t for t in label_freq.keys()]
        id2label = {i: t for i, t in enumerate(id2label)}
        label2id = {t: i for i, t in id2label.items()}
        return id2label, label2id

    def interpreter(self, name, path=None, tfrecord=False):
        opts = self._options
        sentence_list, label_list = self.read_csv(path)
        print('The length of sentence_list and label_list are {} and {} respectively'.
              format(len(sentence_list), len(label_list)))
        num_sentence_list = self.word2id_mapper(sentence_list)
        num_label_list = self.label2id_mapper(label_list)
        print('The length of sentence_list and label_list are {} and {} respectively'.
              format(len(num_sentence_list), len(num_label_list)))
        assert len(num_sentence_list) == len(num_label_list)
        dataset = [num_sentence_list, num_label_list]
        save_tfrecord(dataset, os.path.join(opts.train_dir, name)) if tfrecord else \
            save_to_pickle_file(dataset, os.path.join(opts.train_dir, name))

    def word2id_mapper(self, sentence_list):
        opts = self._options
        word2id = self.word2id
        unk_id, pad_id = word2id.get(self._UNK), word2id.get(self._PAD)
        max_length = opts.max_sentence_length
        num_sentence_list = []
        for sentence in sentence_list:
            num_sentence = []
            for word in sentence:
                num_sentence.append(word2id.get(word, unk_id))
            if len(num_sentence) > max_length:
                num_sentence = num_sentence[:max_length]
            else:
                pad_length = max_length - len(num_sentence)
                num_sentence = num_sentence + [pad_id] * pad_length
            assert len(num_sentence) == max_length
            num_sentence_list.append(num_sentence)
        print('The shape of num_sentence_list is {}'.format(np.array(num_sentence_list).shape))
        return num_sentence_list

    def label2id_mapper(self, label_list):
        label2id = self.label2id
        num_label_list = []
        for label in label_list:
            num_label_list.append(label2id[label])
        return num_label_list


def main(FLAGS):
    FLAGS.use_pre_trained_vec = True
    BuildVocab(FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="raw_data",
        help="The local path or hdfs path of result files"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="train.txt",
        help="The data file that used to train"
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="The data file that used to validation"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="The data file that used to test"
    )
    parser.add_argument(
        "--pre_trained_vec_path",
        type=str,
        default=None,
        help="path of pretrained model_path"
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=100,
        help="The data file that used to test"
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="The min word count"
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=20,
        help="The maximum limitation of sentence that we take into consideration"
    )
    parser.add_argument(
        "--min_sentence_length",
        type=int,
        default=4,
        help="The minimum limitation of sentence that we take into consideration"
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
