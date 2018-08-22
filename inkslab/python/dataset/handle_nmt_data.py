# -*- coding: utf-8 -*-

import logging
import argparse
import os
import numpy as np
import _pickle as pickle
from inkslab.python.dataset.constants import _START_VOCAB, UNK, SOS, EOS
from inkslab.python.dataset.constants import UNK_ID


class BuildVocab(object):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.doc_filename = os.path.join(args.data_dir, 'train.article.txt')
        self.sum_filename = os.path.join(args.data_dir, 'train.title.txt')

        self.valid_doc_filename = os.path.join(args.data_dir, 'valid.article.3k.txt')
        self.valid_sum_filename = os.path.join(args.data_dir, 'valid.title.3k.txt')

        self.test_path = os.path.join(args.data_dir, 'test.txt')

        self.dict_path = os.path.join(args.data_dir, args.dict_name)

        self.max_doc_vocab = args.max_doc_vocab
        self.max_sum_vocab = args.max_sum_vocab

    @staticmethod
    def create_dict(corpus, max_vocab=None, dict_path=None):
        logging.info("Create dict.")
        counter = {}
        for line in corpus:
            for word in line:
                try:
                    counter[word] += 1
                except:
                    counter[word] = 1

        for mark_t in _START_VOCAB:
            if mark_t in counter:
                del counter[mark_t]
                logging.warning("{} appears in corpus.".format(mark_t))

        counter = list(counter.items())
        counter.sort(key=lambda x: -x[1])
        words = list(map(lambda x: x[0], counter))
        words = [UNK, SOS, EOS] + words
        if max_vocab:
            words = words[:max_vocab]

        tok2id = dict()
        id2tok = dict()
        with open(dict_path, 'w', encoding='u8') as dict_file:
            for idx, tok in enumerate(words):
                print(tok, file=dict_file)
                tok2id[tok] = idx
                id2tok[idx] = tok

        return tok2id, id2tok

    @staticmethod
    def corpus_map2id(data, tok2id):
        ret = []
        unk = 0
        tot = 0
        for doc in data:
            tmp = []
            for word in doc:
                tot += 1
                try:
                    tmp.append(tok2id[word])
                except:
                    tmp.append(UNK_ID)
                    unk += 1
            ret.append(tmp)
        return ret, (tot - unk) / tot

    @staticmethod
    def sen_map2tok(sen, id2tok):
        return list(map(lambda x: id2tok[x], sen))

    def load_data(self):
        logging.info("Load document from {}; summary from {}.".format(
                self.doc_filename, self.sum_filename))

        if os.path.exists(self.dict_path):
            doc_dict, sum_dict, doc_id, sum_id = pickle.load(open(self.dict_path, 'rb'))
        else:
            with open(self.doc_filename) as doc_file:
                docs = doc_file.readlines()
            with open(self.sum_filename) as sum_file:
                sums = sum_file.readlines()
            assert len(docs) == len(sums)
            logging.info("Load {num} pairs of data.".format(num=len(docs)))

            docs = list(map(lambda x: x.split(), docs))
            sums = list(map(lambda x: x.split(), sums))

            doc_dict = self.create_dict(docs, self.max_doc_vocab, os.path.join(self.data_dir, "doc_dict"))

            sum_dict = self.create_dict(sums, self.max_sum_vocab, os.path.join(self.data_dir, "sum_dict"))

            doc_id, cover = self.corpus_map2id(docs, doc_dict[0])
            logging.info("Doc dict covers {:.2f}% words.".format(cover * 100))
            sum_id, cover = self.corpus_map2id(sums, sum_dict[0])
            logging.info("Sum dict covers {:.2f}% words.".format(cover * 100))

            pickle.dump([doc_dict, sum_dict, doc_id, sum_id], open(self.dict_path, 'wb'))

        return doc_id, sum_id, doc_dict, sum_dict

    def load_valid_data(self, doc_dict, sum_dict):
        logging.info("Load validation document from {}; summary from {}.".format(
                self.doc_filename, self.sum_filename))

        with open(self.valid_doc_filename) as doc_file:
            docs = doc_file.readlines()
        with open(self.valid_sum_filename) as sum_file:
            sums = sum_file.readlines()
        assert len(sums) == len(docs)

        logging.info("Load {} validation documents.".format(len(docs)))

        docs = list(map(lambda x: x.split(), docs))
        sums = list(map(lambda x: x.split(), sums))

        doc_id, cover = self.corpus_map2id(docs, doc_dict[0])
        logging.info("Doc dict covers {:.2f}% words on validation set.".format(cover * 100))

        sum_id, cover = self.corpus_map2id(sums, sum_dict[0])
        logging.info("Sum dict covers {:.2f}% words on validation set.".format(cover * 100))

        return doc_id, sum_id

    @staticmethod
    def corpus_preprocess(corpus):
        import re
        ret = []
        for line in corpus:
            x = re.sub('\\d', '#', line)
            ret.append(x)
        return ret

    def load_test_data(self, doc_dict):
        logging.info("Load test document from {doc}.".format(doc=self.test_path))

        with open(self.test_path, encoding='u8') as doc_file:
            docs = doc_file.readlines()
        docs = self.corpus_preprocess(docs)

        logging.info("Load {num} testing documents.".format(num=len(docs)))
        docs = list(map(lambda x: x.split(), docs))

        doc_id, cover = self.corpus_map2id(docs, doc_dict[0])
        logging.info("Doc dict covers {:.2f}% words.".format(cover * 100))
        return doc_id


def main(args):
    build_vocab = BuildVocab(args)
    docid, sumid, doc_dict, sum_dict = build_vocab.load_data()
    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], build_vocab.sen_map2tok(docid[checkid], doc_dict[1]))
    print(sumid[checkid], build_vocab.sen_map2tok(sumid[checkid], sum_dict[1]))

    docid, sumid = build_vocab.load_valid_data(doc_dict, sum_dict)

    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], build_vocab.sen_map2tok(docid[checkid], doc_dict[1]))
    print(sumid[checkid], build_vocab.sen_map2tok(sumid[checkid], sum_dict[1]))

    docid = build_vocab.load_test_data(doc_dict)
    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], build_vocab.sen_map2tok(docid[checkid], doc_dict[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/endy/nlp/inkslab/datas/autosum",
        help="data directory"
    )
    parser.add_argument(
        "--dict_name",
        type=str,
        default="dict.txt",
        help="dict name"
    )
    parser.add_argument(
        "--max_doc_vocab",
        type=int,
        default=30000,
        help="Max doc vocab size"
    )
    parser.add_argument(
        "--max_sum_vocab",
        type=int,
        default=30000,
        help="Max sum vocab size"
    )

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
