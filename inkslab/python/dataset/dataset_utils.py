# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import re
import os
import _pickle as pickle
from inkslab.python.dataset.constants import MIN_AFTER_DEQUEUE


def read_csv_data(batch_size, file_name, max_sentence_len):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, field_delim=' ', record_defaults=[[0] for _ in range(max_sentence_len * 2)])

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def read_tfrecord_data(batch_size, train_path, max_sentence_len):
    file_names = train_path
    file_names = [file_names] if not isinstance(file_names, list) else file_names
    file_queue = tf.train.string_input_producer(file_names)

    reader = tf.TFRecordReader()
    _, record_string = reader.read(file_queue)
    features = {'sentence': tf.VarLenFeature(tf.int64), 'label': tf.VarLenFeature(tf.int64)}
    one_line_example = tf.parse_single_example(record_string, features=features)
    sentence = one_line_example['sentence'].values
    label = one_line_example['label'].values
    sentence.set_shape([max_sentence_len])
    label.set_shape(1)
    min_after_dequeue = MIN_AFTER_DEQUEUE
    capacity = min_after_dequeue + 2 * batch_size
    x_batch, y_batch = tf.train.shuffle_batch([sentence, label],
                                              batch_size=batch_size,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue,
                                              enqueue_many=False)
    return x_batch, y_batch


def generate_batch_data(batch_size, file_name, max_sentence_len):
    whole = read_csv_data(batch_size, file_name, max_sentence_len)
    features = tf.transpose(tf.stack(whole[0:max_sentence_len]))
    label = tf.transpose(tf.stack(whole[max_sentence_len:]))
    return features, label


def do_load_data(path, max_sentence_len):
    x = []
    y = []
    fp = open(path, 'r', encoding='u8')
    for line in fp.readlines():
        line = line.rstrip()
        if not line:
            continue
        ss = line.split(" ")
        assert (len(ss) == (max_sentence_len * 2))
        lx = []
        ly = []
        for i in range(max_sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i + max_sentence_len]))
        x.append(lx)
        y.append(ly)
    fp.close()
    return np.array(x), np.array(y)


def load_raw(path, char_to_id):
    fp = open(path, 'r', encoding='u8')
    print("load data from:" + path)
    result = []
    lines = fp.readlines()
    for line in lines:
        line = line.replace(" ", "").strip()
        # line_len = len(line)
        line_ids = [char_to_id[w] if w in char_to_id else 0 for w in line]
        result.append(line_ids)

    return result, lines


def result_to_sentence(string, tags, task="segment"):
    item = []
    word_name = ""
    idx = 0
    for char, tag in zip(string, tags):
        if task == "segment":
            if tag == "S":
                item.append(char)
            elif tag == "B":
                word_name += char
            elif tag == "M":
                word_name += char
            elif tag == "E":
                word_name += char
                item.append(word_name)
                word_name = ""
            else:
                word_name = ""
        elif task == "tagging":
            if tag.startswith("S"):
                item.append(char + "_" + tag[2:])
            elif tag.startswith("B"):
                word_name += char
            elif tag.startswith("M"):
                word_name += char
            elif tag.startswith("E"):
                word_name += char + "_" + tag[2:]
                item.append(word_name)
                word_name = ""
            else:
                word_name = ""
        else:
            raise Exception("Not support this task: %s" % task)
        idx += 1
    return item


def index_of_sorted_suffix(doc, max_word_len):
    indexes = []
    length = len(doc)
    for i in range(0, length):
        for j in range(i + 1, min(i + 1 + max_word_len, length + 1)):
            indexes.append((i, j))
    # 根据内容排序
    return sorted(indexes, key=lambda m: doc[m[0]:m[1]])


def gen_sub_parts(string):
    """
    将字符串分成可能的两部分
    给定 "abcd", 生成[("a", "bcd"), ("ab", "cd"), ("abc", "d")]
    """
    length = len(string)
    res = []
    for i in range(1, length):
        res.append((string[0:i], string[i:]))
    return res


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    new_weights = old_weights
    print('Loading pre trained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(open(emb_path, 'r'), encoding='u8'):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[re.sub('\d', '0', word.lower())]
            c_zeros += 1

    print('Loaded %i pre trained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with pre trained embeddings.' % (
        c_found + c_lower + c_zeros, n_words, 100. * (c_found + c_lower + c_zeros) / n_words))

    print('%i found directly, %i after lowercasing, %i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros))

    return new_weights


def load_w2v(path, expect_dim):
    fp = open(path, 'r', encoding='u8')
    print("load pre trained embeddings from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == expect_dim)
    ws = []
    mv = [0 for _ in range(dim)]
    for t in range(total):
        line = fp.readline().strip()
        ss = line.split(" ")
        if len(ss) != (dim + 1): continue
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total

    ws.append(mv)
    fp.close()
    return np.asarray(ws, dtype=np.float32)


def load_pickle_data(path):
    fr = open(path, 'rb', encoding='u8')
    if os.path.exists(path):
        try:
            fr = open(path, 'rb', encoding='u8')
            obj = pickle.load(fr)
            fr.close()
            return obj
        except:
            print("ERROR: Failed to load pickle file: %s..." % path)
            return None
    else:
        print("WARNING: Input Pickle file doesn't exist %s" % path)
        return None


def save_to_pickle_file(data, path):
    with open(path, 'wb', encoding='u8') as out:
        pickle.dump(data, out)
    out.close()
