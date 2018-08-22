# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def read_data(batch_size, file_name, max_sentence_len):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(
        value,
        field_delim=' ',
        record_defaults=[[0] for i in range(max_sentence_len * 2)])

    # batch actually reads the file and loads "batch_size" rows in a single
    # tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

def get_batch_data(batch_size, file_name, max_sentence_len):
    whole = read_data(batch_size, file_name, max_sentence_len)
    features = tf.transpose(tf.stack(whole[0:max_sentence_len]))
    label = tf.transpose(tf.stack(whole[max_sentence_len:]))
    return features, label

def do_load_data(path, max_sentence_len):
    x = []
    y = []
    fp = open(path, "r")
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

def load_w2v(path, expectDim):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == expectDim)
    ws = []
    mv = [0 for i in range(dim)]
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

def load_raw(path, max_sentence_len, char_to_id):
    fp = open(path, "r")
    print("load data from:" + path)
    result = dict()
    lines = fp.readlines()
    for id, line in enumerate(lines):
        line_sentence = []
        line = line.strip()
        line_len = len(line)
        if line_len <= max_sentence_len:
            line_ids =  [char_to_id[w] if w in char_to_id else 0 for w in line]
            pad_len = max_sentence_len - line_len
            for i in range(pad_len):
                line_ids.append(0)
            line_sentence.append(line_ids)
            result[id] = line_sentence
        else:
           line_sentence = handle_long_sentence(line, max_sentence_len, char_to_id)
           result[id] = line_sentence

    return result

def handle_long_sentence(line, max_sentence_len, char_to_id):
    result = []
    sentences = line.split("。")
    for sentence in sentences:
        sentence_len = len(sentence)
        if sentence_len <= max_sentence_len:
            sentence_ids =  [char_to_id[w] if w in char_to_id else 0 for w in line]
            pad_len = max_sentence_len - sentence_len
            for i in range(pad_len):
                sentence_ids.append(0)
            result.append(sentence_ids)
        else:
            continue

    return result


def load_raw1(path, max_sentence_len, char_to_id):
    fp = open(path, "r")
    print("load data from:" + path)
    result = []
    lines = fp.readlines()
    for id, line in enumerate(lines):
        line = line.strip()
        line_len = len(line)
        if line_len <= max_sentence_len:
            line_ids =  [char_to_id[w] if w in char_to_id else 0 for w in line]
            pad_len = max_sentence_len - line_len
            for i in range(pad_len):
                line_ids.append(0)
            result.append(line_ids)
        else:
           continue

    return result

