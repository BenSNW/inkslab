# -*- coding:utf-8 -*-

import os
import shutil
import codecs

from deepnlp.idcnn_or_bilstm_crf_segment.stand_alone.conlleval import return_report


def test_segment(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "segment_predict.txt")
    with codecs.open(output_file, "w", "utf-8") as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(str(line) for line in to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir(params.log_path):
        os.makedirs(params.log_path)
    if not os.path.isdir(params.vocab_path):
        os.makedirs(params.vocab_path)


def clean_and_make_path(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isdir(params.vocab_path):
        shutil.rmtree(params.vocab_path)
    os.mkdir(params.vocab_path)

    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)
    os.mkdir(params.ckpt_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)
    os.mkdir(params.result_path)

    if os.path.isdir(params.log_path):
        shutil.rmtree(params.log_path)
    os.mkdir(params.log_path)

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isdir(params.config_path):
        shutil.rmtree(params.config_path)
    os.mkdir(params.config_path)

def result_to_sentence(string, tags):
    item = []
    word_name = ""
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item.append(char)
        elif tag[0] == "B":
            word_name += char
        elif tag[0] == "M":
            word_name += char
        elif tag[0] == "E":
            word_name += char
            item.append(word_name)
            word_name = ""
        else:
            word_name = ""
        idx += 1
    return item