# -*- coding: utf-8 -*-

import sys, os
import _pickle as pickle
from deepnlp.idcnn_or_bilstm_crf_segment.dataset.sentence import Sentence


def processToken(token, sentence, end, out):
    nn = len(token)
    while nn > 0 and token[nn - 1] != '/':
        nn = nn - 1

    token = token[:nn - 1].strip()

    sentence.addToken(token)

    if token == '。' or end:
        sentence.generate_line()
        nn = len(sentence.x)
        line = ''
        for i in range(nn):
            if i > 0:
                line += " "
            line += str(sentence.x[i])
        for j in range(nn):
            line += " " + str(sentence.y[j])
        if len(line) > 3:
            out.write("%s\n" % line)
        sentence.clear()


def processLine(line, out):
    line = line.strip()
    nn = len(line)
    seeLeftB = False
    start = 0
    sentence = Sentence()
    try:
        for i in range(nn):
            if line[i] == ' ':
                if not seeLeftB:
                    token = line[start:i]
                    if token.startswith('['):
                        tokenLen = len(token)
                        while tokenLen > 0 and token[tokenLen - 1] != ']':
                            tokenLen = tokenLen - 1
                        token = token[1:tokenLen - 1]
                        ss = token.split(' ')
                        for s in ss:
                            processToken(s, sentence, False, out)
                    else:
                        processToken(token, sentence, False, out)
                    start = i + 1
            elif line[i] == '[':
                seeLeftB = True
            elif line[i] == ']':
                seeLeftB = False
        if start < nn:
            token = line[start:]
            if token.startswith('['):
                tokenLen = len(token)
                while tokenLen > 0 and token[tokenLen - 1] != ']':
                    tokenLen = tokenLen - 1
                token = token[1:tokenLen - 1]
                ss = token.split(' ')
                ns = len(ss)
                for i in range(ns - 1):
                    processToken(ss[i], sentence, False, out)
                processToken(ss[-1], sentence, True, out)
            else:
                processToken(token, sentence, True, out)
    except Exception as e:
        pass
    #return sentence

def handle_data(out_file, rootDir):
    out = open(out_file, "w")
    for dirName, subdirList, fileList in os.walk(rootDir):
        for subdir in subdirList:
            level1Dir = os.path.join(dirName, subdir)
            print(level1Dir)
            for inDirName, inSubdirList, inFileList in os.walk(level1Dir):
                for file in inFileList:
                    if file.endswith(".txt"):
                        curFile = os.path.join(level1Dir, file)
                        print("File name: %s" % curFile)
                        fp = open(curFile, "r")
                        for line in fp.readlines():
                            line = line.strip()
                            processLine(line, out)
                        fp.close()
    out.close()


def transform_to_id(out_file, out_id_file, char_to_id, tag_to_id):
    lines = open(out_file, "r").readlines()
    out = open(out_id_file, "w")
    data = [line.split() for line in lines]
    # max_len = max(len(sen) // 2 for sen in data)

    for sen in data:
        sen_len = len(sen) // 2
        chars = [char_to_id[c] for c in sen[: sen_len]]
        tags = [tag_to_id[t] for t in sen[sen_len:]]

        # if sen_len < max_len:
        #     pad_len = max_len - sen_len
        #     for i in range(pad_len):
        #         chars.append(char_to_id['<UNK>'])
        #         tags.append(0)

        line = ''
        for i in range(sen_len):
            if i > 0:
                line += " "
            line += str(chars[i])
        for j in range(sen_len):
            line += " " + str(tags[j])
        out.write("%s\n" % line)

# 生成训练和测试语料

def main(argc, argv):

    if argc < 3:
        print("Usage:%s <dir> <output>" % (argv[0]))
        sys.exit(1)
    vocab_file = argv[1]
    root_dir= argv[2]
    out_file = argv[3]
    out_id_file = argv[4]

    handle_data(out_file, root_dir)

    char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(open(vocab_file, "rb"))
    transform_to_id(out_file, out_id_file , char_to_id, tag_to_id)

if __name__ == '__main__':
    main(len(sys.argv), sys.argv)