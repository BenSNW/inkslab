# -*- coding: utf-8 -*-

# 处理人民日报语料，生成词典
import sys, os
import _pickle as pickle

longLine = 0
totalLine = 0
maxLen = 100

# 处理一个单词标签，如“华尔街/nsf”
def processToken(token, vocab):
    global longLine
    global totalLine
    nn = len(token)
    while nn > 0 and token[nn - 1] != '/':
        nn = nn - 1

    token = token[:nn - 1].strip()
    for u in token:
        vocab.update(u)


# 处理单行文本
def processLine(line, vocab):
    line = line.strip()
    nn = len(line)
    seeLeftB = False
    start = 0
    try:
        for i in range(nn):  # 去掉中括号
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
                            processToken(s, vocab)
                    else:
                        processToken(token, vocab)
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
                    processToken(ss[i], vocab)
                processToken(ss[-1], vocab)
            else:
                processToken(token, vocab)
    except Exception as e:
        pass

def main(argc, argv):
    if argc < 3:
        print("Usage:%s <dir> <output>" % (argv[0]))
        sys.exit(1)

    rootDir = argv[1]
    outFile = argv[2]
    vocab = set()
    out = open(outFile, 'wb')
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
                            processLine(line, vocab)
                        fp.close()



    id_to_char = dict()
    char_to_id = dict()
    char_to_id["<PAD>"] = 0
    char_to_id['<UNK>'] = 1
    id_to_char[0] = "<PAD>"
    id_to_char[1] = '<UNK>'

    for id, c in enumerate(vocab):
        id_to_char[id + 2] = c
        char_to_id[c] = id + 2

    tag_to_id = {"S" : 0, "B" : 1, "M": 2, "E" : 3}
    id_to_tag = {0 : "S", 1 : "B", 2 : "M", 3 : "E"}

    pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], out)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
