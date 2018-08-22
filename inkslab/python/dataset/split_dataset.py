# -*- coding: utf-8 -*-

import sys
import random

'''
python inkslab/python/dataset/split_dataset.py \
    /Users/endy/nlp/inkslab/datas/segment/full_id.txt \
    /Users/endy/nlp/inkslab/datas/segment/train.txt \
    /Users/endy/nlp/inkslab/datas/segment/dev.txt \
    /Users/endy/nlp/inkslab/datas/segment/test.txt
'''
# 将数据集切分为训练集和测试集
def main(argc, argv):
    if argc < 5:
        print("Usage:%s <input> <train> <dev> <test>" % (argv[0]))
        sys.exit(1)
    fp = open(argv[1], 'r', encoding='u8')
    nl = 0
    bad = 0
    test = 0
    dev = 0
    tr_p = open(argv[2], "w", encoding='u8')
    dev_p = open(argv[3], "w", encoding='u8')
    te_p = open(argv[4], "w", encoding='u8')
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        ss = line.split(' ')

        num_v = 0
        for i in range(len(ss)):
            if int(ss[i]) != 0:
                num_v += 1
                if num_v > 2:
                    break
        if num_v <= 2:
            bad += 1
        else:
            r = random.random()
            if r <= 0.02 and test < 8000:
                te_p.write("%s\n" % line)
                test += 1
            elif 0.02 < r <= 0.04 and dev < 8000:
                dev_p.write("%s\n" % line)
                dev += 1
            else:
                tr_p.write("%s\n" % line)
        nl += 1
    fp.close()
    print("got bad:%d" % bad)


if __name__ == '__main__':
    args = ['', 'datas/segment/full_id.txt', 'datas/segment/train.txt',
            'datas/segment/dev.txt', 'datas/segment/test.txt']
    main(len(args), args)
    # main(len(sys.argv), sys.argv)
