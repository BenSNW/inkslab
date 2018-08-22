# -*- coding: utf-8 -*-

import math


def entropy_of_list(ls):
    """
    给定列表，计算熵值
    The entropy is sum of -p[i] * log(p[i]) for every unique element i in the list, and p[i] is its frequency
    """
    elements = {}
    for e in ls:
        elements[e] = elements.get(e, 0) + 1
    length = float(len(ls))
    return sum(map(lambda v: -v / length * math.log(v / length), elements.values()))


class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avg_dl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []
        self.df = {}
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                if not word in tmp:
                    tmp[word] = 0
                tmp[word] += 1
            self.f.append(tmp)
            for k, v in tmp.items():
                if k not in self.df:
                    self.df[k] = 0
                self.df[k] += 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1) /
                      (self.f[index][word] + self.k1 * (1 - self.b + self.b * d / self.avg_dl)))
        return score

    def sim_all(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores
