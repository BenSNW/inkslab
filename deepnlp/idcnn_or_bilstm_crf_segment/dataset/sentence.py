# -*- coding: utf-8 -*-

class Sentence:
    def __init__(self):
        self.tokens = [] # 词集合
        self.chars = 0
        self.x = []
        self.y = []

    def addToken(self, t):
        self.chars += len(t)
        self.tokens.append(t)

    def clear(self):
        self.tokens = []
        self.x = []
        self.y = []
        self.chars = 0

    # label -1, unknown
    # 0-> 'S'
    # 1-> 'B'
    # 2-> 'M'
    # 3-> 'E'
    def generate_tr_line(self, vob):
        for t in self.tokens:
            if len(t) == 1:
                self.x.append(vob[str(t[0])])
                self.y.append(0) # 'S'，表示单字为词
            else:
                nn = len(t)
                for i in range(nn):
                    self.x.append(vob[str(t[i])])
                    if i == 0:
                        self.y.append(1) # 'B', 表示词的首字
                    elif i == (nn - 1):
                        self.y.append(3) # 'E'，表示词的尾字
                    else:
                        self.y.append(2) # 'M'，表示词的中间

    def generate_line(self):
        for t in self.tokens:
            if len(t) == 1:
                self.x.append(str(t[0]))
                self.y.append('S') # 'S'，表示单字为词
            else:
                nn = len(t)
                for i in range(nn):
                    self.x.append(str(t[i]))
                    if i == 0:
                        self.y.append('B') # 'B', 表示词的首字
                    elif i == (nn - 1):
                        self.y.append('E') # 'E'，表示词的尾字
                    else:
                        self.y.append('M') # 'M'，表示词的中间

