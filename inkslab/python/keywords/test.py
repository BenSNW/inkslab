# -*- coding: utf-8 -*-

from inkslab.python.keywords import textrank


def test_textrank(limit):
    rank = textrank.KeywordTextRank(doc)
    rank.solve()

    for w in rank.top_index(limit):
        print(w)

if __name__ == "__main__":
    doc = [["程序员", "英文", "程序", "开发", "维护", "专业", "人员", "程序员", "分为", "程序", "设计",
           "人员", "程序", "编码", "人员", "界限", "特别", "中国", "软件", "人员", "分为", "程序员", "高级",
           "程序员", "系统", "分析员", "项目", "经理"]]

    print("textrank print:")
    test_textrank(5)
