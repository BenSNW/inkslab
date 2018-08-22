# -*- coding: utf-8 -*-

import os
from inkslab.python.new_words.new_words import NewWordFinder

if __name__ == '__main__':
    doc = '十四是十四四十是四十，，十四不是四十，，，，四十不是十四'
    cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    # cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    pos_prop_path = os.path.join(cur_dir, 'datas', 'common', 'pos_prop.txt')
    ws = NewWordFinder(doc, pos_prop_path, max_word_len=4, min_freq=0.001, min_cohesion=1.0, min_entropy=0.5, pp=0.01)
    print('\n'.join(map(lambda w: '《%s》 freq:%f,left_entropy:%f,right_entropy:%f,cohesion:%f,pos_prop:%f' % w,
                        ws.new_words)))

