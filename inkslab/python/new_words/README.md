# 新词发现

## 新词发现实现流程

- 1 计算文本片段的出现频率
- 2 计算文本片段的内部凝固程度
- 3 计算文本片段的左右信息熵
- 4 计算文本片段的词缀成词率
- 5 根据以上四个值过滤得到发现的新词

## 新词发现后处理

- 1 去除停用词
- 2 去除字典中已经存在的词
- 3 手动去除不符合语法的词

## 运行

```bash
python test.py
```
或者使用下面的程序调用：

```python
from inkslab.python.new_words.new_words import NewWordFinder

doc = '十四是十四四十是四十，，十四不是四十，，，，四十不是十四'
pos_prop_path = 'pos_prop.txt'

ws = NewWordFinder(doc, pos_prop_path,
  max_word_len=4,
  min_cohesion=1.2,
  min_entropy=0.4,
  pp=0.01)
  
print('\n'.join(map(lambda w: '《%s》 freq:%f,left_entropy:%f,right_entropy:%f,cohesion:%f,pos_prop:%f' % w, ws.new_words)))

```

## 参考文献

http://www.matrix67.com/blog/archives/5044

https://github.com/sing1ee/dict_build