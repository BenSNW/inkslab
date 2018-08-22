# 关系抽取

关系抽取可以简单理解为一个分类问题：给定两个实体和两个实体共同出现的句子文本，判别两个实体之间的关系。使用CNN或者双向RNN加Attention的深度学习方法被认为是现在关系抽取state of art的解决方案。

## 双向GRU加Dual Attention模型

双向`GRU`加字级别`attention`来自参考文献1。这里将原文的模型结构中的`LSTM`改为`GRU`，且对句子中的每一个中文字符输入为`character embedding`。这样的模型对每一个句子输入做训练，加入字级别的`attention`。

![关系抽取1](../../images/relation_extract1.jpg)

句子级别`attention`参考文献2。原文模型结构图如下，这里将其中对每个句子进行`encoding`的`CNN`模块换成上面的双向`GRU`模型。这样的模型对每一种类别的句子输入做共同训练，加入句子级别的`attention`。

![关系抽取2](../../images/relation_extract2.jpg)

# 参考文献

- [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://anthology.aclweb.org/P16-2034)
- [Neural Relation Extraction with Selective Attention over Instances](http://aclweb.org/anthology/P16-1200)
- [Information-Extraction-Chinese](https://github.com/crownpku/Information-Extraction-Chinese/tree/master/RE_BGRU_2ATT)