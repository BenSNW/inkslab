# 分词

基于双向GRU-CRF的中文联合分词和词性标注。

改方法综合使用基于字的特征，基于二元词的特征、基于三元词的特征构建神经网络。最后一层采用CRF来决定字的标签。具体结构如下图。

![联合分词和词性标注结构图](../../images/joint_segment_tagger.png)

# 参考

- [Character-based Joint Segmentation and POS Tagging for Chinese using Bidirectional RNN-CRF](https://arxiv.org/pdf/1704.01314.pdf)