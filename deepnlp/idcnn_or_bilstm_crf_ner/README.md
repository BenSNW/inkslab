# 用IDCNN和CRF做端到端的中文实体识别

对于序列标注来讲，普通CNN有一个劣势，就是卷积之后，末层神经元可能只是得到了原始输入数据中一小块的信息。而对NER来讲，整个句子的每个字都有可能都会对当前需要标注的字做出影响。为了覆盖到输入的全部信息就需要加入更多的卷积层， 导致层数越来越深，参数越来越多，而为了防止过拟合又要加入更多的Dropout之类的正则化，带来更多的超参数，整个模型变得庞大和难以训练。因为CNN这样的劣势，大部分序列标注问题人们还是使用biLSTM之类的网络结构，尽可能使用网络的记忆力记住全句的信息来对单个字做标注。

但这带来的问题是，biLSTM毕竟是一个序列模型，在对GPU并行计算的优化上面不如CNN那么强大。如何能够像CNN那样给GPU提供一个火力全开的战场，而又像LSTM这样用简单的结构记住尽可能多的输入信息呢？

Fisher Yu and Vladlen Koltun 2015 提出了一个dilated CNN的模型，意思是“膨胀的”CNN。想法其实很简单：正常CNN的filter，都是作用在输入矩阵一片连续的位置上，不断sliding做卷积。dilated CNN为这片filter增加了一个dilation width，作用在输入矩阵的时候，会skip掉所有dilation width中间的输入数据；而filter矩阵本身的大小仍然不变，这样filter获取到了更广阔的输入矩阵上的数据，看上去就像是“膨胀”了一般。

具体使用时，dilated width会随着层数的增加而指数增加。这样随着层数的增加，参数数量是线性增加的，而receptive field却是指数增加的，可以很快覆盖到全部的输入数据。

![Dilated CNN](../../images/dilated_cnn.jpg)

对应在文本上，输入是一个一维的向量，每个元素是一个character embedding：

![Dilated CNN block](../../images/dilated_cnn_block.jpg)

我们的模型是4个大的相同结构的Dilated CNN block拼在一起，每个block里面是dilation width为1, 1, 2的三层Dilated卷积层，所以叫做Iterated Dilated CNN。

IDCNN对输入句子的每一个字生成一个logits，这里就和biLSTM模型输出logits之后完全一样，放入CRF Layer，用Viterbi算法解码出标注结果。

# 参考文献

- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) 
- [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)