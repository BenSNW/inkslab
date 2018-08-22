# -*- coding:utf-8 -*-

from tensorflow.contrib.layers.python.layers import initializers

from deepnlp.idcnn_or_bilstm_crf_segment.dataset.dataset import *


class Model(object):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.use_idcnn = args.use_idcnn
        self.c2v = load_w2v(args.word2vec_path, args.embedding_size)
        self.words = tf.Variable(self.c2v, name="words")
        layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        if args.use_idcnn:
            self.model = IDCNNModel(layers, 3, args.num_hidden, args.embedding_size,
                                    args.max_sentence_len, args.num_tags)
        else:
            self.model = BILSTMModel(
                args.num_hidden, args.max_sentence_len, args.num_tags)
        self.trains_params = None
        self.inp = tf.placeholder(tf.int32,
                                  shape=[None, args.max_sentence_len],
                                  name="input_placeholder")

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        pass

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, X, reuse=False):
        word_vectors = tf.nn.embedding_lookup(self.words, X)
        length = self.length(X)

        if self.use_idcnn:
            word_vectors = tf.expand_dims(word_vectors, 1)
            unary_scores = self.model.inference(word_vectors, reuse=reuse)
        else:
            unary_scores = self.model.inference(
                word_vectors, length, reuse=reuse)
        return unary_scores, length

    def loss(self, X, Y):
        P, sequence_length = self.inference(X)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, Y, sequence_length)
        self.loss = tf.reduce_mean(-log_likelihood)

    def test_unary_score(self):
        P, sequence_length = self.inference(self.inp, reuse=True)
        return P, sequence_length

    def optimizer(self, learning_rate):
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def run(self, X, Y):
        self.loss(X, Y)
        self.optimizer(self.learning_rate)

class IDCNNModel:
    def __init__(self,
                 layers,
                 filterWidth,
                 numFilter,
                 embeddingDim,
                 maxSeqLen,
                 numTags,
                 repeatTimes=4):
        self.layers = layers
        self.filter_width = filterWidth
        self.num_filter = numFilter
        self.embedding_dim = embeddingDim
        self.repeat_times = repeatTimes
        self.num_tags = numTags
        self.max_seq_len = maxSeqLen
        self.initializer = initializers.xavier_initializer()

    def inference(self, X, reuse=False):
        with tf.variable_scope("idcnn", reuse=reuse):
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            layerInput = tf.nn.conv2d(X,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])

            finalW = tf.get_variable(
                "finalW",
                shape=[totalWidthForLastDim, self.num_tags],
                initializer=tf.contrib.layers.xavier_initializer())

            finalB = tf.get_variable("finalB",
                                     initializer=tf.constant(
                                         0.001, shape=[self.num_tags]))

            scores = tf.nn.xw_plus_b(finalOut, finalW, finalB, name="scores")
        if reuse:
            scores = tf.reshape(scores, [-1, self.max_seq_len, self.num_tags],
                                name="Reshape_7")
        else:
            scores = tf.reshape(scores, [-1, self.max_seq_len, self.num_tags],
                                name=None)
        return scores

class BILSTMModel:
    def __init__(self,
                 numHidden,
                 maxSeqLen,
                 numTags):
        self.num_hidden = numHidden
        self.num_tags = numTags
        self.max_seq_len = maxSeqLen
        self.initializer = initializers.xavier_initializer()
        self.W = tf.get_variable(
            shape=[numHidden * 2, numTags],
            initializer=self.initializer,
            name="weights",
            regularizer=tf.contrib.layers.l2_regularizer(0.001))
        self.b = tf.Variable(tf.zeros([numTags], name="bias"))

    def inference(self, X, length, reuse=False):
        length_64 = tf.cast(length, tf.int64)
        with tf.variable_scope("bilstm", reuse=reuse):
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.num_hidden, reuse=reuse),
                X,
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.num_hidden,
                                        reuse=reuse),
                inputs=tf.reverse_sequence(X,
                                           length_64,
                                           seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backword")

        backward_output = tf.reverse_sequence(backward_output_,
                                              length_64,
                                              seq_dim=1)

        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.num_hidden * 2])
        if reuse is None or not reuse:
            output = tf.nn.dropout(output, 0.5)

        matricized_unary_scores = tf.matmul(output, self.W) + self.b
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, self.max_seq_len, self.num_tags],
            name="Reshape_7" if reuse else None)
        return unary_scores

