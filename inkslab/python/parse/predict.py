# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np

from inkslab.python.models import parse_model
from inkslab.python.dataset import handle_conll_data
from inkslab.python.parse import transition_system
from inkslab.python.parse.transition_system import Configuration
from inkslab.python.dataset.handle_conll_data import Sentence
from inkslab.python.dataset.handle_conll_data import DependencyTree
from inkslab.python.dataset.constants import UNKNOWN


class ModelLoader(object):
    def __init__(self, args):
        self.data_path = args.parse_data_path
        self.model_dir = args.model_dir
        print("NOTICE: Starting new Tensorflow session...")
        self.session = tf.Session()
        print("NOTICE: Initializing nn_parser class...")
        self.var_scope = "parse_var_scope"
        self.model = None
        self._init_model(self.session, args)

        # data utils for parsing:
        vocab_dict_path = os.path.join(self.data_path, "vocab_dict")
        pos_dict_path = os.path.join(self.data_path, "pos_dict")
        label_dict_path = os.path.join(self.data_path, "label_dict")
        feature_tpl_path = os.path.join(self.data_path, "parse.template")

        self.vocab_dict = handle_conll_data.read_vocab(vocab_dict_path)
        self.pos_dict = handle_conll_data.read_vocab(pos_dict_path)
        self.label_dict = handle_conll_data.read_vocab(label_dict_path)
        self.feature_tpl = handle_conll_data.read_template(feature_tpl_path)
        self.arc_labels = transition_system.generate_arcs(self.label_dict)  # Arc Labels: <K,V>   (L(1), 1)

        self.rev_vocab_dict = handle_conll_data.reverse_map(self.vocab_dict)
        self.rev_pos_dict = handle_conll_data.reverse_map(self.pos_dict)
        self.rev_label_dict = handle_conll_data.reverse_map(self.label_dict)
        self.rev_arc_labels = handle_conll_data.reverse_map(self.arc_labels)

        print("NOTICE: vocab_dict size %d" % len(self.vocab_dict))
        print("NOTICE: pos_dict size %d" % len(self.pos_dict))
        print("NOTICE: label_dict size %d" % len(self.label_dict))
        print("NOTICE: arc_labels size %d" % len(self.arc_labels))
        print("NOTICE: feature templates size %d" % len(self.feature_tpl))

    def predict(self, words, tags):
        dep_tree = self._predict(self.session, self.model, words, tags)
        return dep_tree

    def _init_model(self, session, args):
        """Create Parser model and initialize with random or load parameters in session."""

        if self.model is None:  # Create Graph Only once
            self.model = parse_model.NNParserModel(config=args)

        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt:
            model_checkpoint_path = ckpt.model_checkpoint_path
            print("Reading model parameters from %s" % model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(args.model_dir))
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

    def _predict(self, session, model, words, tags):
        ''' Define prediction function of Parsing
        Args:  
            Words : list of Words tokens; 
            Tags: list of POS tags
        Return: 
            Object of DependencyTree, defined in reader
            list: label_names, label name of each input word
        '''
        if len(words) != len(tags):
            print("words list and tags list length is different")
            return
        if len(words) == 0 or len(tags) == 0:
            print("DEBUG: words or tags list is empty")
            return
        # convert words and tags to Sentence object, sentence save the word_id and tag_id; UNKNOWN is '*
        sent = Sentence()
        for w, t in zip(words, tags):
            w_id = self.vocab_dict[w] if w in self.vocab_dict.keys() else self.vocab_dict[UNKNOWN]
            t_id = self.pos_dict[t] if t in self.pos_dict.keys() else self.pos_dict[UNKNOWN]
            sent.add(w_id, t_id)

        tree_idx = self._predict_tree(session, model, sent)

        # New Dependency Tree object to convert label ids to actual labels, and words, pos tags, etc.
        num_token = tree_idx.count()
        tree_label = DependencyTree()  # final tree to store the label value of dependency
        for i in range(0, num_token):
            cur_head = tree_idx.get_head(i + 1)
            cur_label_id = tree_idx.get_label(i + 1)  # 0 is ROOT, actual index start from 1
            cur_label_name = self.rev_label_dict[cur_label_id] if cur_label_id in self.rev_label_dict.keys() else ""
            tree_label.add(words[i], tags[i], cur_head, cur_label_name)
        return tree_label

    def _predict_tree(self, session, model, sent, debug=False):
        """ Generate a greedy decoding parsing of a sent object with list of [word, tag]
            Return: dep parse tree, dep-label is in its id
        """
        target_num = len(self.rev_arc_labels)
        # print ("target number is %d" % target_num)
        config = Configuration(
            sent)  # create a parsing transition arc-standard system configuration, see (chen and manning.2014)
        while not config.is_terminal():
            features = transition_system.get_features(config, self.feature_tpl)  # 1-D list
            X = np.array([features])  # convert 1-D list to ndarray size [1, dim]
            Y = np.array([[0] * target_num])

            # generate greedy prediction of the next_arc_id
            fetches = [model.loss, model.logit]  # fetch out prediction logit
            feed_dict = {}
            feed_dict[model.X] = X
            feed_dict[model.Y] = Y  # dummy input y: 1D list of shape [, target_num]
            _, logit = session.run(fetches, feed_dict)  # not running eval_op, just for prediction

            pred_next_arc_id = int(np.argmax(logit))  # prediction of next arc_idx of 1 of (2*Nl +1)
            pred_next_arc = self.rev_arc_labels[pred_next_arc_id]  # 5-> L(2)   idx -> L(label_id)
            config.step(pred_next_arc)  # Configuration Take One Step
        # When config is terminal, return the final dependence trees object
        dep_tree = config.tree
        return dep_tree


def predict(args):
    parser = ModelLoader(args)
    with open(args.raw_data_path, 'r') as reader, open(args.result_data_path, 'w') as writer:

        writer.writelines("id\tword\tpos\thead\tlabel\n")
        for line in reader.readlines():
            terms = line.split()
            words = [term.split('/')[0] for term in terms]
            tags = [term.split('/')[1] for term in terms]
            # Parsing
            dep_tree = parser.predict(words, tags)
            num_token = dep_tree.count()
            writer.writelines("\n")
            for i in range(num_token):
                cur_id = str(dep_tree.tree[i + 1].id)
                cur_form = str(dep_tree.tree[i + 1].form)
                cur_pos = str(dep_tree.tree[i + 1].pos)
                cur_head = str(dep_tree.tree[i + 1].head)
                cur_label = str(dep_tree.tree[i + 1].deprel)
                writer.writelines(cur_id + "\t")
                writer.writelines(cur_form + "\t")
                writer.writelines(cur_pos + "\t")
                writer.writelines(cur_head + "\t")
                writer.writelines(cur_label + "\t")
                writer.writelines("\n")
