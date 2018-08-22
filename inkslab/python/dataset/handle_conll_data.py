# -*- coding:utf-8 -*-

from collections import namedtuple
from collections import Counter
import os
import pickle
import argparse
from inkslab.python.dataset.constants import UNKNOWN,  NONE_LABEL


class TaggedWord(object):
    _word = 0
    _tag = 0

    def __init__(self, word, tag):
        self._word, self._tag = word, tag

    @property
    def word(self):
        return self._word

    @word.setter
    def word(self, word):
        self._word = word

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, tag):
        self._tag = tag


class Sentence(object):
    ''' List[TaggedWord]  
        First element of sentence is 'ROOT' token, actual word/tag pair index start from 1
    '''

    def __init__(self):
        self._tokens = []
        self._tokens.append(TaggedWord(0, 0))  # ROOT Node, word index and tag index set to (0,0)

    def add(self, word, tag):
        self._tokens.append(TaggedWord(word, tag))

    def get_word(self, i):  # starting from 0 ROOT node
        return self._tokens[i].word

    def get_tag(self, i):
        return self._tokens[i].tag

    @property
    def tokens(self):
        return self._tokens

    def __repr__(self):
        out = ""
        for t in self._tokens:
            word_str = str(t.word)
            tag_str = str(t.tag)
            out += (word_str + "/" + tag_str + " ")
        return out

    __str__ = __repr__


# See CONLL 2006/2009 data format for details
# ID FORM LEMMA POS PPOS _ HEAD DEPREL _ _
# Not all columns are needed for specific language, see details for your choozen one
Transition = namedtuple("Transition", ["id", "form", "lemma", "pos", "ppos", "head", "deprel"])


class DependencyTree():
    ''' List[Transition]
        First element of sentence is 'ROOT' transition, actual transition index start from 1
        Input sentence: w1,w2,...wn
        ID FORM LEMMA POS PPOS _ HEAD DEPREL _ _
        0  ROOT  ROOT None  None  _ 0 _ _
        1   w1  w1 pos1 pos1 _ 4 l1 _ _
        ...
    '''
    row_id = 0

    def __init__(self):
        self.tree = []  # list of transitions
        self.tree.append(Transition(0, 'ROOT', 'ROOT', None, None, None, None))

    def add(self, word, pos, head, deprel):
        ''' Add import features to tree
        '''
        self.row_id += 1
        self.tree.append(Transition(self.row_id, word, word, pos, pos, head, deprel))

    def set(self, k, word, pos, head, deprel):
        '''Set word , pos, head and label to the kth node, k beginns at 1, 0 is ROOT node
        '''
        if k >= len(self.tree):
            raise Exception("k is out of index of head and label list")
            return
        self.tree[k] = Transition(k, word, word, pos, pos, head, deprel)

    def get_head(self, k):
        '''Return: int, the head word index of the kth word
        '''
        if self.tree[k].head is not None:  # Not None, Might be mistaken ignored as if (head), head might be 0
            return int(self.tree[k].head)
        else:
            return None

    def get_label(self, k):
        '''Return: int, the label index of the kth word 
        '''
        if self.tree[k].deprel is not None:  # Not None
            return int(self.tree[k].deprel)
        else:
            return None

    def get_root(self):
        ''' Get the index of the node, which has head as 'ROOT'
        '''
        for k in range(len(self.tree)):
            if (self.tree[k].head == 0):  # kth node has head as id 0: 'ROOT'
                return k
        return 0

    def count(self):
        '''Return the number of Transition in the tree, index start from 1, excluding ROOT node
        '''
        return (len(self.tree) - 1)

    def __repr__(self):
        out = u""
        for t in self.tree:
            id_str = str(t.id)  # integer
            form_str = t.form if t.form is not None else NONE_LABEL
            lemma_str = t.lemma if t.lemma is not None else NONE_LABEL
            pos_str = t.pos if t.pos is not None else NONE_LABEL
            ppos_str = t.ppos if t.ppos is not None else NONE_LABEL
            head_str = str(t.head) if t.head is not None else NONE_LABEL
            dep_rel_str = t.deprel if t.deprel is not None else NONE_LABEL
            line = u"%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
            id_str, form_str, lemma_str, pos_str, ppos_str, head_str, dep_rel_str)
            out += (line + "\n")
        return out

    __str__ = __repr__


def _read_file(filename):
    ''' Read the transitions into the program, empty line means end of each sentence
    '''
    transitions = []  # list[list[transitions]]
    file = open(filename, encoding='utf-8')
    trans = []  # empty list of tuples
    for line in file:
        if len(line.strip()) == 0:
            transitions.append(trans)
            trans = []
        else:
            items = line.split("\t")
            if len(items) == 9:  # id, form, lemma, pos, ppos, _ , head, deprel, _ , _
                trans.append(Transition(items[0], items[1], items[2], items[3], items[4], items[6], items[7]))

    if len(trans) > 0:  # append the last sentence
        transitions.append(trans)

    count = len(transitions)
    print("Number of sentences read : %d, file: %s" % (count, filename))
    return transitions


def _gen_index_dict(l):
    ''' Generate index dictionary based on the <key> and <the sorted order index>
        UNKNOWN: <UNKNOWN, 0> always has index 0
        Args: dict <K,V> key, count
        Returns: dict <K,V> key, id
    '''
    counter = Counter(l)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    sorted_list, _ = list(zip(*count_pairs))
    idx_dict = dict(zip(sorted_list, range(1, len(sorted_list) + 1)))  # idx 1:n
    idx_dict[UNKNOWN] = 0
    return idx_dict


def _build_dict(transitions):
    vocabs = []  # word
    pos_tags = []  # pos tags
    labels = []  # labels

    for trans in transitions:
        for t in trans:
            vocabs.append(t.form)  # word
            pos_tags.append(t.pos)  # pos
            labels.append(t.deprel)  # dep relations labels
    # Sort the keys and generate index
    vocab_dict = _gen_index_dict(vocabs)
    pos_dict = _gen_index_dict(pos_tags)
    label_dict = _gen_index_dict(labels)
    return vocab_dict, pos_dict, label_dict


def _get_tree(transitions, vocab_dict, pos_dict, label_dict):
    ''' Generate dependency trees, items are the index of vocab, pos and label
        Return: list[DependencyTree]
    '''
    trees = []  # list of dependency trees
    for trans in transitions:
        tree = DependencyTree()
        for t in trans:
            # word_index, pos_index, head_id, label_index
            word_idx = vocab_dict[t.form] if t.form in vocab_dict.keys() else vocab_dict[UNKNOWN]
            pos_idx = pos_dict[t.pos] if t.pos in pos_dict.keys() else pos_dict[UNKNOWN]
            label_idx = label_dict[t.deprel] if t.deprel in label_dict.keys() else label_dict[UNKNOWN]
            tree.add(word_idx, pos_idx, t.head, label_idx)
        trees.append(tree)
    return trees


def _get_sentence(transitions, vocab_dict, pos_dict):
    '''Return list[Sentence]   word/tag pairs, first node is ROOT index 0, actual tokens start at index 1
    '''
    sentences = []
    for trans in transitions:
        sentence = Sentence()
        root_word = vocab_dict[UNKNOWN]
        root_pos = pos_dict[UNKNOWN]
        sentence.add(root_word, root_pos)  # adding Root Node to sentences
        for t in trans:
            word_idx = vocab_dict[t.form] if t.form in vocab_dict.keys() else vocab_dict[UNKNOWN]
            pos_idx = pos_dict[t.pos] if t.pos in pos_dict.keys() else pos_dict[UNKNOWN]
            sentence.add(word_idx, pos_idx)
        sentences.append(sentence)
    return sentences


def _tokenize_data(transitions, vocab_dict, pos_dict, label_dict):
    trees = []  # list of dependency trees
    sentences = []
    count = 0
    for trans in transitions:
        if count % 1000 == 0:
            print("Tokenizing Data Line %d ...." % count)
        tree = DependencyTree()
        sentence = Sentence()
        for t in trans:
            # word_index, pos_index, head_id, label_index
            word_idx = vocab_dict[t.form] if t.form in vocab_dict.keys() else vocab_dict[UNKNOWN]
            pos_idx = pos_dict[t.pos] if t.pos in pos_dict.keys() else pos_dict[UNKNOWN]
            label_idx = label_dict[t.deprel] if t.deprel in label_dict.keys() else label_dict[UNKNOWN]
            tree.add(word_idx, pos_idx, t.head, label_idx)
            sentence.add(word_idx, pos_idx)
        trees.append(tree)
        sentences.append(sentence)
        count += 1
    return sentences, trees


def save_vocab(dict, path):
    # save utf-8 code dictionary
    file = open(path, "w", encoding='utf-8')
    for k, v in dict.items():
        # k is unicode, v is int
        line = k + "\t" + str(v) + "\n"  # unicode
        file.write(line)


def read_vocab(path):
    # read utf-8 code
    file = open(path, encoding='utf-8')
    vocab_dict = {}
    for line in file:
        pair = line.replace("\n", "").split("\t")
        vocab_dict[pair[0]] = int(pair[1])
    return vocab_dict


def reverse_map(dic):
    rev_map = {}
    for k, v in dic.items():
        rev_map[v] = k
    return rev_map


def read_template(path):
    fileTpl = open(path, encoding='utf-8')
    tpls = []
    feature_type_count = {}
    for line in fileTpl:
        line = line.strip()
        if not line.startswith("#"):
            tpls.append(line)
            items = line.split("_")
            feature_type = items[0]
            if feature_type in feature_type_count:
                feature_type_count[feature_type] = feature_type_count[feature_type] + 1
            else:
                feature_type_count[feature_type] = 1
        else:  # Comments
            continue
    # Log
    for k in feature_type_count.keys():
        print("NOTICE: Feature Template Type Count: %s,%d" % (k, feature_type_count[k]))
    return tpls


def build_data(data_path=None):
    train_path = os.path.join(data_path, "train.conll")
    dev_path = os.path.join(data_path, "dev.conll")
    train_transitions = _read_file(train_path)
    dev_transitions = _read_file(dev_path)

    # Generate Dictionary from training dataset
    vocab_dict, pos_dict, label_dict = _build_dict(train_transitions)
    print("NOTICE: Building Vocabulary...")
    print("NOTICE: Vocab dict size %d ..." % len(vocab_dict))
    print("NOTICE: POS dict size %d ..." % len(pos_dict))
    print("NOTICE: Dependency label dict size %d ..." % len(label_dict))

    print("NOTICE: Saving vocab_dict, pos_dict, label_dict, ...")
    save_vocab(vocab_dict, os.path.join(data_path, "vocab_dict"))
    save_vocab(pos_dict, os.path.join(data_path, "pos_dict"))
    save_vocab(label_dict, os.path.join(data_path, "label_dict"))

    train_sents_instance_path = os.path.join(data_path, "train_sents.pkl")
    train_trees_instance_path = os.path.join(data_path, "train_trees.pkl")

    print("Tokenizing Train Data...")
    train_sents, train_trees = _tokenize_data(train_transitions, vocab_dict, pos_dict, label_dict)
    print("NOTICE: Saving Train Sents: %s ..." % train_sents_instance_path)
    save_instance(train_sents_instance_path, train_sents)
    print("NOTICE: Saving Train Trees: %s ..." % train_trees_instance_path)
    save_instance(train_trees_instance_path, train_trees)

    dev_sents_instance_path = os.path.join(data_path, "dev_sents.pkl")
    dev_trees_instance_path = os.path.join(data_path, "dev_trees.pkl")
    print("Tokenizing Dev Data...")
    dev_sents, dev_trees = _tokenize_data(dev_transitions, vocab_dict, pos_dict, label_dict)
    print("NOTICE: Saving Dev Sents: %s ..." % dev_sents_instance_path)
    save_instance(dev_sents_instance_path, dev_sents)
    print("NOTICE: Saving Dev Trees: %s ..." % dev_trees_instance_path)
    save_instance(dev_trees_instance_path, dev_trees)


def load_data(data_path=None, is_train=True):
    """Load raw training and development data from data directory "data_path".
    Args: 
        data_path
    Returns:
        train_sents, train_trees, dev_sents, dev_trees, vocab_dict, pos_dict, label_dict
    """

    # Loading Feature Template
    feature_tpl_path = os.path.join(data_path, "parse.template")
    feature_tpl = read_template(feature_tpl_path)

    vocab_dict = read_vocab(os.path.join(data_path, "vocab_dict"))
    pos_dict = read_vocab(os.path.join(data_path, "pos_dict"))
    label_dict = read_vocab(os.path.join(data_path, "label_dict"))

    if is_train:
        # Generate Training Dataset
        sents_instance_path = os.path.join(data_path, "train_sents.pkl")
        trees_instance_path = os.path.join(data_path, "train_trees.pkl")
        print("NOTICE: Restoring data from path %s" % sents_instance_path)
        print("NOTICE: Restoring data from path %s" % trees_instance_path)
        sents = load_instance(sents_instance_path)
        trees = load_instance(trees_instance_path)
        # Log
        print("NOTICE: Loading Training Dataset Sentences Number %d ..." % len(sents))
    else:
        # Generate Dev Dataset
        sents_instance_path = os.path.join(data_path, "dev_sents.pkl")
        trees_instance_path = os.path.join(data_path, "dev_trees.pkl")
        print("NOTICE: Restoring data from path %s" % sents_instance_path)
        print("NOTICE: Restoring data from path %s" % trees_instance_path)
        sents = load_instance(sents_instance_path)
        trees = load_instance(trees_instance_path)
        # Log
        print("NOTICE: Loading Development Dataset Sentences Number %d ..." % len(sents))
    return sents, trees, vocab_dict, pos_dict, label_dict, feature_tpl


def save_instance(pickle_file, obj):
    fw = open(pickle_file, 'wb')
    pickle.dump(obj, fw)
    fw.close()


def load_instance(pickle_file):
    fr = open(pickle_file, 'rb')
    if os.path.exists(pickle_file):
        try:
            fr = open(pickle_file, 'rb')
            obj = pickle.load(fr)
            fr.close()
            return obj
        except:
            print("ERROR: Failed to load pickle file: %s..." % pickle_file)
            return None
    else:
        print("WARNING: Input Pickle file doesn't exist %s" % pickle_file)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_dir",
        help="The local path or hdfs path of data files"
    )

    FLAGS, unparsed = parser.parse_known_args()
    build_data(FLAGS.data_dir)






