# -*- coding: utf-8 -*-

import re

UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"
UNK_ID = 0

UNKNOWN = "*"
DELIMITER = "\s+"  # line delimiter
NONE_LABEL = "NONE"

_START_VOCAB = [UNK, SOS, EOS]


_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# We use a number of buckets for sampling
_buckets = [(30, 10), (50, 20), (70, 20), (100, 20), (200, 30)]

MIN_AFTER_DEQUEUE = 200

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 50000

