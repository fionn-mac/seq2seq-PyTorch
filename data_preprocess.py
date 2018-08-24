from io import open
import unicodedata
import re

from data import Data

class Data_Preprocess(object):
    def __init__(self, max_length):
        self.max_length = max_length
        self.eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )

    def read_langs(self, lang1, lang2, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open('./Datasets/%s-%s.txt' % (lang1, lang2), encoding='utf-8').readlines()

        # Split every line into pairs and normalize
        pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Data instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Data(lang2)
            output_lang = Data(lang1)
        else:
            input_lang = Data(lang1)
            output_lang = Data(lang2)

        return input_lang, output_lang, pairs

    # Turn a Unicode string to plain ASCII, thanks to
    # http://stackoverflow.com/a/518232/2809427
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def clean_pair(self, pair):
        print(pair)
        for text in pair:
            text = sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
            text = sub(r"what's", "what is ", text)
            text = sub(r"\'s", " ", text)
            text = sub(r"\'ve", " have ", text)
            text = sub(r"can't", "cannot ", text)
            text = sub(r"n't", " not ", text)
            text = sub(r"i'm", "i am ", text)
            text = sub(r"\'re", " are ", text)
            text = sub(r"\'d", " would ", text)
            text = sub(r"\'ll", " will ", text)
            text = sub(r",", " ", text)
            text = sub(r"\.", " ", text)
            text = sub(r"!", " ! ", text)
            text = sub(r"\/", " ", text)
            text = sub(r"\^", " ^ ", text)
            text = sub(r"\+", " + ", text)
            text = sub(r"\-", " - ", text)
            text = sub(r"\=", " = ", text)
            text = sub(r"'", " ", text)
            text = sub(r"(\d+)(k)", r"\g<1>000", text)
            text = sub(r":", " : ", text)
            text = sub(r" e g ", " eg ", text)
            text = sub(r" b g ", " bg ", text)
            text = sub(r" u s ", " american ", text)
            text = sub(r"\0s", "0", text)
            text = sub(r" 9 11 ", "911", text)
            text = sub(r"e - mail", "email", text)
            text = sub(r"j k", "jk", text)
            text = sub(r"\s{2,}", " ", text)
        print(pair)

    def filter_pair(self, count, p):
        return len(p[0].split(' ')) < self.max_length and \
               len(p[1].split(' ')) < self.max_length and \
               p[1].startswith(self.eng_prefixes)

    def filter_pairs(self, pairs):
        return [[pair[0].split(' '), pair[1].split(' ')] for count, pair in enumerate(pairs) if self.filter_pair(count, pair)]

    def prepare_data(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self.read_langs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filter_pairs(pairs)

        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])

        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        # Sort data in reverse order of lengths for easy batch processing
        pairs = sorted(pairs, key=lambda l: len(l[0]), reverse=True)
        return input_lang, output_lang, pairs
