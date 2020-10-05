import os
import json
import sys
import numpy as np
import torch
import random
import copy

PADDING = 0
START = 1
END = 2
OOV = 3
MASK = 4
MAX_LENGTH = 100


class Vocab:
    def __init__(self, vocab_file, content_file, vocab_size=30000):
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3, '[MASK]': 4}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]', '[MASK]']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1, '[MASK]': 1}
        if not os.path.exists(vocab_file):
            self.build_vocab(content_file, vocab_file)
        self.load_vocab(vocab_file, vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0

    @staticmethod
    def build_vocab(corpus_file, vocab_file):
        word2count = {}
        for line in open(corpus_file):
            words = line.strip().split()
            for word in words:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word] += 1
        word2count = list(word2count.items())
        word2count.sort(key=lambda k: k[1], reverse=True)
        write = open(vocab_file, 'w')
        for word_pair in word2count:
            write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
        write.close()

    def load_vocab(self, vocab_file, vocab_size):
        for line in open(vocab_file):
            term_ = line.strip().split('\t')
            if len(term_) != 2:
                continue
            word, count = term_
            assert word not in self._word2id
            self._word2id[word] = len(self._word2id)
            self._id2word.append(word)
            self._wordcount[word] = int(count)
            if len(self._word2id) >= vocab_size:
                break
        assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        return self._word2id['[OOV]']

    def sent2id(self, sent, add_start=False, add_end=False):
        result = [self.word2id(word) for word in sent]
        if add_start:
            result = [self._word2id['[START]']] + result

        if add_end:
            result = result + [self._word2id['[END]']]
        return result

    # def id2word(self, word_id):
    #     return self._id2word[word_id]
    
    def id2word(self, word_id):
        if word_id >= self.voc_size:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id2word[word_id]


    def id2sent(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]']:
                continue
            result.append(self._id2word[id])
        return result

