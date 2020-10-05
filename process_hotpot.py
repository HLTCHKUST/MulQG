import config
import json
import pickle
import time
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import nltk

from data_utils import make_embedding, make_conll_format, make_vocab, convert_idx

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

def make_sent_dataset():
    embedding_file = "./glove/glove.840B.300d.txt"
    embedding = "./hotpot/embedding.pkl"
    src_word2idx_file = "./hotpot/word2idx.pkl"

    train_hotpot = "./hotpot/data/hotpot_train_v1.1.json"
    dev_hotpot = "./hotpot/data/hotpot_dev_distractor_v1.json"

    train_src_file = "./hotpot-sent/para-train.txt"
    train_trg_file = "./hotpot-sent/tgt-train.txt"
    dev_src_file = "./hotpot-sent/para-dev.txt"
    dev_trg_file = "./hotpot-sent/tgt-dev.txt"

    test_src_file = "./hotpot-sent/para-test.txt"
    test_trg_file = "./hotpot-sent/tgt-test.txt"

    # pre-process training data
    train_examples, counter = process_file(train_hotpot, "sent")
    make_conll_format(train_examples, train_src_file, train_trg_file)
    word2idx = make_vocab_from_hotpot(src_word2idx_file, counter, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)

    # split dev into dev and test
    dev_test_examples, _ = process_file(dev_hotpot, "sent")
    # random.shuffle(dev_test_examples)
    num_dev = len(dev_test_examples) // 2
    dev_examples = dev_test_examples[:num_dev]
    test_examples = dev_test_examples[num_dev:]
    make_conll_format(dev_examples, dev_src_file, dev_trg_file)
    make_conll_format(test_examples, test_src_file, test_trg_file)


def make_para_dataset():
    embedding_file = "./glove/glove.840B.300d.txt"
    embedding = "./hotpot/embedding.pkl"
    src_word2idx_file = "./hotpot/word2idx.pkl"

    train_hotpot = "./hotpot/data/hotpot_train_v1.1.json"
    dev_hotpot = "./hotpot/data/hotpot_dev_distractor_v1.json"

    train_src_file = "./hotpot/para-train.txt"
    train_trg_file = "./hotpot/tgt-train.txt"
    dev_src_file = "./hotpot/para-dev.txt"
    dev_trg_file = "./hotpot/tgt-dev.txt"

    test_src_file = "./hotpot/para-test.txt"
    test_trg_file = "./hotpot/tgt-test.txt"

    # pre-process training data
    train_examples, counter = process_file(train_hotpot, "para")
    make_conll_format(train_examples, train_src_file, train_trg_file)
    word2idx = make_vocab_from_hotpot(src_word2idx_file, counter, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)

    # split dev into dev and test
    dev_test_examples, _ = process_file(dev_hotpot, "para")
    # random.shuffle(dev_test_examples)
    num_dev = len(dev_test_examples) // 2
    dev_examples = dev_test_examples[:num_dev]
    test_examples = dev_test_examples[num_dev:]
    make_conll_format(dev_examples, dev_src_file, dev_trg_file)
    make_conll_format(test_examples, test_src_file, test_trg_file)


def make_vocab_from_hotpot(output_file, counter, max_vocab_size):
    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3

    for idx, (token, freq) in enumerate(sorted_vocab, start=4):
        if len(word2idx) == max_vocab_size:
            break
        word2idx[token] = idx
    with open(output_file, "wb") as f:
        pickle.dump(word2idx, f)

    return word2idx


def retrieve_start_end(context_tokens, answer_tokens):
    start_idx = -1
    end_idx = -1
    context_len = len(context_tokens)
    answer_len = len(answer_tokens)

    for idx, token in enumerate(context_tokens):
        if token==answer_tokens[0]:
            context_cand = " ".join(context_tokens[idx:idx+answer_len])    
            answer_cand = " ".join(answer_tokens)
            if context_cand == answer_cand or context_cand[:-1].strip() == answer_cand.strip():
                start_idx = idx
                end_idx = idx + answer_len -1
            if answer_cand[-1]==".":
                if context_cand[:-1].strip() == answer_cand[:-1].strip():
                    start_idx = idx
                    end_idx = idx + answer_len -1
            
            if (not start_idx>=0) or (not end_idx>=0):
                if answer_tokens[-1]==".":
                    context_cand2 = " ".join(context_tokens[idx:idx+answer_len-1])
                    if context_cand2 == answer_cand or context_cand2[:-1].strip() == answer_cand.strip():
                        start_idx = idx
                        end_idx = idx + answer_len -2
                    if answer_cand[-1]==".":
                        if context_cand2[:-1].strip() == answer_cand[:-1].strip():
                            start_idx = idx
                            end_idx = idx + answer_len -2
        elif token==answer_tokens[0]+"." and answer_len==1:
            start_idx = idx
            end_idx = idx
        if idx == context_len-answer_len:
            break
    
    if (not start_idx>=0) or (not end_idx>=0):
        print(context_tokens)
        print(answer_tokens)
        
    # assert start_idx>=0
    # assert end_idx>=0
    return start_idx, end_idx

def retrieve_answer_start(context, answer):
    start_idx = context.find(answer)
    return start_idx


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def process_file(file_name, mode):
    counter = defaultdict(lambda: 0)
    examples = list()
    no_answer = 0

    with open(file_name, "r") as f:
        data = json.load(f)

    for i, sample in tqdm(enumerate(data), total=len(data)):
        question_type = sample["type"]
        if question_type=="comparison":
            continue

        ques = sample["question"].replace("''", '" ').replace("``", '" ').lower()
        ques_tokens = word_tokenize(ques)
        for token in ques_tokens:
            counter[token] += 1


        answer_text = sample["answer"].replace("''", '" ').replace("``", '" ').lower()
        answer_tokens = word_tokenize(answer_text)
        answer_texts = [answer_text]

        raw_context = sample["context"]
        supporting_facts = sample["supporting_facts"]

        titles = []
        sents = []
        for item in supporting_facts:
            titles.append(item[0])
            sents.append(item[1])

        contexts = []
        context_tokens = []
        for item in raw_context:
            if item[0] in titles:
                index = titles.index(item[0]) 
                # lower case, tokenize
                para = " ".join(item[1])
                para = para.replace("''", '" ').replace("``", '" ').lower()

                sent = item[1][sents[index]]
                sent = sent.replace("''", '" ').replace("``", '" ').lower()

                tokenized_para = word_tokenize(para)
                tokenized_sent = word_tokenize(sent)

                if mode == "para":
                    contexts.append(para)
                    context_tokens.extend(tokenized_para)
                else:
                    contexts.append(sent)
                    context_tokens.extend(tokenized_sent)

        for token in context_tokens:
            counter[token] += 1

        spans = convert_idx(" ".join(contexts), context_tokens)
        answer_start = retrieve_answer_start(" ".join(contexts), answer_text)
        answer_end = answer_start + len(answer_text)
        
        answer_span = []
        for idx, span in enumerate(spans):
            if not (answer_end <= span[0] or answer_start >= span[1]):
                answer_span.append(idx)
        if len(answer_span) == 0:
            no_answer += 1
            print("no answer", no_answer)
            continue

        y1, y2 = answer_span[0], answer_span[-1]
        # y1, y2 = retrieve_start_end(context_tokens, answer_tokens)
        y1s = [y1]
        y2s = [y2]

        example = {"context_tokens": context_tokens, "ques_tokens": ques_tokens,
                    "y1s": y1s, "y2s": y2s, "answers": answer_texts}
        examples.append(example)

    return examples, counter


if __name__ == "__main__":
    # make_sent_dataset()
    make_para_dataset()