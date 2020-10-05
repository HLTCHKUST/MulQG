import os
import csv
import codecs
import yaml
import time
import numpy as np
import nltk
from nltk.translate import bleu_score
import pickle
import gzip

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    '''读取config文件'''

    return AttrDict(yaml.load(open(path, 'r')))


def read_datas(filename, trans_to_num=False):
    lines = open(filename, 'r').readlines()
    lines = list(map(lambda x: x.split(), lines))
    if trans_to_num:
        lines = [list(map(int, line)) for line in lines]
    return lines


def save_datas(data, filename, trans_to_str=False):
    if trans_to_str:
        data = [list(map(str, line)) for line in data]
    lines = list(map(lambda x: " ".join(x), data))
    with open(filename, 'w') as f:
        f.write("\n".join(lines))


def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s)

    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)

    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def eval_multi_bleu(references, candidate, log_path):
    ref_1, ref_2, ref_3, ref_4 = [], [], [], []
    candidate_new = []

    for cand in candidate:
        # ref_1.append(refs)
        cand_new = []
        for s in cand:
            if str(s) == "[SEP]" or str(s) == "[PAD]":
                break
            cand_new.append(s)
        candidate_new.append(cand_new)
        # if len(refs) > 1:
        #     ref_2.append(refs[1])
        # else:
        #     ref_2.append([])
        # if len(refs) > 2:
        #     ref_3.append(refs[2])
        # else:
        #     ref_3.append([])
        # if len(refs) > 3:
        #     ref_4.append(refs[3])
        # else:
        #     ref_4.append([])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')   #log_path ends with "/"
    ref_file_1 = log_path + 'reference_1.txt'
    # ref_file_2 = log_path + 'reference_2.txt'
    # ref_file_3 = log_path + 'reference_3.txt'
    # ref_file_4 = log_path + 'reference_4.txt'
    cand_file = log_path + 'candidate.txt'
    print(cand_file)
    print(ref_file_1)
    with open(ref_file_1, 'w') as f:
        for ref in references:
            f.write((" ".join(str(s) for s in ref) + '\n'))


    # with open(ref_file_2, 'w') as f:
    #     for s in ref_2:
    #         f.write(" ".join(s) + '\n')
    # with open(ref_file_3, 'w') as f:
    #     for s in ref_3:
    #         f.write(" ".join(s) + '\n')
    # with open(ref_file_4, 'w') as f:
    #     for s in ref_4:
    #         f.write(" ".join(s) + '\n')
    with open(cand_file, 'w') as f:
        for cand_new in candidate_new:
            f.write(" ".join(str(s) for s in cand_new) + '\n')
        
    temp = log_path + "result.txt"
    # command = "perl multi-bleu.perl " + ref_file_1 + " " + ref_file_2 + " " + ref_file_3 + " " + ref_file_4 + "<" + cand_file + "> " + temp
    command = "perl /home/ACL2020/GPG/multi-bleu.perl " + ref_file_1 + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    try:
        print("====result is")
        print(result)
        # bleu = float(result.split(',')[0][7:])
        bleu_1 = float(result.split('/')[0][-4:])

    except ValueError:
        bleu_1 = 0
    return result, bleu_1


def eval_bleu(reference, candidate, log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')
    ref_file = log_path + 'reference.txt'
    cand_file = log_path + 'candidate.txt'
    with codecs.open(ref_file, 'w') as f:
        for s in reference:
            f.write(" ".join(s) + '\n')
    with codecs.open(cand_file, 'w') as f:
        for s in candidate:
            f.write(" ".join(s).strip() + '\n')

    temp = log_path + "result.txt"
    command = "perl multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    try:
        bleu = float(result.split(',')[0][7:])
    except ValueError:
        bleu = 0
    return result, bleu


def write_result_to_file(candidates, references, log_path):
    assert len(references) == len(candidates), (len(references), len(candidates))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')
    log_file = log_path + 'observe_result.tsv'
    with open(log_file, 'w') as f:
        for cand, ref in zip(candidates, references):
            f.write("".join(cand).strip() + '\t')
            f.write("".join(e.ori_title).strip() + '\t')
            # f.write(";".join(["".join(sent).strip() for sent in e.ori_content]) + '\t')
            f.write("".join(e.ori_original_content).strip() + '\t')
            # f.write("$$".join(["".join(comment).strip() for comment in e.ori_targets]) + '\t')
            f.write("\n")


def write_result_to_file_QG(source_qid, candidates, references, log_path):
    assert len(references) == len(candidates) == len(source_qid), (len(references), len(candidates), len(source_qid))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')
    log_file = log_path + 'qid_observe_result.tsv'
    print(source_qid[0])
    print(candidates[0])
    print(references[0])
    with open(log_file, 'w') as f:
        for qid, cand, ref in zip(source_qid, candidates, references):
            cand_new = []
            for s in cand:
                if str(s) == "[SEP]" or str(s) == "[PAD]":
                    break
                # print(s)
                cand_new.append(s)
            # print((" ".join(str(s) for s in cand_new) + '\t'))
            # print((" ".join(str(s) for s in ref) + '\t'))
            f.write(qid + '\t')
            f.write(" ".join(str(s) for s in cand_new) + '\t')
            f.write(" ".join(str(s) for s in ref) + '\t')
            # # f.write(";".join(["".join(sent).strip() for sent in e.ori_content]) + '\t')
            # f.write("".join(e.ori_original_content).strip() + '\t')
            # f.write("$$".join(["".join(comment).strip() for comment in e.ori_targets]) + '\t')
            f.write("\n")


def write_qid_paragraphs_answers(qid, paragraphs, answers, log_path):
    assert len(qid) == len(paragraphs) == len(answers), (len(qid), len(paragraphs), len(answers))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = log_path.strip('/')
    log_file_qid_para = log_path + 'qid_para.tsv'
    log_file_qid_answer = log_path + 'qid_answer.tsv'
    with open(log_file_qid_para, 'w') as f:
        for q, p in zip(qid, paragraphs):
            f.write((str(q) + '\t'))
            f.write(" ".join(str(s) for s in p) + '\t')
            f.write("\n")
    with open(log_file_qid_answer, 'w') as f:
        for q, a in zip(qid, answers):
            f.write((str(q) + '\t'))
            f.write((str(a) + '\t'))
            f.write("\n")


def count_entity_num(candidates, tags):
    assert type(candidates) == list and type(tags) == list
    num = 0.
    for c, t in zip(candidates, tags):
        for word in c:
            if word in t:
                num += 1.
    return num / float(len(candidates))


def bow(word_list):
    word_dict = {}
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = 1
        #word_dict[word] += 1
    return word_dict
