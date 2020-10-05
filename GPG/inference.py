import argparse
import os
from os.path import join
from tqdm import tqdm
import numpy as np
import time
import random
import torch
import json
import pickle

from GPG.models.model_utils import outputids2words, words2sents
from GPG.models.model import GraphPointerGenerator
from GPG.data.vocab import Vocab
from GPG.data.feature import InputFeaturesQG, Example
from GPG.util import utils
from GPG.data.data_helper import DataHelper

class BeamSearcher(object):
    def __init__(self, args):
        self.args = args
        print('loading checkpoint...\n')
        checkpoints = torch.load(self.args.restore)

        with open(self.args.word2idx_file, "rb") as f:
            word2idx = pickle.load(f)
        
        self.tok2idx = word2idx
        self.idx2tok = {idx: tok for tok, idx in self.tok2idx.items()}

        start_time = time.time()
        self.dataloader = DataHelper(gz=True, config=self.args)
        self.args.n_type = self.dataloader.n_type

        torch.backends.cudnn.benchmark = True
        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))

        self.model = GraphPointerGenerator(config=self.args)


        self.gpus = [int(i) for i in self.args.gpus.split(',')]
        self.model = self.model.cuda()
    
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)

        self.model.load_state_dict(checkpoints['model'], strict=True)

        model_name = self.args.restore.split("/")[-1]
        self.output_dir = self.args.output_dir + model_name
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)

    def decode(self):
        self.model.eval()
        multi_ref, candidate, source_qid = [], [], []
        paragraphs = []
        answers = []
        test_dataloader = self.dataloader.test_loader
        test_example_dict = self.dataloader.dev_example_dict
        # test_dataloader = self.dataloader.train_loader
        # test_example_dict = self.dataloader.train_example_dict

        # i = 0
        for batch in tqdm(test_dataloader):
            # i += 1
            # if i > 5:
            #     break
            best_summary = self.model.module.beam_search(batch)
            # best_summary = self.model.beam_sample(batch)
            samples = [int(t) for t in best_summary.tokens[1:]]
            # print(samples)
            decoded_words = outputids2words(samples, self.idx2tok, (batch['context_oovs'][0] if self.args.use_copy else None))
            decoded_words_final = words2sents(decoded_words)
            candidate += [decoded_words_final]

            eos_trg = batch['tgt_idxs_extend'] * batch['tgt_mask'].type_as(batch['tgt_idxs_extend']) 
            target_ids = [int(t.item()) for t in eos_trg[0]]
            reference_ids = outputids2words(target_ids[1:], self.idx2tok, (batch['context_oovs'][0] if self.args.use_copy else None))
            multi_ref += [words2sents(reference_ids)]

            paragraphs += [test_example_dict[batch['ids'][0]].doc_tokens] 
            source_qid += [batch['ids'][0]]
            answers += [test_example_dict[batch['ids'][0]].orig_answer_text] 

        utils.write_result_to_file_QG(source_qid, candidate, multi_ref, self.output_dir)
        utils.write_qid_paragraphs_answers(source_qid, paragraphs, answers, self.output_dir)
        text_result, bleu = utils.eval_multi_bleu(multi_ref, candidate, self.output_dir)
        print(text_result, flush=True)
        print("Best bleu score: %.2f\n" % (bleu))
        return bleu
