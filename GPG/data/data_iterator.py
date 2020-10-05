import torch
import numpy as np
from numpy.random import shuffle
from GPG.data.graph_util import create_entity_graph, bfs_step, normalize_answer
from random import choice

IGNORE_INDEX = -100

# we remove the y1 and y2, which is the answer start and end position, but instead, we will put the answer in the query field, and put the query part in the target part
class DataIterator(object):
    def __init__(self, features, example_dict, graph_dict, bsz, mode, entity_limit, n_layers,
                 entity_type_dict=None, sequential=False,):
        self.mode = mode        
        self.bsz = bsz
        # self.features = features
        self.orig_batch_size = bsz
        self.example_dict = example_dict
        self.graph_dict = graph_dict
        self.entity_type_dict = entity_type_dict
        self.sequential = sequential
        self.entity_limit = entity_limit
        self.example_ptr = 0
        self.n_layers = n_layers
        
        features_filtered = []
        for case in features:
            if case.qas_id in self.graph_dict:
                features_filtered.append(case)
        self.features = features_filtered
        if not sequential:
            shuffle(self.features)

        
    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        return self.example_ptr >= len(self.features)

    def __len__(self):
        return int(np.ceil(len(self.features)/self.bsz))


    def __iter__(self):

        # re-filter the features first according to the graph qas_id

        context_idxs = torch.LongTensor(self.bsz, 384)
        context_mask = torch.FloatTensor(self.bsz, 384)
        # segment_idxs = torch.LongTensor(self.bsz, 384)

        # Graph and Mappings
        entity_graphs = torch.Tensor(self.bsz, self.entity_limit, self.entity_limit)
        answer_mapping = torch.Tensor(self.bsz, 384)
        entity_mapping = torch.Tensor(self.bsz, self.entity_limit, 384)

        # pointer generator
        context_batch_extend_vocab = torch.LongTensor(self.bsz, 384)
        tgt_idxs_extend = torch.LongTensor(self.bsz, 50)

        # Target tensor
        tgt_idxs = torch.LongTensor(self.bsz, 50)
        tgt_mask = torch.FloatTensor(self.bsz, 50)

        # answer part
        answer_idxs = torch.LongTensor(self.bsz, 20)
        answer_mask = torch.FloatTensor(self.bsz, 20)

        y1 = torch.LongTensor(self.bsz)
        y2 = torch.LongTensor(self.bsz)
        q_type = torch.LongTensor(self.bsz)


        start_mask = torch.FloatTensor(self.bsz, self.entity_limit)
        start_mask_weight = torch.FloatTensor(self.bsz, self.entity_limit)
        bfs_mask = torch.FloatTensor(self.bsz, self.n_layers, self.entity_limit)
        entity_label = torch.LongTensor(self.bsz)

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            if cur_bsz < self.bsz:
                break

            if self.mode == "decoding":                
                cur_batch_1 = self.features[start_id]
                cur_batch = [cur_batch_1 for _ in range(cur_bsz)]
                self.example_ptr += 1

            else:
                cur_batch = self.features[start_id: start_id + cur_bsz]
                cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)
                self.example_ptr += cur_bsz


            ids = []
            max_sent_cnt = 0
            max_context_oovs = 0 
            max_entity_cnt = 0
            for mapping in [entity_mapping, answer_mapping]:
                mapping.zero_()
            entity_label.fill_(IGNORE_INDEX)
           
            max_context_oovs = int(max([len(fe.doc_input_oovs) for fe in cur_batch]))
            context_oovs = []
            context_oovs = [case.doc_input_oovs for case in cur_batch]
            break_flag = False
            for i in range(len(cur_batch)):
                case = cur_batch[i]
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))

                tgt_idxs[i].copy_(torch.Tensor(case.tgt_question_ids))
                tgt_mask[i].copy_(torch.Tensor(case.tgt_question_mask))

                tgt_idxs_extend[i].copy_(torch.Tensor(case.tgt_question_extend_ids))

                answer_idxs[i].copy_(torch.Tensor(case.answer_input_ids))
                answer_mask[i].copy_(torch.Tensor(case.answer_input_mask))

                context_batch_extend_vocab[i].copy_(torch.Tensor(case.doc_input_extend_vocab))

                tem_graph = self.graph_dict[case.qas_id]
                adj = torch.from_numpy(tem_graph['adj'])
                start_entities = torch.from_numpy(tem_graph['start_entities'])
                entity_graphs[i] = adj
                for l in range(self.n_layers):
                    bfs_mask[i][l].copy_(start_entities)
                    start_entities = bfs_step(start_entities, adj)

                start_mask[i].copy_(start_entities)
                start_mask_weight[i, :tem_graph['entity_length']] = start_entities.byte().any().float()
                
                if len(case.ans_end_position) == 0:
                    y1[i] = y2[i] = 0
                elif case.ans_end_position[0] < 384:
                    if case.ans_start_position[0] < 384:
                        y1[i] = case.ans_start_position[0]
                        y2[i] = case.ans_end_position[0]
                    else:
                        y1[i] = 384 - 1
                        y2[i] = 384 - 1
                else:
                    y1[i] = y2[i] = 0
                q_type[i] = 0

                ids.append(case.qas_id)
                answer = self.example_dict[case.qas_id].orig_answer_text
                for j, entity_span in enumerate(case.entity_spans[:self.entity_limit]):
                    _, _, ent, _  = entity_span
                    if normalize_answer(ent) == normalize_answer(answer):
                        entity_label[i] = j
                        break
                    
                entity_mapping[i] = torch.from_numpy(tem_graph['entity_mapping'])
                max_entity_cnt = max(max_entity_cnt, tem_graph['entity_length'])

            entity_lengths = (entity_mapping[:cur_bsz] > 0).float().sum(dim=2)
            entity_lengths = torch.where((entity_lengths > 0), entity_lengths, torch.ones_like(entity_lengths))
            entity_mask = (entity_mapping > 0).any(2).float()

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            tgt_lengths = (tgt_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_t_len = int(tgt_lengths.max())

            ans_lengths = (answer_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_ans_len = int(ans_lengths.max())

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'tgt_idxs': tgt_idxs[:cur_bsz, :max_t_len].contiguous(),  # the three are newly added, for the target part
                'tgt_mask': tgt_mask[:cur_bsz, :max_t_len].contiguous(),
                'tgt_lens': tgt_lengths,
                'ans_idxs': answer_idxs[:cur_bsz, :max_ans_len].contiguous(),
                'ans_mask': answer_mask[:cur_bsz, :max_ans_len].contiguous(),
                'entity_graphs': entity_graphs[:cur_bsz, :max_entity_cnt, :max_entity_cnt].contiguous(),
                'context_lens': input_lengths,
                'y1': y1[:cur_bsz].contiguous(),
                'y2': y2[:cur_bsz].contiguous(),
                'ids': ids,
                'context_batch_extend_vocab': context_batch_extend_vocab[:cur_bsz, :max_c_len].contiguous(),
                'max_context_oovs': max_context_oovs,     #int
                'context_oovs': context_oovs,
                'tgt_idxs_extend' : tgt_idxs_extend[:cur_bsz, :max_t_len].contiguous(),
                'entity_mapping': entity_mapping[:cur_bsz, :max_entity_cnt, :max_c_len].contiguous(),
                'entity_lens': entity_lengths[:cur_bsz, :max_entity_cnt].contiguous(),
                'entity_mask': entity_mask[:cur_bsz, :max_entity_cnt].contiguous(),
                'entity_label': entity_label[:cur_bsz].contiguous(),
                'start_mask': start_mask[:cur_bsz, :max_entity_cnt].contiguous(),
                'start_mask_weight': start_mask_weight[:cur_bsz, :max_entity_cnt].contiguous(),
                'bfs_mask': bfs_mask[:cur_bsz, :, :max_entity_cnt].contiguous()
            }
