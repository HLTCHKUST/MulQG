from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import gzip
import pickle
from tqdm import tqdm
import nltk

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_tokens,  #
                 answer_tokens,   #
                 question_text,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_tokens = question_tokens
        self.answer_tokens = answer_tokens
        self.question_text = question_text
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


# we modify this feature class so that it will suit our QG tasks
class InputFeaturesQG(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                #  doc_segment_ids,
                 answer_tokens,
                 answer_input_ids,
                 answer_input_mask,
                #  answer_segment_ids,
                 para_spans,
                 sent_spans,
                 entity_spans,
                #  sup_fact_ids,
                #  token_to_orig_map,
                 tgt_question_tokens,    # the target sequence, which is the question tokens
                 tgt_question_ids,
                 tgt_question_mask,
                 tgt_question_extend_ids,
                 doc_input_extend_vocab,
                 doc_input_oovs,                 # add the oov list 
                 ans_start_position,
                 ans_end_position
                #  start_position=None,
                #  end_position=None):
                ):
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        # self.doc_segment_ids = doc_segment_ids

        self.answer_tokens = answer_tokens
        self.answer_input_ids = answer_input_ids
        self.answer_input_mask = answer_input_mask
        # self.answer_segment_ids = answer_segment_ids

        self.para_spans = para_spans
        self.sent_spans = sent_spans
        self.entity_spans = entity_spans
        # self.sup_fact_ids = sup_fact_ids
        # self.ans_type = ans_type

        # self.token_to_orig_map = token_to_orig_map

        self.tgt_question_tokens = tgt_question_tokens
        self.tgt_question_ids = tgt_question_ids
        self.tgt_question_mask = tgt_question_mask
        self.tgt_question_extend_ids = tgt_question_extend_ids
        self.doc_input_extend_vocab = doc_input_extend_vocab
        self.doc_input_oovs = doc_input_oovs
        self.ans_start_position = ans_start_position
        self.ans_end_position = ans_end_position


def clean_entity(entity):
    Type = entity[1]
    Text = entity[0]
    # if Type == "DATE" and ',' in Text:
    #     Text = Text.replace(' ,', ',')
    if '?' in Text:
        Text = Text.split('?')[0]
    # Text = Text.replace("\'\'", "\"")
    # Text = Text.replace("# ", "#")
    Text = Text.replace("''", '" ').replace("``", '" ').lower()
    return Text, Type


def check_in_full_paras(answer, paras):
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans


def read_hotpot_examples_QG(para_file, full_file, entity_file):
    with open(para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    with open(entity_file, 'r', encoding='utf-8') as reader:
        entity_data = json.load(reader)

    cnt = 0
    examples = []
    cn = 0
    for case in tqdm(full_data):
        question_type = case["type"]
        if question_type=="comparison":
            continue
        
        cn += 1
        key = case['_id']
        qas_type = case['type']
        orig_answer_text = case['answer']
        orig_question_text = case['question']

        doc_tokens = []
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []
        ans_start_position, ans_end_position = [], []
        question_tokens = []
        answer_tokens = []

        answer_text_lower = orig_answer_text.replace("''", '" ').replace("``", '" ').lower()
        question_text_lower = orig_question_text.replace("''", '" ').replace("``", '" ').lower()

        answer_tokens = word_tokenize(answer_text_lower)        
        question_tokens = word_tokenize(question_text_lower)
        
        for paragraph in para_data[key]:
            title = paragraph[0]
            sents = paragraph[1]   # sents is still a list of sentences
            if title in entity_data[key]:
                entities = entity_data[key][title]
            else:
                entities = []

            para_start_position = len(doc_tokens)

            for id, sent in enumerate(sents):
                sent = sent.strip()
                sent = sent.replace("''", '" ').replace("``", '" ').lower()
                sent_start_word_id = len(doc_tokens)

                sent_tokens = word_tokenize(sent)
                doc_tokens.extend(sent_tokens)
            
                sent_end_word_id = len(doc_tokens) - 1
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))

                # Find Entity Position 
                sent_tmp = " ".join(sent_tokens)
                sent_spans = convert_idx(sent_tmp, sent_tokens)
                entity_pointer = 0
                find_start_index = 0

                for entity in entities:
                    entity_text, entity_type = clean_entity(entity)
                    ent_tmp = " ".join(word_tokenize(entity_text))
                    entity_start = sent_tmp.find(ent_tmp, find_start_index)
                    entity_span = []
                    if (entity_start != -1):
                        entity_end = entity_start + len(ent_tmp)
                        for idx, span in enumerate(sent_spans):
                            if not (entity_end <= span[0] or entity_start >= span[1]):
                                entity_span.append(idx)
                        if len(entity_span) > 0:
                            ent_start_in_sent, ent_end_in_sent = entity_span[0], entity_span[-1]
                            ent_start_position = ent_start_in_sent + sent_start_word_id
                            ent_end_position = ent_end_in_sent + sent_start_word_id
                            entity_start_end_position.append((ent_start_position, ent_end_position, entity_text, entity_type))
                            entity_pointer += 1
                            find_start_index = entity_start + len(ent_tmp)
                    else:
                        break

                entities = entities[entity_pointer:]

                # Truncate longer document
                if len(doc_tokens) > 384:
                    break
            
            para_end_position = len(doc_tokens) - 1
            para_start_end_position.append((para_start_position, para_end_position, title))

        context = " ".join(doc_tokens)
        
        token_spans = convert_idx(context, doc_tokens)
        answer_start = context.find(" ".join(answer_tokens))
        answer_end = answer_start + len(" ".join(answer_tokens))

        answer_span = []
        for idx, span in enumerate(token_spans):
            if not (answer_end <= span[0] or answer_start >= span[1]):
                answer_span.append(idx)
        if len(answer_span) == 0:
            no_answer += 1
            print("no answer", no_answer)
            continue
        
        y1, y2 = answer_span[0], answer_span[-1]
        ans_start_position.append(y1)
        ans_end_position.append(y2)

        # if(cn < 20):
        #     print("this is for debug")
        #     print(doc_tokens[y1])
        #     print(doc_tokens[y2])
        #     print(answer_tokens)
        #     print(answer_text_lower)

        if len(ans_end_position) >= 1:
            cnt += 1

        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=question_text_lower,
            question_tokens=question_tokens,
            answer_tokens=answer_tokens,
            para_start_end_position=para_start_end_position,
            sent_start_end_position=sent_start_end_position,
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=answer_text_lower,
            start_position=ans_start_position,
            end_position=ans_end_position)
        examples.append(example)
    
    print(cnt)
    print(cn)
    return examples


def context2ids_hotpot(tokens, word2idx, max_seq_length):
    ids = list()
    extended_ids = list()
    oov_lst = list()
    # START and END token is already in tokens lst
    for token in tokens:
        if token in word2idx:
            ids.append(word2idx[token])
            extended_ids.append(word2idx[token])
        else:
            ids.append(word2idx[UNK_TOKEN])
            if token not in oov_lst:
                oov_lst.append(token)
            extended_ids.append(len(word2idx) + oov_lst.index(token))
        if len(ids) == max_seq_length:
            break

    # ids = torch.Tensor(ids)
    # extended_ids = torch.Tensor(extended_ids)
    return ids, extended_ids, oov_lst

def question2ids_hotpot(tokens, word2idx, oov_lst):
    ids = list()
    extended_ids = list()
    ids.append(word2idx[START_TOKEN])
    extended_ids.append(word2idx[START_TOKEN])

    for token in tokens:
        if token in word2idx:
            ids.append(word2idx[token])
            extended_ids.append(word2idx[token])
        else:
            ids.append(word2idx[UNK_TOKEN])
            if token in oov_lst:
                extended_ids.append(len(word2idx) + oov_lst.index(token))
            else:
                extended_ids.append(word2idx[UNK_TOKEN])
    ids.append(word2idx[END_TOKEN])
    extended_ids.append(word2idx[END_TOKEN])

    return ids, extended_ids


def convert_examples_to_features_QG(examples, word2idx, max_seq_length, max_query_length=50, max_answer_length=20):
    features = []

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    for (example_index, example) in enumerate(tqdm(examples)):
        para_spans = []
        entity_spans = []
        sentence_spans = []
        all_doc_tokens = []
        tgt_question_tokens = []
        answer_tokens = []
        ans_start_position = []
        ans_end_position = []

        orig_to_tok_index = []
        orig_to_tok_back_index = []
        tok_to_orig_index = [0] * len(answer_tokens)

        answer_tokens = example.answer_tokens

        if len(answer_tokens) > max_answer_length:
            answer_tokens = answer_tokens[:max_answer_length]
        
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = [token]
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)

        if(len(all_doc_tokens) > max_seq_length):
            all_doc_tokens = all_doc_tokens[:max_seq_length]

        tgt_question_tokens = example.question_tokens 
        if len(tgt_question_tokens) > max_query_length - 2:
            tgt_question_tokens = tgt_question_tokens[:max_query_length - 2]


        for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position):
            s_pos, e_pos = ans_start_pos, ans_end_pos
            if s_pos > max_seq_length - 1:
                s_pos = max_seq_length - 1
            if e_pos > max_seq_length - 1:
                e_pos = max_seq_length - 1
            ans_start_position.append(s_pos)
            ans_end_position.append(e_pos)

        doc_input_oovs = []
        tgt_question_extend_ids = []
        doc_input_extend_vocab = []

        doc_input_ids, doc_input_extend_vocab, doc_input_oovs = context2ids_hotpot(all_doc_tokens, word2idx, max_seq_length)
        tgt_question_ids, tgt_question_extend_ids = question2ids_hotpot(tgt_question_tokens,word2idx,doc_input_oovs)
        answer_input_ids, _, _ =  context2ids_hotpot(answer_tokens, word2idx, max_seq_length)

        for entity_span in example.entity_start_end_position:
            ent_start_position, ent_end_position = entity_span[0], entity_span[1]
            entity_spans.append((ent_start_position, ent_end_position, entity_span[2], entity_span[3]))
        

        for sent_span in example.sent_start_end_position:
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue
            sent_start_position = orig_to_tok_index[sent_span[0]]
            sent_end_position = orig_to_tok_back_index[sent_span[1]]
            sentence_spans.append((sent_start_position, sent_end_position))

        for para_span in example.para_start_end_position:
            if para_span[0] >= len(orig_to_tok_index) or para_span[0] >= para_span[1]:
                continue
            para_start_position = orig_to_tok_index[para_span[0]]
            para_end_position = orig_to_tok_back_index[para_span[1]]
            para_spans.append((para_start_position, para_end_position, para_span[2]))


        doc_input_mask = [1] * len(doc_input_ids)

        # Padding Answer
        answer_input_mask = [1] * len(answer_input_ids)

        # Padding target question
        tgt_question_mask = [1] * len(tgt_question_ids)


        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_input_extend_vocab.append(0)

        while len(answer_input_ids) < max_answer_length:
            answer_input_ids.append(0)
            answer_input_mask.append(0)
        
        while len(tgt_question_ids) < max_query_length:
            tgt_question_ids.append(0)
            tgt_question_mask.append(0)
            tgt_question_extend_ids.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(answer_input_ids) == max_answer_length
        assert len(answer_input_mask) == max_answer_length
        assert len(tgt_question_ids) == max_query_length
        assert len(tgt_question_mask) == max_query_length

        # Dropout out-of-bound span
        entity_spans = entity_spans[:_largest_valid_index(entity_spans, max_seq_length)]
        sentence_spans = sentence_spans[:_largest_valid_index(sentence_spans, max_seq_length)]
        
        if (example_index < 10):
            print("debug ===")
            print(example.qas_id)
            print(doc_input_oovs)
    

        features.append(
            InputFeaturesQG(qas_id=example.qas_id,
                        doc_tokens=all_doc_tokens,
                        doc_input_ids=doc_input_ids,
                        doc_input_mask=doc_input_mask,
                        answer_tokens=answer_tokens,
                        answer_input_ids=answer_input_ids,
                        answer_input_mask=answer_input_mask,
                        para_spans=para_spans,
                        sent_spans=sentence_spans,
                        entity_spans=entity_spans,
                        tgt_question_tokens=tgt_question_tokens,
                        tgt_question_ids=tgt_question_ids,
                        tgt_question_mask=tgt_question_mask,
                        doc_input_oovs =doc_input_oovs,
                        tgt_question_extend_ids = tgt_question_extend_ids,
                        doc_input_extend_vocab = doc_input_extend_vocab,
                        ans_start_position = ans_start_position,
                        ans_end_position = ans_end_position
                        )
        )
    
    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--entity_path", required=True, type=str)
    parser.add_argument("--para_path", required=True, type=str)
    parser.add_argument("--example_output", required=True, type=str)
    parser.add_argument("--feature_output", required=True, type=str)

    # Other parameters
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--full_data", type=str, required=True)
    parser.add_argument("--word2idx_file", default = "/home/multihop_question_generation/hotpot/word2idx.pkl" )
    parser.add_argument("--vocab_size", type=int, default=45000)

    args = parser.parse_args()
    
    with open(args.word2idx_file, "rb") as f:
        word2idx = pickle.load(f)
 
    examples = read_hotpot_examples_QG(para_file=args.para_path, full_file=args.full_data, entity_file=args.entity_path)
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features_QG(examples, word2idx, max_seq_length=args.max_seq_length, max_query_length=50)

    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)











