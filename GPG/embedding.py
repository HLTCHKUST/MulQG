import os
import sys
from io import open
import torch
from torch import nn
import numpy as np

from GPG.models.model_utils import init_wt_normal

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, position_embeddings_flag=False):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.bert_input_size, padding_idx=0)
        self.position_embeddings =  None
        self.position_embeddings_flag = position_embeddings_flag
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        if (self.position_embeddings_flag):
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.position_emb_size)
            self.emb_size = config.bert_input_size + config.position_emb_size
        else:
            self.emb_size = config.bert_input_size
        self.LayerNorm = torch.nn.LayerNorm(self.emb_size, eps=config.layer_norm_eps)
        self.LayerNorm_no_position = torch.nn.LayerNorm(config.bert_input_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        words_embeddings = self.word_embeddings(input_ids)
        if torch.isnan(torch.sum(words_embeddings)).item():
            print("words_embeddings is na")
            print(words_embeddings)

        if(self.position_embeddings_flag and (position_ids is not None)):
            # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            if torch.isnan(torch.sum(position_embeddings)).item():
                print(" position_embeddings is na")
                print(position_embeddings)

            embeddings = torch.cat([words_embeddings, position_embeddings], 2)
            embeddings = self.LayerNorm(embeddings)
        else:
            embeddings = words_embeddings
            embeddings = self.LayerNorm_no_position(embeddings)
        embeddings = self.dropout(embeddings)

        if torch.isnan(torch.sum(embeddings)).item():
            print("embedding is na")
            print(embeddings)
        
        emb_dim = self.emb_size

        return embeddings, emb_dim


class GloveEmbeddings(nn.Module):

    def __init__(self, config, vocab):
        super(GloveEmbeddings, self).__init__()
        # self.vocab = vocab
        # self.config = config
        self.emb_dim = config.emb_dim   # 300
        self.position_embeddings = None
        self.position_embeddings_flag = config.position_embeddings_flag
        self.word_embeddings = self.share_embedding(vocab, config.emb_file, self.emb_dim)
        if (self.position_embeddings_flag):
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.position_emb_size)
            # self.position_embeddings.weight.requires_grad = False
            self.emb_dim = self.emb_dim + config.position_emb_size
        self.LayerNorm = torch.nn.LayerNorm(self.emb_dim, eps=config.layer_norm_eps)
        self.LayerNorm_no_position = torch.nn.LayerNorm(config.bert_input_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, position_ids=None):
        words_embeddings = self.word_embeddings(input_ids)

        if position_ids is not None:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = torch.cat([words_embeddings, position_embeddings], 2)
            embeddings = self.LayerNorm(embeddings)
        else:
            embeddings = words_embeddings
            embeddings = self.LayerNorm_no_position(embeddings)     
        embeddings = self.dropout(embeddings)
        emb_dim = self.emb_dim
        return embeddings, emb_dim

    def gen_embeddings(self, vocab, emb_dim, emb_file):
        embeddings = np.random.randn(vocab.voc_size, emb_dim) * 0.01 
        print('Embeddings: %d x %d' % (vocab.voc_size, emb_dim))
        if emb_file is not None:
            print('Loading embedding file: %s' % emb_file)
            pre_trained = 0
            for line in open(emb_file).readlines():
                sp = line.split()
                if(len(sp) == emb_dim + 1):
                    if sp[0] in vocab._word2id:
                        pre_trained += 1
                        embeddings[vocab._word2id[sp[0]]] = [float(x) for x in sp[1:]]
                else:
                    print(sp[0])
            print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.voc_size))
        return embeddings

    def share_embedding(self, vocab, emb_file, emb_dim =300, pretrain=True):
        embedding = nn.Embedding(vocab.voc_size, emb_dim)
        init_wt_normal(embedding.weight)

        if(pretrain):
            pre_embedding = self.gen_embeddings(vocab, emb_dim, emb_file)
            embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
            embedding.weight.requires_grad = False
        return embedding

