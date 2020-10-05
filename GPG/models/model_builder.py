import torch
from torch import nn
from GPG.embedding import GloveEmbeddings, BertEmbeddings
from GPG.models.model_utils import init_wt_normal
import numpy as np
import os
import sys
from io import open

def build_embeddings(config, for_encoder=True, vocab = None):
    """
    Args:
        config: the option in current environment.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """

    if for_encoder:
        position_embeddings_flag = config.position_embeddings_flag
    else:
        position_embeddings_flag = False

    if(config.bert_embedding):
        embedding, emb_dim = BertEmbeddings(config, position_embeddings_flag)   #
    elif(config.glove_embedding and (vocab is not None)):
        embedding, emb_dim = GloveEmbeddings(config, vocab, position_embeddings_flag)
    else:
        embedding = nn.Embedding(config.vocab_size, config.input_dim)   # input_dim = 128
        init_wt_normal(embedding.weight, config.trunc_norm_init_std)
        emb_dim = config.emb_dim

    return embedding, emb_dim

def build_glove_embeddings(config, vocab):
    emb_dim = config.emb_dim   # 300
    position_embeddings = None
    position_embeddings_flag = config.position_embeddings_flag
    word_embeddings = share_embedding(vocab, config.emb_file, emb_dim)
    if position_embeddings_flag:
        position_embeddings = nn.Embedding(config.max_position_embeddings, config.position_emb_size)
        position_embeddings.weight.requires_grad = False
        emb_dim = emb_dim + config.position_emb_size
    # LayerNorm = torch.nn.LayerNorm(emb_dim, eps=config.layer_norm_eps)
    # LayerNorm_no_position = torch.nn.LayerNorm(config.bert_input_size, eps=config.layer_norm_eps)
    # dropout = nn.Dropout(config.hidden_dropout_prob)

    return word_embeddings, position_embeddings

def gen_embeddings(vocab, emb_dim, emb_file):
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

def share_embedding(vocab, emb_file, emb_dim =300, pretrain=True):
    embedding = nn.Embedding(vocab.voc_size, emb_dim)
    init_wt_normal(embedding.weight)

    if(pretrain):
        pre_embedding = gen_embeddings(vocab, emb_dim, emb_file)
        embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.weight.requires_grad = False
    return embedding

