import torch
from torch import nn

# PADDING = 0
# START = 1
# END = 2
# OOV = 3
# MASK = 4

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

from torch.nn import CrossEntropyLoss


def cal_position_id(input_ids, ans_start_ids, ans_end_ids, context_mask):
    assert input_ids.size(0) == ans_start_ids.size(0) == ans_end_ids.size(0)
    seq_length = input_ids.size(1)
    position_ids = torch.zeros((input_ids.size(0), seq_length), dtype=torch.long, device=input_ids.device)
    for i in range(input_ids.size(0)):
        for j in range(seq_length):
            if (j <= ans_start_ids[i]):
                position_ids[i,j] = abs(ans_start_ids[i] - j)
            elif (j >= ans_end_ids[i]):
                position_ids[i,j] = abs(ans_end_ids[i] - j)
            else:
                position_ids[i,j] = 0
    position_ids = position_ids * context_mask.type_as(position_ids)

    return position_ids


def init_wt_normal(wt, trunc_norm_init_std = 1e-4):
    wt.data.normal_(std=trunc_norm_init_std).contiguous()

def init_lstm_wt(lstm,rand_unif_init_mag = 0.02):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag).contiguous()
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.).contiguous()
def init_linear_wt(linear, trunc_norm_init_std = 1e-4):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)
def init_wt_unif(wt, rand_unif_init_mag):
    wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)

def cal_position_id(input_ids, ans_start_ids, ans_end_ids, context_mask):
    assert input_ids.size(0) == ans_start_ids.size(0) == ans_end_ids.size(0)
    seq_length = input_ids.size(1)
    # position_ids = Variable(torch.arange(seq_length, dtype=torch.long, device=input_ids.device))
    position_ids = torch.zeros((input_ids.size(0), seq_length), dtype=torch.long, device=input_ids.device)
    for i in range(input_ids.size(0)):
        for j in range(seq_length):
            if (j <= ans_start_ids[i]):
                position_ids[i,j] = abs(ans_start_ids[i] - j)
            elif (j >= ans_end_ids[i]):
                position_ids[i,j] = abs(ans_end_ids[i] - j)
            else:
                position_ids[i,j] = 0
    position_ids = position_ids * context_mask.type_as(position_ids)

    return position_ids

def cal_position_id_2(input_ids, ans_start_ids, ans_end_ids, context_mask):
    # here we use the BOI scheme as described in  https://arxiv.org/pdf/1704.01792.pdf
    assert input_ids.size(0) == ans_start_ids.size(0) == ans_end_ids.size(0)
    seq_length = input_ids.size(1)
    # position_ids = Variable(torch.arange(seq_length, dtype=torch.long, device=input_ids.device))
    position_ids = torch.zeros((input_ids.size(0), seq_length), dtype=torch.long, device=input_ids.device)

    for i in range(input_ids.size(0)):
        # print("the current start and end is {} and {}".format(ans_start_ids[i], ans_end_ids[i]))
        for j in range(seq_length):
            if j == ans_start_ids[i]:
                position_ids[i,j] = 1
            elif (j > ans_start_ids[i]) and (j <= ans_end_ids[i]):
                position_ids[i,j] = 2
            else:
                position_ids[i,j] = 0
    position_ids = position_ids * context_mask.type_as(position_ids)

    return position_ids


def get_weights(size, gain=1.414):
    weights = nn.Parameter(torch.zeros(size=size))
    nn.init.xavier_uniform_(weights, gain=gain)
    return weights

def get_act(act):
    if act.startswith('lrelu'):
        return nn.LeakyReLU(float(act.split(':')[1]))
    elif act == 'relu':
        return nn.ReLU()
    else:
        raise NotImplementedError

def compute_loss_2(hidden_outputs, targets, bfs_mask, masks, num_reason_layers, bfs_clf = True, bfs_lambda = 0.5):
    assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)

    outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))    # [42* 16, 60000]
    targets = targets.contiguous().view(-1)        # [42*16]

    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    loss_1 = loss_func(outputs,targets)

    
    loss3 = 0.0
    binary_criterion = nn.BCEWithLogitsLoss(size_average=True)
 
    if bfs_clf:
        for l in range(num_reason_layers - 1):
            pred_mask = masks[l].view(-1)
            gold_mask = bfs_mask[:, l, :].contiguous().view(-1)
            loss3 += binary_criterion(pred_mask, gold_mask)
    loss_3 = bfs_lambda * loss3

    loss = loss_1 + loss_3
    
    pred = outputs.max(dim=1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(PAD_ID).data).sum()
    num_total = targets.ne(PAD_ID).data.sum()
    loss = loss.div(num_total.float())
    acc = num_correct.float() / num_total.float()
    return loss, acc


def compute_loss(hidden_outputs, targets, attens, coverages, bfs_mask, masks, num_reason_layers, is_coverage = False, bfs_clf = True, bfs_lambda = 0.5):
    assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)

    outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))    # [42* 16, 60000]
    targets = targets.contiguous().view(-1)        # [42*16]

    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    loss_1 = loss_func(outputs,targets)
    # weight = torch.ones(outputs.size(-1))   # 42* 16
    # weight[PADDING] = 0
    # weight[OOV] = 0
    # weight = weight.to(outputs.device)

    # loss = F.nll_loss(torch.log(outputs), targets, weight=weight, reduction='sum')

    coverage_loss = 0
    if is_coverage:
        attens_dist = attens.contiguous().view(-1, attens.size(2))
        coverage_dist = coverages.contiguous().view(-1, coverages.size(2))
        coverage_loss = torch.sum(torch.sum(torch.min(attens_dist, coverage_dist), 1),0)
    
    loss3 = 0.0
    binary_criterion = nn.BCEWithLogitsLoss(size_average=True)
 
    if bfs_clf:
        for l in range(num_reason_layers - 1):
            pred_mask = masks[l].view(-1)
            gold_mask = bfs_mask[:, l, :].contiguous().view(-1)
            loss3 += binary_criterion(pred_mask, gold_mask)
    loss_3 = bfs_lambda * loss3

    loss_2 = 1.0 * coverage_loss
    loss = loss_1 + loss_2 + loss_3
    
    pred = outputs.max(dim=1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(PAD_ID).data).sum()
    num_total = targets.ne(PAD_ID).data.sum()
    loss = loss.div(num_total.float())
    acc = num_correct.float() / num_total.float()
    return loss, acc

def outputids2words(id_list, idx2word, article_oovs=None):
    """
    :param id_list: list of indices
    :param idx2word: dictionary mapping idx to word
    :param article_oovs: list of oov words
    :return: list of words
    """
    words = []
    word = ""
    # print(idx2word)
    for idx in id_list:
        try:
            word = idx2word[idx]
        except KeyError:
            if article_oovs is not None:
                article_oov_idx = idx - len(idx2word)
                try:
                    word = article_oovs[article_oov_idx]
                except IndexError:
                    print("the id is not legal {}".format(idx))
                    print("there's no such a word in extended vocab")
            else:
                word = idx2word[UNK_ID]
        words.append(word)
    return words

def words2sents(decoded_words):
    result = []
    for w in decoded_words:
        if w == END_TOKEN:
            break
        elif w == PAD_TOKEN:
            continue
        result.append(w)
    return result


