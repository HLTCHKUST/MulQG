from GPG.data.vocab import Vocab

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PADDING]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[OOV]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[END]' # This has a vocab id, which is used at the end of untruncated target sequences

def outputids2words(id_list, vocab, article_oovs):
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i) # might be [UNK]
    except ValueError as e: # w is OOV
      assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      article_oov_idx = i - vocab.voc_size
      try:
        w = article_oovs[article_oov_idx]
      except ValueError as e: # i doesn't correspond to an article oov
        raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
    words.append(w)
  return words

def words2sents(decoded_words):
    result = []
    for w in decoded_words:
        if w == STOP_DECODING:
            break
        elif w == PAD_TOKEN:
            continue
        result.append(w)
    return result

def context2ids(context_words, vocab):
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in context_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is OOV
      if w not in oovs: # Add to list of OOVs
        oovs.append(w)
      oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
      ids.append(vocab.voc_size + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    else:
      ids.append(i)
  return ids, oovs


def question2ids(question_words, vocab, context_oovs):
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in question_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
      if w in context_oovs: # If w is an in-article OOV
        vocab_idx = vocab.voc_size + context_oovs.index(w) # Map to its temporary article OOV number
        ids.append(vocab_idx)
      else: # If w is an out-of-article OOV
        ids.append(unk_id) # Map to the UNK token id
    else:
      ids.append(i)
  return ids
