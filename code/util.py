import sys,os
import nltk
import numpy as np
import pandas as pd
from collections import Counter

# Get vocabulary and vectors from pretrained vectors
def get_vocab(data, mode='word', wordemb_path=None):
  vocab = []
  tokenizer = nltk.tokenize.TweetTokenizer()

  if mode == 'word':
    for i in range(data.shape[0]):
      text = data[i]
      text = tokenizer.tokenize(text)
      text = [wd.encode('ascii','ignore') for wd in text]
      vocab.extend(text)
  else:
    for i in range(data.shape[0]):
      text = data[i].encode('ascii','ignore')
      vocab.extend(text)
  vocab = Counter(vocab).most_common()
  vocab = [tup[0] for tup in vocab if tup[1] >= 3] # Remove infrequent words/chars

  if os.path.exists(wordemb_path) and mode == 'word':
    wordvec_df = pd.read_csv(wordemb_path, delim_whitespace=True, header=None)
    wordvec_df = wordvec_df.set_index(0)
    wordvec_df = wordvec_df.rename(index={'\\"':'"'})
    wordvec_df.index = wordvec_df.index.astype(str)
    vocab = vocab + ['<unk>']
    vocabsz = len(vocab)
    vecsize = wordvec_df.shape[1]
    vectors = np.random.rand(vocabsz,vecsize)
    for word,row in wordvec_df.iterrows():
      if word in vocab:
        idx = vocab.index(word)
        vectors[idx] = row.values
  else:
    vocab = vocab + ['<unk>']
    vectors = None

  return vocab, vectors
