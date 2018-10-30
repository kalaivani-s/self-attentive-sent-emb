import sys,os
import argparse, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import Counter
import pandas as pd
import json
import cPickle as pickle
import numpy as np
import nltk
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from model import SelfAttentiveEncoder

from bokeh.io import show
from bokeh.models import (
  ColumnDataSource,
  HoverTool,
  LinearColorMapper,
  BasicTicker,
  PrintfTickFormatter,
  ColorBar,
  DataRange1d
)
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
import colorcet as cc

def preprocess_batch(batch, labels, vocab, mode='word', author=None):
  batchsz = len(batch)
  if mode == 'word':
    lengths = [len(x.split()) for x in batch]
  else:
    lengths = [len(x) for x in batch]
  max_sentlen = int(np.mean(lengths)+np.std(lengths))
  tokenizer = nltk.tokenize.TweetTokenizer()
  token_mat = np.zeros((batchsz, max_sentlen), dtype= np.int)
  batchlen = []
  valid_idx = []
  for i in xrange(token_mat.shape[0]):
    if mode == 'word':
      text = tokenizer.tokenize(batch[i])
    else:
      text = batch[i]
    if len(text):
      valid_idx.append(i)
    textlen = min(len(text),max_sentlen)
    batchlen.append(textlen)
    for j,token in enumerate(text):
      if j == max_sentlen:
        break
      if token in vocab:
        token_mat[i,j] = vocab.index(token)+1
      else:
        token_mat[i,j] = vocab.index('<unk>')+1
  valid_idx = np.array(valid_idx)
  batchlen = np.array(batchlen)[valid_idx]
  token_mat = token_mat[valid_idx]
  labels = labels[valid_idx]
  if author is not None:
    author = [author[i] for i in valid_idx]
  # Sort batch in decreasing order of sequence length
  indices = np.argsort(-batchlen)
  batchlen = batchlen[indices]
  token_mat = token_mat[indices]
  labels = labels[indices]
  if author is not None:
    author = [author[i] for i in indices]
  return Variable(torch.from_numpy(token_mat).transpose(0,1)), labels, batchlen.tolist(), author

def iterate_minibatches(inputs, targets, batchsize, author=None, shuffle=False):
  assert inputs.shape[0] == targets.shape[0]
  if shuffle:
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
  for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batchsize]
    else:
      excerpt = slice(start_idx, start_idx + batchsize)
    if author is not None:
      auth = author[excerpt]
    else:
      auth = None
    yield inputs[excerpt], targets[excerpt], auth

def gen_minibatch(tokens, labels, batchsz, vocab, mode='word', author=None, shuffle= True):
  for token, label, auth in iterate_minibatches(tokens, labels, batchsz, author, shuffle=shuffle):
    token, label, seqlen, auth = preprocess_batch(token, label, vocab, mode, auth)
    yield token.cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda(), seqlen, auth

def plot_data(X, y, model, batchsz, vocab, exptdir, mode, classes, author=None):
  TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
  linelen = 75
  text_color = '#000000'
  hex_color = '#FF0000'
  correct_plots = []
  incorrect_plots = []

  model.eval()
  num_batches = X.shape[0] // batchsz
  g = gen_minibatch(X, y, batchsz, vocab, mode, author)
  for j in range(1,num_batches+1):
    tokens, labels, batch_seqlen, auth = next(g)
    seqlen, batchsz = tokens.size()
    hidden_state = model.init_hidden(batchsz)
    output, attn = model(tokens, hidden_state, batch_seqlen)
    y_pred = F.softmax(output,dim=1)
    _, y_pred = torch.max(y_pred,1)
    y_pred = y_pred.data.cpu().numpy().astype(int)
    y_true = labels.data.cpu().numpy().astype(int)
   
    seqlen = min(300,seqlen) # limit text to be plotted 
    text = ' '.join(vocab[tokens.data[i][0]-1] for i in range(seqlen))
    nlines = len(text) // linelen
    title = 'true_'+classes[y_true[0]]+'_pred_'+classes[y_pred[0]]
    print('%d/%d - %s' % (j,num_batches,title))
    p1 = figure(title=title, plot_width=linelen*12, plot_height=nlines*25,
                  tools=TOOLS, toolbar_location='below')
    p1.xaxis.visible = False
    p1.xgrid.visible = False
    p1.yaxis.visible = False
    p1.ygrid.visible = False
     
    shade = []
    words = text.split()
    attn = torch.mean(attn,dim=1)
    factor = 1.0/torch.max(attn.data[0])
    for i in range(seqlen):
      shade.extend([attn.data[0][i] * factor]*len(words[i]))
      shade.append(0.0)
    row = nlines; col = 0
    x = []; y = []; txt = []
    for ch in text[:nlines*linelen]:
      if row > 0 and col % linelen == 0 and col > 0:
        row -= 1
        col = 0
      x.append(col+0.5)
      y.append(row+0.5)
      txt.append(ch)
      col += 1
    shade = shade[:len(txt)]

    source = ColumnDataSource(data=dict(text=txt, x=x, y=y, shade=shade))
    p1.rect(x='x',y='y',width=1,height=1,fill_color=hex_color,line_color=None,alpha='shade',source=source)
    p1.text(x='x',y='y',text='text',text_color=text_color,text_font_size='10pt',
                text_baseline='middle',text_align='center',source=source)
    if y_pred[0] == y_true[0]:
      correct_plots.append(p1)    
    else:
      incorrect_plots.append(p1)
 
  p = column(correct_plots)
  output_file(os.path.join(exptdir,'correct.html'))
  save(p)
  p = column(incorrect_plots)
  output_file(os.path.join(exptdir,'incorrect.html'))
  save(p)

def main(argv):
  parser = argparse.ArgumentParser(description='Structured self-attention')
  parser.add_argument("--input", action="store", help="input path")
  parser.add_argument("--results", default='./results', action="store", help="result path")
  parser.add_argument("--expt", default='self-attn', action="store", help="experiment name")
  parser.add_argument("--wordemb", default='../data/glove/glove.6B.100d.txt', action="store", help="word embedding vectors")
  parser.add_argument("--batchsz", type=int, default=50, action="store", help="batch size")
  parser.add_argument("--nepoch", type=int, default=100, action="store", help="number of epochs")
  parser.add_argument("--embedsz", type=int, default=100, action="store", help="word embedding size")
  parser.add_argument("--hiddensz", type=int, default=300, action="store", help="hidden size")
  parser.add_argument("--nlayers", type=int, default=2, action="store", help="number of layers")
  parser.add_argument("--attnsz", type=int, default=350, action="store", help="attention units d_a")
  parser.add_argument("--attnhops", type=int, default=30, action="store", help="attention hops r")
  parser.add_argument("--fcsize", type=int, default=2000, action="store", help="FC layer size")
  parser.add_argument("--mode", type=str, default='word', action="store", help="input as words or characters")
  parser.add_argument("--attr", type=str, default='gender', action="store", help="attribute to estimate - gender, age_group, lang")
  parser.add_argument("--lr", type=float, default=0.001, action="store", help="learning rate")
  args = vars(parser.parse_args())

  input_path = args['input']
  wordemb_path = args['wordemb']
  resultdir = args['results']
  expt = args['expt']
  batchsz = args['batchsz']
  nepoch = args['nepoch']
  embedsz = args['embedsz']
  hiddensz = args['hiddensz']
  nlayers = args['nlayers']
  attnsz = args['attnsz']
  attnhops = args['attnhops']
  fcsize = args['fcsize']
  mode = args['mode']
  attr = args['attr']
  lr = args['lr']

  seed = 1111
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)

  # Read data
  d = pd.read_csv(os.path.join(input_path,'train.csv'), delimiter='\t')  
  X = d['text'].astype(str)
  y = d[attr].str.lower().astype(str)
  X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, test_size = 0.2, random_state= 42)
  d = pd.read_csv(os.path.join(input_path,'test.csv'), delimiter='\t')  
  X_test = d['text'].values.astype(str)
  y_test = d[attr].str.lower().values.astype(str)

  classes = np.unique(y.values).tolist()
  num_classes = len(classes)
  for i in range(y_train.shape[0]):
    y_train[i] = classes.index(y_train[i])
  for i in range(y_val.shape[0]):
    y_val[i] = classes.index(y_val[i])
  for i in range(y_test.shape[0]):
    y_test[i] = classes.index(y_test[i]) 
  y_train = y_train.astype(int)
  y_val = y_val.astype(int)
  y_test = y_test.astype(int)

  ctr = Counter(y_train)
  wts = np.array([0.0]*len(ctr.keys()))
  for i in range(len(ctr)):
    wts[i] = y_train.shape[0]/float(ctr[i])
  weights = torch.from_numpy(wts).float().cuda()

  # Populate pretrained word embedding vectors
  exptdir = os.path.join(resultdir,expt)
  if not os.path.exists(exptdir):
    os.makedirs(exptdir) 
  if not os.path.exists(os.path.join(exptdir,'wordvec.pth')):
    vocab, vectors = get_vocab(X.values, mode, wordemb_path) 
    torch.save({'vocab': vocab, 'vectors': vectors}, os.path.join(exptdir,'wordvec.pth'))
  else:
    data = torch.load(os.path.join(exptdir,'wordvec.pth'))
    vocab = data['vocab']
    vectors = data['vectors']
  if vectors is not None:
    vectors = torch.from_numpy(vectors).cuda()
    num_tokens, embedsz = vectors.size()
  else:
    num_tokens = len(vocab)

  print('Vocab size = %d' % len(vocab))

  model = SelfAttentiveEncoder(num_tokens, embedsz, hiddensz, nlayers, attnsz, attnhops, fcsize,num_classes,vectors).cuda()

  # Load model at best validation loss and compute test accuracy
  state_files = os.listdir(exptdir)
  state_files = [fname for fname in state_files if fname.startswith('model') and fname.endswith('.pth')]
  best_acc = 0.0
  for fname in state_files:
    acc = float(fname.split('.pth')[0].split('_')[3].split('-')[1])
    if acc > best_acc:
      best_acc = acc
      best_state_file = fname
  print('Loading %s for testing' % best_state_file)
  best_state = torch.load(os.path.join(exptdir, best_state_file))
  model.load_state_dict(best_state['model'])
  
  plot_data(X_test, y_test, model, 1, vocab, exptdir, mode, classes)

if __name__ == "__main__":
  main(sys.argv[1:])
