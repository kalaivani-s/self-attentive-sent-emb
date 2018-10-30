import sys,os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import nltk
import random
from sklearn.model_selection import train_test_split
from model import SelfAttentiveEncoder
from util import *
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

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

def penalization_term(attn):
  batchsz, attnhops, seqlen = attn.size()
  attnT = attn.transpose(1,2).contiguous()
  loss = torch.bmm(attn,attnT) - Variable(torch.eye(attnhops).repeat(batchsz,1).view(batchsz,attnhops,-1)).cuda()
  loss = torch.norm(loss,2)
  loss = loss*loss
  return loss

def train_minibatch(tokens, labels, batch_seqlen, model, optimizer, criterion):
  # Train model with minibatch and update weights
  model.train()
  seqlen, batchsz = tokens.size()
  hidden_state = model.init_hidden(batchsz)
  optimizer.zero_grad()
  output, attn = model(tokens, hidden_state, batch_seqlen)
  clf_loss = criterion(output, labels)
  attn_loss = penalization_term(attn)
  loss = clf_loss + 0.0005*attn_loss
  loss.backward()
  optimizer.step()

  # Compute accuracy
  y_pred = F.softmax(output,dim=1)
  _, y_pred = torch.max(y_pred,1)
  y_pred = y_pred.data.cpu().numpy().astype(int)
  y_true = labels.data.cpu().numpy().astype(int)
  acc = (y_pred == y_true).sum() / float(y_true.shape[0])

  return loss.data[0], acc

def test_data(X, y, model, criterion, batchsz, vocab, mode, author=None):
  model.eval()
  num_batches = X.shape[0] // batchsz
  g = gen_minibatch(X, y, batchsz, vocab, mode, author)
  test_loss = 0.0
  pred_lbl = []
  true_lbl = []
  author = []
  for j in range(1,num_batches+1):
    tokens, labels, batch_seqlen, auth = next(g)
    seqlen, batchsz = tokens.size()
    hidden_state = model.init_hidden(batchsz)
    output, attn = model(tokens, hidden_state, batch_seqlen)
    clf_loss = criterion(output, labels)
    attn_loss = penalization_term(attn)
    loss = clf_loss + 0.0005*attn_loss
    test_loss += loss.data[0]
    y_pred = F.softmax(output,dim=1)
    _, y_pred = torch.max(y_pred,1)
    y_pred = y_pred.data.cpu().numpy().astype(int)
    y_true = labels.data.cpu().numpy().astype(int)
    pred_lbl.extend(y_pred.tolist())
    true_lbl.extend(y_true.tolist())
    if auth is not None:
      author.extend(auth)
  pred_lbl = np.array(pred_lbl)
  true_lbl = np.array(true_lbl)
  print(confusion_matrix(true_lbl, pred_lbl))
  test_loss = test_loss/num_batches
  test_accuracy = accuracy_score(true_lbl, pred_lbl)
  test_fscore = f1_score(true_lbl, pred_lbl, average='macro')

  return test_loss, test_accuracy, test_fscore

def train_with_early_stopping(X_train, y_train, X_val, y_val, model, optimizer, criterion, vocab, nepoch, batchsz, num_classes, resultdir, mode='word', print_loss_every=50, print_val_loss_every=500):
  num_batches = X_train.shape[0] // batchsz
  min_val_loss = 1e10
  best_epoch = 0
  for epoch in range(1,nepoch+1):
    loss_epoch = []
    accuracy_epoch = []
    g = gen_minibatch(X_train, y_train, batchsz, vocab, mode)
    for j in range(1,num_batches+1):
      if epoch - best_epoch > 10 and best_epoch > 0:
        break # Early stopping
      tokens, labels, batch_seqlen, auth = next(g)
      loss,acc = train_minibatch(tokens, labels, batch_seqlen, model, optimizer, criterion)
      loss_epoch.append(loss)
      accuracy_epoch.append(acc)
      i = (epoch-1)*num_batches + j
      if i % print_loss_every == 0:
        print('Epoch %d minibatch %d: loss = %0.4f, accuracy = %0.4f' % (epoch,i,np.mean(loss_epoch),np.mean(accuracy_epoch)*100))
      # check validation loss every n passes
      if i % print_val_loss_every == 0:
        val_loss, val_acc = test_data(X_val, y_val, model, criterion, batchsz, vocab, mode)
        print('Validation: loss = %0.4f, accuracy = %0.4f' % (val_loss,val_acc*100))
        # Save model with best validation accuracy
        if val_loss < min_val_loss:
          best_epoch = epoch
          print 'Saving best validation loss of %0.4f after %d passes' % (val_loss,i)
          min_val_loss = val_loss
          best_state = {'epoch': epoch, 'loss': val_loss, 'accuracy': val_acc, 'model': model.state_dict()}
          state_fname = str.format('model_mb-%d_loss-%0.4f_acc-%0.4f.pth' % (i,val_loss,val_acc))
          torch.save(best_state, os.path.join(resultdir,state_fname))

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
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
  criterion = nn.CrossEntropyLoss(weights) 

  # Train with early stopping
  #train_with_early_stopping(X_train, y_train, X_val, y_val, model, optimizer, criterion, vocab, nepoch, batchsz, num_classes, exptdir, mode)

  # Load model at best validation loss and compute test accuracy
  state_files = os.listdir(exptdir)
  state_files = [fname for fname in state_files if fname.startswith('model') and fname.endswith('.pth')]
  min_val_loss = 1e10
  for fname in state_files:
    val_loss = float(fname.split('.pth')[0].split('_')[2].split('-')[1])
    if val_loss < min_val_loss:
      min_val_loss = val_loss
      best_state_file = fname
  print('Loading %s for testing' % best_state_file)
  best_state = torch.load(os.path.join(exptdir, best_state_file))
  model.load_state_dict(best_state['model'])
  test_loss, test_acc, test_fscore = test_data(X_test, y_test, model, criterion, 1, vocab, mode)
  print('Test: loss = %0.4f, accuracy = %0.4f, fscore = %0.4f' % (test_loss,test_acc*100,test_fscore*100))

if __name__ == "__main__":
  main(sys.argv[1:])
