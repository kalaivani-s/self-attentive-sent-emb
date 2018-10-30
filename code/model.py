import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):

  def __init__(self, num_tokens, embed_size, hidden_size, nlayers, dropout=0.5, bidirectional=True, emb_vectors=None):
    super(LSTM, self).__init__()
    self.ntokens = num_tokens
    self.embedsz = embed_size
    self.hiddensz = hidden_size
    self.nlayers = nlayers
    self.bidirectional = bidirectional

    self.embed = nn.Embedding(num_tokens+1, embed_size, padding_idx=0)
    ### Initialize with pretrained word vectors
    if emb_vectors is not None:
      self.embed.weight.data[1:,:] = emb_vectors.clone()
    self.lstm = nn.LSTM(embed_size, hidden_size, nlayers, dropout=dropout, bidirectional=bidirectional)
    self.dropout = nn.Dropout(p=dropout)
 
  def forward(self, inp, hidden, batch_seqlen):
    emb = self.embed(inp)
    packed = pack_padded_sequence(emb, batch_seqlen, batch_first=False)
    out, hidden = self.lstm(packed, hidden)
    out, seqlen = pad_packed_sequence(out)
    out = out.transpose(0, 1).contiguous()
    return out, hidden

  def init_weights(self, init_range=0.1):
    self.embed.weight.data.uniform_(-init_range, init_range)

  def init_hidden(self, bsz):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.nlayers * 2, bsz, self.hiddensz).zero_()),
            Variable(weight.new(self.nlayers * 2, bsz, self.hiddensz).zero_()))

class SelfAttentiveEncoder(nn.Module):

  def __init__(self, num_tokens, embed_size, hidden_size, nlayers, attn_units, attn_hops, fcsize, nclasses, emb_vectors=None, bidirectional=True, drop=0.2):
    super(SelfAttentiveEncoder, self).__init__()
    self.hiddensz = hidden_size
    self.attnunits = attn_units
    self.attnhops = attn_hops
    self.fcsize = fcsize
    self.bidirectional = bidirectional
    
    self.lstm = LSTM(num_tokens, embed_size, hidden_size, nlayers, dropout=drop, bidirectional=self.bidirectional, emb_vectors=emb_vectors)
    self.ws1 = nn.Linear(2*hidden_size, attn_units)
    self.ws2 = nn.Linear(attn_units, attn_hops)
    self.fc = nn.Linear(2*hidden_size*attn_hops, fcsize)
    self.finalfc = nn.Linear(fcsize, nclasses)
    self.dropout = nn.Dropout(drop)
        
  def forward(self, inp, hidden, batch_seqlen):
    H, hidden = self.lstm(inp, hidden, batch_seqlen)
    bsz, seqlen, nhid = H.size()   
    H_comp = H.view(-1, nhid) # bsz*seqlen x nhid
    A = self.ws2(F.tanh(self.ws1(self.dropout(H_comp)))) # bsz*seqlen x r
    A = A.view(bsz, seqlen, -1).transpose(1,2).contiguous() # bsz x r x seqlen
    A = F.softmax(A.view(-1,seqlen),dim=1).view(bsz, -1, seqlen) # bsz x r x seqlen
    M = torch.bmm(A,H).view(bsz,-1)
    out = F.relu(self.fc(M))
    out = self.finalfc(self.dropout(out))
    return out, A
 
  def init_weights(self, init_range=0.1):
    self.ws1.weight.data.uniform_(-init_range, init_range)
    self.ws2.weight.data.uniform_(-init_range, init_range)
    self.fc.weight.data.uniform_(-init_range, init_range)
    self.fc.bias.data.fill_(0)
    self.finalfc.weight.data.uniform_(-init_range, init_range)
    self.finalfc.bias.data.fill_(0)

  def init_hidden(self, bsz):
    return self.lstm.init_hidden(bsz)
