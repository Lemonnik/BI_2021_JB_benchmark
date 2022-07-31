import pandas as pd
import numpy as np
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import copy

from models.BaseModel import DtiModel

# TODO: DtiModel inherit from nn.Module ?
class MhsadtiModel(DtiModel):
# class MhsadtiModel(nn.Module):
    """
    Implementation of MHSADTI model detailed in 2021 paper by Cheng Z, et al.

    References
    ----------
    * Cheng Z., Yan C., Wu F., Wang J..
      Drug-target interaction prediction using multi-head
      self-attention and graph attention network.
      IEEE/ACM Trans Comput Biol Bioinform. 2021;
      doi:10.1109/TCBB.2021.3077905

    Notes
    -----
    All code is taken from authors' orignal github with small changes
    https://github.com/czjczj/MHSADTI

    Parameters
    ----------
    dim: int
        Dimension of embedding space.

    """

    def __init__(self,
                 n_word: int,
                 n_fingerprint: int,
                 dim: int,
                 layer_gnn: int,
                 layer_cnn: int,
                 window: int,
                 layer_output: int,
                 N: int,
                 d_model: int,
                 d_ff: int,
                 h: int,
                 dropout: float,
                 MAX_LEN: int):
        super(MhsadtiModel, self).__init__()
        self.dim = dim
        self.n_word = n_word
        self.n_fingerprint = n_fingerprint
        self.layer_gnn = layer_gnn
        self.layer_cnn = layer_cnn
        self.window = window
        self.layer_output = layer_output
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout
        self.MAX_LEN = MAX_LEN

        self._return_type = ['compounds_mhsadti',
                             'compound_masks_mhsadti',
                             'adjacencies_mhsadti',
                             'proteins_mhsadti',
                             'protein_masks_mhsadti',
                             'Label']

        self.embed_fingerprint = nn.Embedding(n_fingerprint, self.dim)
        self.embed_word = nn.Embedding(self.n_word, self.dim)
        self.W_gnn = nn.ModuleList([nn.Linear(self.dim, self.dim)
                                    for _ in range(self.layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*self.window+1,
                     stride=1, padding=self.window) for _ in range(self.layer_cnn)])
        self.W_attention = nn.Linear(self.dim, self.dim)
        self.W_out = nn.ModuleList([nn.Linear(2*self.dim, 2*self.dim)
                                    for _ in range(self.layer_output)])
        self.W_interaction = nn.Linear(2*self.dim, 2)

        self.trans = make_model(self.n_word,
                                N=N,
                                d_model=self.d_model,
                                d_ff=self.d_ff,
                                h=self.h,
                                dropout=self.dropout,
                                MAX_LEN=MAX_LEN)
        self.trans_out = nn.Linear(128, 10)

        self.dropout = 0.6
        self.weight = None
        self.compound_attn = nn.ParameterList([nn.Parameter(torch.randn(size=(2 * self.dim, 1)))
                                               for _ in range(self.layer_output)])
        # self.compound_attn = [nn.Parameter(torch.randn(size=(2 * dim, 1))) for _ in range(layer_output)]
        # for i in range(layer_output):
        #     self.register_parameter('compound_attn_{}'.format(i), self.compound_attn[i])

    def gat(self, xs, x_mask, A, layer):
        x_mask = x_mask.reshape(x_mask.size()[0], x_mask.size()[1], 1)
        for i in range(layer):
            h = torch.relu(self.W_gnn[i](xs))
            h = h*x_mask
            size = h.size()[0]
            N = h.size()[1]
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1, 2 * self.dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(A > 0, e, zero_vec)  # 保证softmax 不为 0
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout)
            h_prime = torch.matmul(attention, h)
            xs = xs+h_prime
        xs = xs*x_mask
        return torch.unsqueeze(torch.mean(xs, 1), 1)

    def attention_cnn(self, x, xs, layer):
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(torch.matmul(h, hs.permute([0, 2, 1])))
        ys = weights.permute([0, 2, 1]) * hs
        self.weight = weights
        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 1), 1)

    def forward(self, inputs):
        fingerprints, fingerprints_mask, adjacency, words, words_mask = inputs

        """Compound vector with GNN."""
        # First, embed drug fingerprint
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        # Second, apply GNN (layer_gnn - number of layers) on drug fingerprint
        compound_vector = self.gat(fingerprint_vectors, fingerprints_mask, adjacency, self.layer_gnn)

        """Protein vector with attention-CNN."""
        words_mask = words_mask.unsqueeze(-2)
        word_vectors_trans = self.trans(words, words_mask)  # [batch, length, feature_len]
        word_vectors = self.trans_out(F.relu(word_vectors_trans))  # [batch, length, feature_conv_len]
        protein_vector = self.attention_cnn(compound_vector, word_vectors, self.layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 2)
        for j in range(self.layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return torch.squeeze(interaction, 1)

    def __call__(self, data, train=True, device='cpu'):
        inputs, correct_interaction = data[:-1], data[-1]

        fingerprints, fingerprints_mask, adjacency, words, words_mask = inputs
        fingerprints = torch.squeeze(fingerprints, 0).type(torch.LongTensor).to(device)
        adjacency = torch.squeeze(adjacency, 0).type(torch.FloatTensor).to(device)
        words = torch.squeeze(words, 0).type(torch.LongTensor).to(device)
        correct_interaction = correct_interaction.type(torch.LongTensor).to(device)
        inputs = [fingerprints, fingerprints_mask, adjacency, words, words_mask]

        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            # TODO: check whether my personal changes is OK
            # return predicted_interaction
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, src_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed

    def forward(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# mutil-head attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model//h
        self.h = h
        self.linears = clones(nn.Linear(d_model,d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches,-1,self.h, self.d_k).transpose(1,2) for l,x in zip(self.linears, (query, key, value))]
        x,self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5038):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2)*-(math.log(10000.0)/d_model)).float())
        pe[:, 0::2] = torch.sin((position.float()*div_term).float())
        pe[:, 1::2] = torch.cos((position.float()*div_term).float())
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, N, d_model, d_ff, h, dropout, MAX_LEN):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, MAX_LEN)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
