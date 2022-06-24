import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            # Linear layer + Nonlinearity applied on fingerprint
            hs = torch.relu(self.W_gnn[i](xs))
            # fingerpint = fingerprint + AdjMatrix * hs
            xs = xs + torch.matmul(A, hs)

        # FINAL OUTPUT = average of the vertex vectors (formula 7)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        # ADD some dimensions to word-embeddings
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        # Some convolution layers + Nonlinearity for word-embeddings
        # kernel_size=2*window+1, padding=window
        # where WINDOW -- how many embedded words need to be concatenated (section 4.1)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        # REMOVE dimensions (??????????????????????????????????????)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        # get scalar values -- weights for neural attention mechamism
        # W_attention -- weight matrix W_inter
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        # for each WORD we get weight -- attention -- interaction strength between a molecule and the subsequence of a protein (WORD)
        # (formula 10)
        weights = torch.tanh(F.linear(h, hs))
        # weighted sum of H_i
        ys = torch.t(weights) * hs

        # FINAL OUTPUT = average of the hidden vectors (formula 9)
        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        # First, embed drug fingerprint
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        # Second, apply GNN (layer_gnn - number of layers) on drug fingerprint 
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        # words = encoded N-GRAM-plets (e.g. triplets like AAG, TGC, ...)
        # First we translate words to randomly initialized embeddings (word-embeddings)
        word_vectors = self.embed_word(words)
        # Then we apply CNN
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))  # (2*dim, 2*dim)
        interaction = self.W_interaction(cat_vector)  # (2*dim, b_size) --> (2, b_size)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores