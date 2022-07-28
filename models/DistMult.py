import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import nn

from models.BaseModel import DtiModel


class DistMult(DtiModel):
    """
    Implementation of Knowledge graph embedding (KGE) model called DistMult
    detailed in 2014 paper by Yang B, et al. 

    References
    ----------
    * B.Yang et al (2014):
      https://arxiv.org/abs/1412.6575

    Parameters
    ----------
    n_nodes : int
        Number of nodes in knowledge graphs. 
        In DTI problem - total number of all proteins and drugs.
    n_relations : int
        Number of interactions between proteins and drugs.
    embedding_dim : int
        Embedding dimension. Hyperparameter.
    """

    def __init__(self, n_nodes: int, n_relations: int, embedding_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim

        self._return_type = ['DrugInd', 'ProtInd', 'Label']

        self.node_embedding = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        torch.nn.init.xavier_uniform_(self.node_embedding.weight)
        self.relation_embedding = nn.Embedding(num_embeddings=n_relations, embedding_dim=embedding_dim)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, head_indices, tail_indices, relation_indices):
        """
        Predicts a relation head -> tail.

        Parameters
        ----------
        head_indices : torch.tensor[batch_size]
            Tensor containing all head indices,
            e.g. encoded drugs/proteins.
        tail_indices : torch.tensor[batch_size]
            Tensor containing all tail indices,
            e.g. encoded drugs/proteins.
        relation_indices : torch.tensor[batch_size]
            Tensor containing labels,
            specifying the "head -> tail" relation type.
            In DTI task, there is only one relation type (0/1).

        Returns
        -------
        scores : torch.tensor[batch_size]
            Predicted type of relation.
        """

        head_embeddings = self.node_embedding(head_indices)
        tail_embeddings = self.node_embedding(tail_indices)
        relation_embeddings = self.relation_embedding(relation_indices)
        scores = torch.sum(
            head_embeddings * relation_embeddings * tail_embeddings, dim=-1
        )
        return scores

    def __call__(self, data, train=True, device='cpu'):
        head_indices, tail_indices, relation_indices = data
        head_indices = head_indices.type(torch.LongTensor).to(self.device)
        tail_indices = tail_indices.type(torch.LongTensor).to(self.device)
        relation_indices = relation_indices.type(torch.LongTensor).to(self.device)
        scores = self.forward(head_indices, tail_indices, relation_indices)

        if train:
            loss = F.mse_loss(scores, relation_indices.to(torch.float32))
            return loss
        else:
            correct_labels = relation_indices.to('cpu').data.numpy()
            # ys = F.softmax(scores, 0).to('cpu').data.numpy()
            predicted_labels = np.array([int(j > 0.5) for j in torch.sigmoid(scores).detach().cpu().numpy()])
            predicted_scores = torch.sigmoid(scores).detach().cpu().numpy()
            return correct_labels, predicted_labels, predicted_scores

    def get_embeddings(self, entities: list, embedding_type='entity'):
        """
        Get embeddings.

        Parameters
        ----------
        entities : list
            List of embedding id's.
        embedding_type: {'entity', 'relation'}
            Get node embeddings / relation embeddings.

        Returns
        -------
        emb_list : np.array
            Vector of embeddings.
        """
        if embedding_type == 'entity':
            emb_list = self.node_embedding
        elif embedding_type == 'relation':
            emb_list = self.relation_embedding
        else:
            raise ValueError(f"embedding_type should be 'entity' or 'relation'. Got {embedding_type}")

        return emb_list(torch.tensor(entities)).detach().numpy()
