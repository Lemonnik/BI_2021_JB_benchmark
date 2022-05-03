from torch import nn
import torch.nn.init

class DistMult(nn.Module):
    '''
    Knowledge graph embedding (KGE) model called DistMult.

    See Also
    --------
    B.Yang et al (2014):
        https://arxiv.org/abs/1412.6575
    '''
    def __init__(self, n_nodes: int, n_relations: int, embedding_dim: int):
        '''
        Initialize model

        Parameters
        ----------
        n_nodes : int
            Number of nodes in knoledge graphs. 
            In DTI problem - total number of all proteins and drugs.
        n_relations : int
            Number of interactions between proteins and drugs.
        embedding_dim : int
            Embedding dimension. Hyperparameter.
        '''
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        # TODO: add option to work with graphs, where heads and tail are different kind of entities
        self.node_embedding = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        torch.nn.init.xavier_uniform_(self.node_embedding.weight)
        self.relation_embedding = nn.Embedding(num_embeddings=n_relations, embedding_dim=embedding_dim)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, head_indices, tail_indices, relation_indices):
        '''
        Predicts a relation head -> tail.

        Parameters
        ----------
        head_indices : torch.tensor[batch_size]
            Tensor containing all head indicices,
            e.g. encoded drugs/proteins.
        tail_indices : torch.tensor[batch_size]
            Tensor containing all tail indicices,
            e.g. encoded drugs/proteins.
        relation_indices : torch.tensor[batch_size]
            Tensor containing labels 0/1,
            showing whether relation head -> tail exists.

        Returns
        -------
        scores : torch.tensor[batch_size]
            Probabilites-like array showing the "probability" of the existance of relation.
        '''
        head_embeddings = self.node_embedding(head_indices)
        tail_embeddings = self.node_embedding(tail_indices)
        relation_embeddings = self.relation_embedding(relation_indices)
        scores = torch.sum(
            head_embeddings * relation_embeddings * tail_embeddings, dim=-1
        )
        return scores


    def get_embeddings(self, entities: list, embedding_type='entity'):
        '''
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
        '''
        if embedding_type == 'entity':
            emb_list = self.node_embedding
        elif embedding_type == 'relation':
            emb_list = self.relation_embedding

        

        return emb_list(torch.tensor(entities)).detach().numpy()