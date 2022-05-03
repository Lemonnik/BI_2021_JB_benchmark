import torch
import torch.nn as nn


class NFM(nn.Module):
    '''
    Neural Factorization Machine (NFM) model.

    See Also
    --------
    X He, TS Chua (2017):
        https://arxiv.org/abs/1708.05027
    '''
    def __init__(self, num_features: int, num_factors: int, layers: list, batch_norm: bool, drop_prob: list):
        super(NFM, self).__init__()
        """
        Initialize model

        Parameters
        ----------
        num_features : int
            Number of features;
            features in our case = DistMult embeddings + drug features + protein features
        num_factors : int
            Number of hidden factors. Hyperparameter.
        act_function : {'relu', 'tanh'}
            Activation function for MLP layer. NOT IMPLEMENTED.
        layers : list
            List of dimension of deep layers.
        batch_norm : bool 
            Whether to use batch norm or not.
        drop_prob: list[2]
            Dropout rate for FM and MLP.
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))      
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        # Linear layers (+ batch norm and activation function)
        MLP_module = []
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            
            MLP_module.append(nn.ReLU())

            MLP_module.append(nn.Dropout(drop_prob[-1]))
        self.deep_layers = nn.Sequential(*MLP_module)

        # Prediction layer
        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight_()

    def _init_weight_(self):
        '''Initialize weights

        '''
        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)

        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, features, feature_values):
        '''
        Predicts a relation based on vactor of feature values.

        Parameters
        ----------
        features : torch.tensor[batch_size, num_features]
            Non-zero features. NOT IMPLEMENTED.
            Takes all features atm.
        feature_values : torch.tensor[batch_size, num_features]
            Tensor of features (KGE embeddings, drug features, protein features).

        Returns
        -------
        logits : torch.tensor[batch_size]
            Probabilites-like array showing the "probability" of the existance of relation.
        '''
        nonzero_embed = self.embeddings(features)             # shape: (batch_size, n_feats, dimension)
        feature_values = feature_values.unsqueeze(dim=-1)     # shape: (batch_size, n_feats, 1)
        nonzero_embed = nonzero_embed * feature_values        # shape: (batch_size, n_feats, dimension)

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)    # shape: (batch_size, dimension)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)  # shape: (batch_size, dimension)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)       # BiInteractionPooling
        FM = self.FM_layers(FM)                                # BatchNorm + Dropout
        if self.layers:
            FM = self.deep_layers(FM)                          # Deep layers
        FM = self.prediction(FM)                               # Linear prediction layer

        # Bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)