import torch.nn as nn

class NFM(nn.Module):
    def __init__(self, num_features, num_factors, layers, batch_norm, drop_prob):
        super(NFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        """
        self.num_features = num_features
        self.num_factors = num_factors
        # self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        # self.pretrain_FM = pretrain_FM

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))      
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        # наслаиваем нужное количество линейных слоёв (+слои нормализации и активационной функции, если нужны)
        MLP_module = []                                 # контейнер слоёв
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            
            MLP_module.append(nn.ReLU())                 # или Sigmoid, или танг

            MLP_module.append(nn.Dropout(drop_prob[-1]))
        self.deep_layers = nn.Sequential(*MLP_module)

        # слой предсказаний
        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight_()

    def _init_weight_(self):
        # инициализация весов
        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)

        # для внутренних линейных слоёв веса с другим распределением (почему бы не везде) 
        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)  # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, features, feature_values):
        # пока что выбор ненулевых не реализован..., так что берём все
        nonzero_embed = self.embeddings(features)             # shape: (batch_size, n_feats, dimension)
        feature_values = feature_values.unsqueeze(dim=-1)     # shape: (batch_size, n_feats, 1)
        nonzero_embed = nonzero_embed * feature_values        # shape: (batch_size, n_feats, dimension)

        # Bi-Interaction layer
        # суммировали по всем фичам
        # то есть для каждого батча получили 1 вектор размера [dimension]
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)    # shape: (batch_size, dimension)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)  # shape: (batch_size, dimension)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)       # применили BiInteractionPooling
        FM = self.FM_layers(FM)                                # нормализация + дропаут
        if self.layers:
            FM = self.deep_layers(FM)                          # применили глубокие слои (нелинейность)
        FM = self.prediction(FM)                               # линейный слой предсказаний

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)