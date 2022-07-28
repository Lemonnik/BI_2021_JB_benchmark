from omegaconf import DictConfig

from models.CPI_2018_model import CpiModel
from models.DistMult import DistMult
from models.MolTrans_2020_model import MolTransModel
from models.NFM import NFM
from preprocess.CPI_2018_preprocess import cpi_preprocess
from preprocess.MolTrans_2020_preprocess import moltrans_preprocess

preprocess_functions = {'CPI_2018': cpi_preprocess,
                        'MolTrans_2020': moltrans_preprocess}


def select_model(model_name: str, params: DictConfig, dataset):
    """
       Model selection must be stated here for each model.
    """
    if model_name == 'DistMult':
        model = DistMult(n_nodes=dataset.n_entities,
                         n_relations=2,
                         embedding_dim=params.embed_dim)
    elif model_name == 'CPI_2018':
        model = CpiModel(n_word=len(dataset.word_dict),
                         n_fingerprint=len(dataset.fingerprint_dict),
                         dim=params.dim,
                         layer_gnn=params.layer_gnn,
                         layer_cnn=params.layer_cnn,
                         window=params.window,
                         layer_output=params.layer_output)
    elif model_name == 'MolTrans_2020':
        model = MolTransModel(**params)
    elif model_name == 'nfm':
        model = NFM(num_features=len(dataset[0]),
                    num_factors=params.embed_dim,
                    layers=params.layers,
                    batch_norm=params.batch_norm,
                    drop_prob=params.drop_prob)
    else:
        raise ValueError(f'No model is stated for {model_name} in model_and_preprocess_selection.py file.')

    return model


def preprocess_dataset(model_name: str, params: DictConfig, dataset):
    """
        Preprocesses the dataset using function from preprocess_functions dictionary.
        Each model must have its own preprocessing function,
            that adds required features to dataset via 'add_feature' method
        or 'None' value if preprocessing is not needed.
    """
    if params == 'None':
        return dataset
    if model_name not in preprocess_functions.keys():
        raise ValueError(
            f'Preprocessing is not stated for {model_name} model in model_and_preprocess_selection.py file.')

    if preprocess_functions[model_name]:
        dataset = preprocess_functions[model_name](dataset, **params)
    return dataset
