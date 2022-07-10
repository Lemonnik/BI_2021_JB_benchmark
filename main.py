import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import pandas as pd
import time
import timeit
from tqdm import trange
# import wandb

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_auc_score, roc_curve, RocCurveDisplay, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from datasets.datasets import Davis
from preprocess.CPI_prediction_2018_preprocess import CPI_prediction_2018_preprocess
from models.CPI_prediction_2018_model import CompoundProteinInteractionPrediction
from models.DistMult import DistMult



class Trainer(object):
    """
    Function to train model.

    Parameters
    ----------
    model
        Any model compatible with our DTI benchmark (inherited from BaseModel Class).
    features_needed : list

    """
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset, batch_size, device):
        dataset.return_type = self.model.return_type
        loader = DataLoader(dataset, batch_size=batch_size)

        self.model.train()

        loss_total = 0

        for params in loader:  
            self.optimizer.zero_grad() 
            params = [p.to(device).long() for p in params]
            loss = self.model(params)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.to('cpu').data.numpy()
        return loss_total

class Tester(object):
    """
    Function to test model.

    Parameters
    ----------
    model
        Any model compatible with our DTI benchmark (inherited from BaseModel Class).
    features_needed : list

    """
    def __init__(self, model):
        self.model = model

    def test(self, dataset, batch_size, device):
        dataset.return_type = self.model.return_type
        loader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()

        T, Y, S = [], [], []
        for params in loader: 
            params = [p.to(device).long() for p in params]
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(params, train=False)
            T.extend(correct_labels)
            Y.extend(predicted_labels)
            S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall


def run_model(model, 
              dataset_train, 
              dataset_test,
              device,
              batch_size=64, 
              n_epochs=100, 
              lr_decay=0.5, 
              decay_interval=10, 
              lr=0.7e-3, 
              weight_decay=1e-6):

    trainer = Trainer(model, lr, weight_decay)
    tester = Tester(model)

    AUCs = ('Epoch\tTime(sec)\tLoss_train\t'
                'AUC_test\tPrecision_test\tRecall_test')

    """ Start training. """
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    for epoch in range(1, n_epochs):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, batch_size, device)
        # AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, precision_test, recall_test = tester.test(dataset_test, batch_size, device)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train,
                AUC_test, precision_test, recall_test]

        if epoch % 10 == 0:
            print('\t'.join(map(str, AUCs)))



@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:

    base_path = 'data/'

    """ Load dataset """
    # TODO: (HERE USER SHOULD CHOOSE DATASET TO LOAD)
    d_train = Davis(base_path, download=True)
    d_test = d_train
    d_test.mode = 'test'
    # atm. dataset contains all information (train and test)
    # we can decide what part should be returned by changing ``mode``

    # wandb.init(project="DTI-prediction")

    # Path to save checkpoints later
    curr_date = time.strftime('%d-%m-%Y')
    curr_time = time.strftime('%H-%M-%S')
    model_save_path = os.path.join(base_path, curr_date, curr_time)

    # Define SEED and DEVICE
    torch.manual_seed(cfg.run_args.seed)

    """ Preprocessing """   
    # TODO: (HERE USER SHOULD CHOOSE PREPROCESSING FUNCTION)
    d_train = CPI_prediction_2018_preprocess(d_train)
    d_test = d_train
    d_test.mode = 'test'

    """ DEVICE """
    if cfg.run_args.gpu:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'\nRunning models on {DEVICE}.\n')

    """ Run model """   
    # TODO: (HERE USER SHOULD CHOOSE MODEL)
    # It is possible to work with two models at the moment

    # model = DistMult(n_nodes=d_train.n_entities, 
                    #  n_relations=2, 
                    #  embedding_dim=cfg.model.kge.embed_dim).to(DEVICE)
    # batch_size_DistMult = 64

    model = CompoundProteinInteractionPrediction(n_word=len(d_train.word_dict),
                                                 n_fingerprint=len(d_train.fingerprint_dict)).to(DEVICE)
    batch_size_CPI = 1

    run_model(model, d_train, d_test, DEVICE, batch_size=1)
    
    


if __name__ == '__main__':
    main()