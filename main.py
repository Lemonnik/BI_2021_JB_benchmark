import os
import time
import timeit

import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from datasets.davisDataset import Davis
from model_and_preprocess_selection import select_model, preprocess_dataset


class Trainer(object):
    """
    Function to train model.

    Parameters
    ----------
    model
        Any model compatible with our DTI benchmark (inherited from BaseModel Class).
    """
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr, weight_decay=weight_decay)

    def train(self, dataset, batch_size, device):
        dataset.return_type = self.model.return_type
        dataset.mode = 'train'
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

        self.model.to(device)
        self.model.train()

        loss_total = 0

        for batch in loader:
            self.optimizer.zero_grad()
            loss = self.model(batch, device=device)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
        return loss_total


class Tester(object):
    """
    Function to test model.

    Parameters
    ----------
    model
        Any model compatible with our DTI benchmark (inherited from BaseModel Class).
    """
    def __init__(self, model):
        self.model = model

    def test(self, dataset, batch_size, device):
        dataset.return_type = self.model.return_type
        dataset.mode = 'test'
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

        self.model.eval()

        T, Y, S = [], [], []
        for params in loader:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(params, train=False, device=device)
            T.extend(correct_labels)
            Y.extend(predicted_labels)
            S.extend(predicted_scores)
        auc_test = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return auc_test, precision, recall


def run_model(model, 
              dataset, 
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
    print('--- Training ---')
    print(AUCs)
    start = timeit.default_timer()

    for epoch in range(1, n_epochs+1):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset, batch_size, device)
        # AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, precision_test, recall_test = tester.test(dataset, batch_size, device)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train,
                AUC_test, precision_test, recall_test]

        AUCs = [round(value, 3) for value in AUCs]

        if epoch % 1 == 0:
            print('\t'.join(map(str, AUCs)))


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    base_path = cfg.base_path

    """ Load dataset """
    # TODO: (HERE USER SHOULD CHOOSE DATASET TO LOAD)
    dataset = Davis(base_path, force_download=True)
    # atm. dataset contains all information (train and test)
    # we can decide what part should be returned by changing ``mode``

    # wandb.init(project="DTI-prediction")

    # Path to save checkpoints later
    curr_date = time.strftime('%d-%m-%Y')
    curr_time = time.strftime('%H-%M-%S')
    model_save_path = os.path.join(base_path, curr_date, curr_time)

    # Define SEED and DEVICE
    torch.manual_seed(cfg.seed)

    """ Preprocess selection """
    print('--- Data Preparation ---')
    dataset = preprocess_dataset(cfg.model_name, cfg[cfg.model_name].preprocess_params, dataset)
    print('--- Finished ---\n')

    """ Device selection"""
    print('--- Device Preparation ---')
    device = "cpu"
    if cfg.gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'Running models on {device}.\n')

    """ Model selection """
    print('--- Model Preparation ---')
    model = select_model(cfg.model_name, cfg[cfg.model_name].model_params, dataset)
    model = model.to(device)
    print(f'--- Using model {type(model).__name__} ---\n')

    """ Run model """
    # TODO: choose default batch_size/n_epoch/etc if parameter is not stated in cfg[cfg.model_name]
    run_model(model=model,
              dataset=dataset,
              device=device,
              batch_size=cfg[cfg.model_name].batch_size,
              n_epochs=cfg[cfg.model_name].epoch)


if __name__ == '__main__':
    main()
