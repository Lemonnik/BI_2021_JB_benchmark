import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import pandas as pd
import time
from tqdm import trange
import wandb

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from datasets import LoadDavisOrDB
from models.KGE_models import DistMult, TriVec
from models.NFM import NFM as NFM
from utils import prepare_embs


def run_kge(dataset, model, cfg, model_save_path, DEVICE, with_nfm):
    '''
    Train and test KGE model.
    '''
    # Define Loss-function and Optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.kge.lr)

    # Train on GPU if possible
    model = model.to(DEVICE)

    # DataLoader
    dataset.return_type = 'ind'
    loader = DataLoader(dataset, 
                        batch_size=cfg.model.kge.batch_size, 
                        pin_memory=True, 
                        shuffle=True)

    # TRAINING CYCLE
    print("Starting training DistMult...")
    train_kge(model, loader, cfg.model.kge.epoch, optimizer, loss_fn, print_every=1)

    # Save model state
    kge_path = os.path.join(model_save_path, cfg.model.kge.save_dir)
    os.makedirs(kge_path, exist_ok=True)
    torch.save(model.state_dict(), f'{kge_path}/checkpoint.pth')

    if not with_nfm:
        wandb.finish()
        # test model
        print("Starting testing DistMult...")
        roc_auc = test_kge(model, dataset, optimizer, loss_fn)
    else:
        roc_auc = run_nfm(model, cfg, dataset, model_save_path, DEVICE)

    return roc_auc

def train_kge(model, loader, n_epoch, optimizer, loss_fn, print_every=1):
    '''
    Train KGE model.
    '''

    model.train()
    for epoch in range(n_epoch):

        running_score = 0
        n_batches = 0
        running_loss = 0

        for drugs, targets, labels in loader:  
            optimizer.zero_grad() 

            score = model(drugs.long(), targets.long(), labels.long())
            loss = loss_fn(score, labels)

            # Statistics
            running_loss += loss
            predictions = np.array([int(j>0.5) for j in torch.sigmoid(score).detach().numpy()])
            running_score += accuracy_score(labels.numpy(), predictions)
            n_batches += 1

            loss.backward()
            optimizer.step()  

        # Print epoch results
        if epoch % print_every == 0:
            wandb.log({"loss": running_loss/n_batches})
            print(f"Epoch {epoch:>3} | Train Loss {running_loss/n_batches:.4f}")


def test_kge(model, dataset, optimizer, loss_fn):
    '''
    Test KGE model.
    '''

    model.eval()

    loader = DataLoader(dataset, 
                        batch_size=len(dataset), 
                        pin_memory=True, 
                        shuffle=True)

    roc_auc = 0
    n_batches = len(dataset)/len(dataset)

    for drugs, targets, labels in loader:  
        score = model(drugs.long(), targets.long(), labels.long())
        loss = loss_fn(score, labels)

        # Statistics
        predictions = np.array([int(j>0.5) for j in torch.sigmoid(score).detach().numpy()])
        roc_auc += roc_auc_score(labels.numpy(), predictions)
        # fpr, tpr, thresholds = roc_curve(labels.numpy(), torch.sigmoid(predictions).detach().numpy())

    return roc_auc/n_batches


def run_nfm(kge_model, cfg, dataset, model_save_path, DEVICE):
    '''
    Train and test NFM.
    '''
    # Get DistMult embeddings and write them in the dataset
    embs = prepare_embs(kge_model, dataset.dataset.ProtIND.values, dataset.dataset.DrugIND.values)
    dataset.add_feature('embeddings', embs.tolist())

    # total_features = DISTMULT_DIM*2 (heads + tails) + N (drug and protein fetures)
    dataset.return_type = 'encoding'  # change __getitem__ returns from (drugs, targets, labels) to (features, labels)
    features, _ = dataset[0]
    total_features = len(features)
    # Initialize NFM model
    model_nfm = NFM(total_features, cfg.model.nfm.embed_dim, cfg.model.nfm.layers, cfg.model.nfm.batch_norm, cfg.model.nfm.dropout)
    wandb.watch(model_nfm)
    # Define Loss-function and Optimizer
    loss_fn_nfm  = torch.nn.MSELoss()
    optimizer_nfm  = torch.optim.Adam(model_nfm.parameters(), lr=cfg.model.nfm.lr)

    # Train on GPU if possible
    model_nfm = model_nfm.to(DEVICE)
    
    nfm_loader = DataLoader(dataset, batch_size=cfg.model.nfm.batch_size, drop_last=True, pin_memory=True)

    # TRAINING CYCLE
    print("\nStarting training NFM...")
    train_nfm(model_nfm, nfm_loader, cfg.model.nfm.epoch, cfg.model.nfm.batch_size, optimizer_nfm, loss_fn_nfm, print_every=1)
    wandb.finish()

    # Save model state
    nfm_path = os.path.join(model_save_path, cfg.model.nfm.save_dir)
    os.makedirs(nfm_path, exist_ok=True)
    torch.save(model_nfm.state_dict(), f'{nfm_path}/checkpoint.pth')

    print("Starting testing NFM...")
    roc_auc = test_nfm(model_nfm, dataset, loss_fn_nfm, cfg.model.nfm.batch_size)
    return roc_auc


def train_nfm(model, loader, n_epoch, batch_size, optimizer, loss_fn, print_every=1):
    '''
    Train NFM model.
    '''
    model.train()

    for epoch in range(n_epoch):

        running_score = 0
        n_batches = 0
        running_loss = 0

        for features, labels in loader:
            optimizer.zero_grad()

            features_non_zero = torch.tensor(list(range(features.shape[1])), dtype=torch.int).repeat(batch_size, 1)
            score = model(features_non_zero.long(), features.long())

            loss = loss_fn(score, labels) 

            # Statistics
            running_loss += loss
            predictions = np.array([int(j>0.5) for j in torch.sigmoid(score).detach().numpy()])
            running_score += accuracy_score(labels.numpy(), predictions)
            n_batches += 1

            loss.backward()
            optimizer.step()

        # Print epoch results
        if epoch % 1 == 0:
            wandb.log({"loss": running_loss/n_batches})
            print(f"Epoch {epoch:>3} | Train Loss {running_loss/n_batches:.4f}")

def test_nfm(model, dataset, loss_fn, batch_size):
    '''
    Test NFM model.
    '''
    model.eval()

    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        pin_memory=True, 
                        shuffle=True, 
                        drop_last=True)

    roc_auc = 0
    i=0

    for features, labels in loader:
        features_non_zero = torch.tensor(list(range(features.shape[1])), dtype=torch.int).repeat(batch_size, 1)
        score = model(features_non_zero.long(), features.long())
        loss = loss_fn(score, labels)

        # Statistics
        predictions = np.array([int(j>0.5) for j in torch.sigmoid(score).detach().numpy()])
        roc_auc += roc_auc_score(labels.numpy(), predictions)
        i+=1
        # fpr, tpr, thresholds = roc_curve(labels.numpy(), torch.sigmoid(predictions).detach().numpy())

    return roc_auc/i


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    # Load dataset
    if cfg.run_args.dataset in ['davis', 'BindingDB']:
        dataset = LoadDavisOrDB(df=cfg.run_args.dataset, drug_enc=cfg.run_args.drug_enc, prot_enc=cfg.run_args.prot_enc)

    wandb.init(project="DTI-prediction")

    # Path to save checkpoints later
    curr_date = time.strftime('%d-%m-%Y')
    curr_time = time.strftime('%H-%M-%S')
    model_save_path = f'{curr_date}/{curr_time}'

    # Define SEED and DEVICE
    torch.manual_seed(cfg.run_args.seed)
    if cfg.run_args.gpu:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'\nRunning models on {DEVICE}.\n')

    # Run model
    if cfg.run_args.model_to_run == 'DistMult':
        model = DistMult(dataset.n_entities, dataset.n_relations, cfg.model.kge.embed_dim)
    elif cfg.run_args.model_to_run == 'TriVec':
        model = TriVec(dataset.n_entities, dataset.n_relations, cfg.model.kge.embed_dim)
    
    config = wandb.config
    wandb.watch(model)

    result_metric = run_kge(dataset, model, cfg, model_save_path, DEVICE, with_nfm=cfg.run_args.nfm)
    print(f'Result metric roc_auc on dataset {cfg.run_args.dataset} = {result_metric}')



if __name__ == '__main__':
    main()