from sklearn.utils import shuffle
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from tqdm import trange
import wandb
import time
import os
import rdkit
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# from utils import LoadDavis
from datasets import LoadDavisOrDB
from models.KGE_models import DistMult, TriVec
from models.NFM import NFM


def prepare_embs(model_name, target_indexes, drug_indexes):
    '''
    Combine KGE model embeddings and drugs/proteins encoding.

    Parameters
    ----------
    model_name : KGE model
        Models from which embeddings will be extracted.
    target_indexes : list
        Indicies of proteins.
    drug_indexes : list
        Indicies of drugs.
    
    Returns
    -------
    emb_features: array[n_relations + DistMult_dimension*2]
        Combined KGE model embeddings and drugs/proteins encoding.
    '''
    head_emb = model_name.get_embeddings(target_indexes)
    tail_emb = model_name.get_embeddings(drug_indexes)

    all_emb = np.concatenate([head_emb, tail_emb], axis=1)  # склеенные эмбеддинги голов и хвостов

    mms = MinMaxScaler(feature_range=(0,1))
    emb_features = mms.fit_transform(all_emb)  # трансформируем

    return emb_features


@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    # Load dataset
    if cfg.run_args.dataset in ['davis', 'BindingDB']:
        dataset = LoadDavisOrDB(df=cfg.run_args.dataset, drug_enc=cfg.run_args.drug_enc, prot_enc=cfg.run_args.prot_enc)

    wandb.init(project="DTI-prediction")

    # Path to save checkpoints later
    curr_date = time.strftime('%d-%m-%Y')
    curr_time = time.strftime('%H-%M-%S')
    model_save_path = f'{curr_date}/{curr_time}'

    ################################################# DistMult #################################################
    # Define SEED and DEVICE
    torch.manual_seed(cfg.run_args.seed)
    if cfg.run_args.gpu:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Running models on {DEVICE}.\n')

    # Initialize KGE model
    model = DistMult(dataset.n_entities, dataset.n_relations, cfg.model.kge.embed_dim)
    config = wandb.config
    wandb.watch(model)
    # Define Loss-function and Optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.kge.lr)

    # Train on GPU if possible
    model = model.to(DEVICE)


    # TRAINING CYCLE
    print("Starting training DistMult...")
    model.train()
    dataset.return_type = 'ind'
    loader = DataLoader(dataset, 
                        batch_size=cfg.model.kge.batch_size, 
                        pin_memory=True, 
                        shuffle=True)

    for epoch in range(cfg.model.kge.epoch):

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
        if epoch % 1 == 0:
            wandb.log({"loss": running_loss/n_batches})
            print(f"Epoch {epoch:>3} | Train Loss {running_loss/n_batches:.4f} | Accuracy {running_score/n_batches:.4f}")

    wandb.finish()
    # Save model state
    kge_path = os.path.join(model_save_path, cfg.model.kge.save_dir)
    os.makedirs(kge_path, exist_ok=True)
    torch.save(model.state_dict(), f'{kge_path}/checkpoint.pth')

    #################################################   PCA   #################################################

    # NOT IMPLEMENTED


    #################################################   NFM   #################################################
    # Get DistMult embeddings and write them in davis dataset
    embs = prepare_embs(model, dataset.dataset.ProtIND.values, dataset.dataset.DrugIND.values)
    dataset.add_feature('embeddings', embs.tolist())

    # total_features = DISTMULT_DIM*2 (heads + tails) + N (drug and protein fetures)
    dataset.return_type = 'encoding'  # change __getitem__ returns from (drugs, targets, labels) to (features, labels)
    features, _ = dataset[0]
    total_features = len(features)
    # Initialize NFM model
    model_nfm = NFM(total_features, cfg.model.nfm.embed_dim, cfg.model.nfm.layers, cfg.model.nfm.batch_norm, cfg.model.nfm.dropout)

    # Define Loss-function and Optimizer
    loss_fn_nfm  = torch.nn.MSELoss()
    optimizer_nfm  = torch.optim.Adam(model_nfm.parameters(), lr=cfg.model.nfm.lr)

    # Train on GPU if possible
    model_nfm = model_nfm.to(DEVICE)
    
    # TRAINING CYCLE
    print("\nStarting training NFM...")
    model_nfm.train()
    
    nfm_loader = DataLoader(dataset, batch_size=cfg.model.nfm.batch_size, drop_last=True, pin_memory=True)

    for epoch in range(cfg.model.nfm.epoch):

        running_score = 0
        n_batches = 0
        running_loss = 0

        for features, labels in nfm_loader:
            optimizer_nfm.zero_grad()

            features_non_zero = torch.tensor(list(range(features.shape[1])), dtype=torch.int).repeat(cfg.model.nfm.batch_size, 1)
            score = model_nfm(features_non_zero.long(), features.long())

            loss = loss_fn_nfm(score, labels) 

            # Statistics
            running_loss += loss
            predictions = np.array([int(j>0.5) for j in torch.sigmoid(score).detach().numpy()])
            running_score += accuracy_score(labels.numpy(), predictions)
            n_batches += 1

            loss.backward()
            optimizer_nfm.step()

        # Print epoch results
        if epoch % 1 == 0:
            print(f"Epoch {epoch:>3} | Train Loss {running_loss/n_batches:.4f} | Accuracy {running_score/n_batches:.4f}")

    # Save model state
    nfm_path = os.path.join(model_save_path, cfg.model.nfm.save_dir)
    os.makedirs(nfm_path, exist_ok=True)
    torch.save(model.state_dict(), f'{nfm_path}/checkpoint.pth')


if __name__ == '__main__':
    main()