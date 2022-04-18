import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import trange

# rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

# deepchem
import deepchem as dc
from deepchem.models import GCNModel

# torch
import torch
import torch_geometric

#
from utils import LoadDavis
from torch_geometric.loader import DataLoader
import DistMult
import NFM


# 
from sklearn.preprocessing import MinMaxScaler

def prepare_embs(model_name, target_indexes, drug_indexes):
    '''
    получить из модели эмбеддинги и склеить их
    output: (n_relations x DISTMULT_DIM*2)
    '''
    head_emb = model_name.get_embeddings(target_indexes)
    tail_emb = model_name.get_embeddings(drug_indexes)

    all_emb = np.concatenate([head_emb, tail_emb], axis=1)  # склеенные эмбеддинги голов и хвостов

    mms = MinMaxScaler(feature_range=(0,1))
    emb_features = mms.fit_transform(all_emb)  # трансформируем

    return emb_features



if __name__ == '__main__':
    # TODO: парсинг аргументов argparse?

    # загрузить датасет
    davis = LoadDavis()

    ################################################# DistMult #################################################
    # гиперпараметры
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_EPOCHS = 10
    LEARNING_RATE = 0.0007

    DISTMULT_DIM = 100


    # инициализируем модель
    model = DistMult(davis.n_entities, davis.n_relations, DISTMULT_DIM)

    # настраиваем loss-функцию и оптимизатор
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # по возможности обучать на GPU
    model = model.to(device)

    # тренировочный цикл
    print("Starting training DistMult...")
    model.train()
    davis.return_type = 'ind'
    loader = DataLoader(davis, batch_size=BATCH_SIZE)
    losses = []

    for epoch in range(N_EPOCHS):
        for i, (drugs, targets, labels) in enumerate(loader):  
            optimizer.zero_grad() 

            score = model(drugs.to(device).long(), targets.to(device).long(), labels.to(device).long())

            # надо ли накидывать сигмоиду/softplus? работает ли лосс функция только с логитами?
            # loss = torch.mean(nn.functional.softplus(- labels.to(device).long() * score))
            # loss = loss_fn(torch.sigmoid(score), labels)
            loss = loss_fn(score, labels)  

            loss.backward()
            optimizer.step()  

        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss {loss:.4f}")


    #################################################   PCA   #################################################

    # сворачиваем фичи белков/лекарств или же эмбеддинги тоже есть смысл сворачивать?


    #################################################   NFM   #################################################
    # получить эмбеддинги для всех элементов датасета и записать их в наш датасет
    embs = prepare_embs(model, davis.dataset.Targ_IND.values, davis.dataset.Drug_IND.values)
    davis.add_feature('embeddings', embs.tolist())
    

    # ГИПЕРПАРАМЕТРЫ (TODO: словарь?; свои lr, число эпох и т.п.)
    NUM_FACTORS = 100         # "глубина"
    LAYERS = [512, 256, 128]  # слои
    BATCH_NORM = True
    DROPOUT = [0.2, 0.2]      # дропаут после этапов FM и MLP
    # LEARNING_RATE_NFM =     # (убрать одинаковые названия переменных)
    # N_EPOCHS_NFM =  
    # BATCH_SIZE_NFM = 


    # инициализируем модель (TODO: задать 1495 через переменную)
    model_nfm = NFM(1495, NUM_FACTORS, LAYERS, BATCH_NORM, DROPOUT)

    # настраиваем loss-функцию и оптимизатор 
    loss_fn_nfm  = torch.nn.MSELoss()
    optimizer_nfm  = torch.optim.Adam(model_nfm.parameters(), lr=LEARNING_RATE)

    # по возможности обучать на GPU
    model_nfm = model_nfm.to(device)
    
    # тренировочный цикл
    print("Starting training NFM...")
    model_nfm.train()
    davis.return_type = 'encoding'  # меняем тип возвращаемых элементов с (drugs, targets, labels) на (features, labels)
    nfm_loader = DataLoader(davis, batch_size=BATCH_SIZE)
    losses_nfm = []

    for epoch in range(N_EPOCHS):
        for features, label in nfm_loader:
            optimizer_nfm.zero_grad()

            features_non_zero = torch.tensor(list(range(features.shape[1])), dtype=torch.int)  # пока что фиктивный признак, выбираются все
            prediction = model_nfm(features_non_zero.to(device).long(), features.to(device).long())  # 
            # loss = loss_fn_nfm(torch.sigmoid(prediction), label) 
            loss = loss_fn_nfm(prediction, label) 
            # loss += 0.001 * model_nfm.embeddings.weight.norm()

            loss.backward()
            optimizer_nfm.step()

            losses_nfm.append(loss)

        losses_nfm.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss {loss:.4f}")
