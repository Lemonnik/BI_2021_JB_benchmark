import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from pybiomed_helper import CalculateConjointTriad
from utils import encode_drugs, encode_targets, get_dataset_path
from sklearn.preprocessing import LabelEncoder

class LoadDavisOrDB(Dataset):
    '''
    Allows to load Davis or BindingDB dataset (not full, only small part), 
    usually used for Drug-Target Interaction (DTI) task.
    Davis contains 9,781 DTI pairs, 68 drugs, 379 proteins.
    BindingDB contains 26,359 DTI pairs, 7165 drugs, 1254 proteins.

    Notes
    -----
    DAVIS and BindingDB are downloaded from 
    https://github.com/kexinhuang12345/MolTrans/tree/master/dataset
    '''
    def __init__(self, df, drug_enc='morgan', prot_enc='conj_triad', mode='train', return_type = 'ind'):
        '''
        Load dataset.

        Parameters
        ----------
        df : {'davis', 'bindingDB'}
            What dataset should be loaded
        drug_enc : {'morgan'}
            Drug encoding type.
        prot_enc : {'conj_triad'}
            Protein encoding type.
        mode : {'train', 'val', 'test'}
            What part of the dataset need to be loaded.
        return_type : {'ind', 'encoding'}
            Defines __getitem__() behavior.
            Whether to return (drug index, target index, label) - used for KGE model
                           or (features, label)                 - used for NFM model
        '''
        # Load dataset
        dataFolder = get_dataset_path(df)

        dataset_train = pd.read_csv(dataFolder + '/train.csv', index_col=0)
        dataset_test = pd.read_csv(dataFolder + '/test.csv', index_col=0)
        dataset_val = pd.read_csv(dataFolder + '/val.csv', index_col=0)
        dataset = dataset_train.append(dataset_test, ignore_index=True).append(dataset_val, ignore_index=True).drop_duplicates(ignore_index=True)

        self.dataset = dataset                                        # full dataset
        self.drug_col = 'SMILES'
        self.target_col = 'Target Sequence'
        self.label_col = 'Label'

        self.n_relations = len(self.dataset)                          # number of relations (drug-target interactions)
        drugs = self.dataset[self.drug_col].unique()
        self.n_drugs = len(drugs)                                     # number of drugs
        prots = self.dataset[self.target_col].unique()
        self.n_prots = len(prots)                                     # number of proteins (targets)
        self.entities = np.append(drugs, prots)                                 # all unique graph nodes (drugs and targets)
        self.n_entities = len(self.entities)                          # number of graph nodes (drugs and targets)

        # Encode drugs and targets by ID's
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.entities)
            # get
        d = self.dataset[self.drug_col].values
        t = self.dataset[self.target_col].values
            # encode
        d = self.entity_to_ind(d)
        t = self.entity_to_ind(t)
            # save in dataset
        self.dataset['DrugIND'] = d
        self.dataset['ProtIND'] = t

        # Encode drugs and targets with some type of encoding
        self.drug_enc = drug_enc
        self.dataset['drug_encoding'] = encode_drugs(self.dataset[self.drug_col].to_list(), enc_type=self.drug_enc)
        self.prot_enc = prot_enc
        self.dataset['target_encoding'] = encode_targets(self.dataset[self.target_col].to_list(), enc_type=self.prot_enc)

        # What part of the dataset should be returned in __getitem__ (train/val/test)
        self.mode = mode

        # Return (drug index, target index, label) or (features, label) in __getitem__
        self.return_type = return_type

        # Embeddings + drug/prot features will be saved in this variable
        self.feat_enc = None


    def add_feature(self, feat_name: str, feat_values: list):
        '''
        Adds column to self.dataset.

        Parameters
        ----------
        feat_name : str
            Name of the column.
        feat_values : list
            Column values.
        
        Notes:
        -----
        Function created for adding embeddings. Should be replaced by smth more relevant.
        '''
        self.dataset[feat_name] = feat_values


    def ind_to_entity(self, ind: list):
        '''
        Gets list of drugs/proteins ID's.
        Returns SMILES strings/Target Sequences.
        '''
        return np.array([self.label_encoder.classes_[i] for i in ind])


    def entity_to_ind(self, s: list):
        '''
        Gets list of SMILES strings/Target Sequences.
        Returns their indexes (ID's).
        '''
        return self.label_encoder.transform(s)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        '''
        Returns (drug index, target index, label) or (features, label) 
        depending on current mode.
        '''
        # Return (drugs, targets, labels)
        if self.return_type == 'ind':
            # get
            d = self.dataset['DrugIND'].values[idx]
            t = self.dataset['ProtIND'].values[idx]
            i = self.dataset[self.label_col].values[idx]
            # convert to tensor
            d = torch.tensor(d, dtype=torch.float32)
            t = torch.tensor(t, dtype=torch.float32)
            i = torch.tensor(i, dtype=torch.float32)

            return d, t, i

        # Return (features, labels)
        elif self.return_type == 'encoding':
            # if features (embeddings and features) not saved -- save them in self.feat_enc
            if self.feat_enc is None:
                d = np.stack(self.dataset['drug_encoding'].values)
                t = np.stack(self.dataset['target_encoding'].values)

                if 'embeddings' in self.dataset.columns:
                    if 'embeddings_w_feats' not in self.dataset.columns:
                        e = np.stack(self.dataset.embeddings.values)
                        feats_and_embs = torch.tensor(np.concatenate([d, t, e], axis=1), dtype=torch.float32)
                        self.dataset['embeddings_w_feats'] = feats_and_embs.tolist()
                    else:
                        feats_and_embs = torch.tensor(np.stack(self.dataset['embeddings_w_feats']), dtype=torch.float32)
                else:
                    print('Embeddings not found')
                    return None
                self.feat_enc = feats_and_embs

            i = self.dataset[self.label_col].values[idx]
            i = torch.tensor(i, dtype=torch.float32)

            return self.feat_enc[idx, :], i