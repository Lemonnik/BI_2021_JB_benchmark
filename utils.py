import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from pybiomed_helper import CalculateConjointTriad


class LoadDavis(Dataset):
    '''
    Load Davis dataset usually used for Drug-Target Interaction (DTI) task.
    Contains 25,772 DTI pairs, 68 drugs, 379 proteins.

    Notes
    -----
    Dataset is downloaded from https://github.com/kexinhuang12345/MolTrans/tree/master/dataset
    '''
    def __init__(self, mode='train', return_type = 'ind'):
        '''
        Load dataset.

        Parameters
        ----------
        mode : {'train', 'val', 'test'}
            What part of the dataset need to be loaded.
        return_type : {'ind', 'encoding'}
            Defines __getitem__() behavior.
            Whether to return (drug index, target index, label) - used for KGE model
                           or (features, label)                 - used for NFM model
        '''
        # Load dataset
        dataFolder = 'https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/DAVIS/'
        davis_train = pd.read_csv(dataFolder + '/train.csv', index_col=0)
        davis_test = pd.read_csv(dataFolder + '/test.csv', index_col=0)
        davis_val = pd.read_csv(dataFolder + '/val.csv', index_col=0)
        davis = davis_train.append(davis_test, ignore_index=True).append(davis_val, ignore_index=True).drop_duplicates(ignore_index=True)

        self.dataset = davis                                          # full dataset
        self.drug_col = 'SMILES'
        self.target_col = 'Target Sequence'
        self.label_col = 'Label'

        self.n_relations = len(self.dataset)                          # number of relations (drug-target interactions)
        self.n_drugs = len(self.dataset[self.drug_col].unique())      # number of drugs
        self.n_prots = len(self.dataset[self.target_col].unique())    # number of proteins (targets)
        self.n_entities = self.n_drugs + self.n_prots                 # number of graph nodes (drugs and targets)

        # Encode drugs and targets by ID's
        # SHOULD BE REPLACED WITH LABEL_ENCODER()
        self.entity_to_ind, self.ind_to_entity = self._encode_by_ind()
        self._encode_drugs()
        self._encode_targets()

        # What part of the dataset should be returned in __getitem__ (train/val/test)
        self.mode = mode

        # Return (drug index, target index, label) or (features, label) in __getitem__
        self.return_type = return_type

        # 
        d = self.dataset['Drug_IND'].values
        t = self.dataset['Targ_IND'].values
        i = self.dataset[self.label_col].values
        self.d_ind = torch.tensor(d, dtype=torch.float32)
        self.t_ind = torch.tensor(t, dtype=torch.float32)
        self.i_ind = torch.tensor(i, dtype=torch.float32)

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


    def get_entity_by_ind(self, i):
        '''
        Returns SMILES/Target Sequence by index.
        '''
        return self.ind_to_entity.get(i, None)


    def get_ind_by_sequence(self, s):
        '''
        Returns Drug/Protein index by it's SMILES/Target Sequence.
        '''
        return self.entity_to_ind.get(s, None)


    def _encode_by_ind(self):
        '''
        Drug and Proteins encoding by ID's.
        '''
        # Encode
        drug_encoding = {d:i for d,i in zip(self.dataset[self.drug_col].unique(), range(self.n_drugs))}
        prot_encoding = {p:i for p,i in zip(self.dataset[self.target_col].unique(), range(self.n_drugs, self.n_drugs+self.n_prots))}
        ind_encoding = {**drug_encoding, **prot_encoding}  # merge two dicts
        reversed_encoding = {v: k for k, v in ind_encoding.items()}

        # Add new column to dataset
        self.dataset['Drug_IND'] = [drug_encoding[i] for i in self.dataset[self.drug_col]]
        self.dataset['Targ_IND'] = [prot_encoding[i] for i in self.dataset[self.target_col]]

        return ind_encoding, reversed_encoding


    def _encode_drugs(self):
        '''
        Encode all drugs.
        '''
        unique_drugs = pd.Series(self.dataset[self.drug_col].unique()).apply(self._smiles2morgan)
        unique_drugs_dict = dict(zip(self.dataset[self.drug_col].unique(), unique_drugs))
        self.dataset['drug_encoding'] = [unique_drugs_dict[i] for i in self.dataset[self.drug_col]]


    def _encode_targets(self):
        '''
        Encode all proteins.
        '''
        unique_prot = pd.Series(self.dataset[self.target_col].unique()).apply(self._target2ct)
        unique_prots_dict = dict(zip(self.dataset[self.target_col].unique(), unique_prot))
        self.dataset['target_encoding'] = [unique_prots_dict[i] for i in self.dataset[self.target_col]]


    def _smiles2morgan(self, s):
        '''
        Encode one drug using Morgans Fingerprint.
        '''
        try:
            mol = Chem.MolFromSmiles(s)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except:
            print(f'Some problems with {s}.. Array of 0\'s will be returned')
            features = np.zeros((1024, ))
        return features


    def _target2ct(self, s):
        '''
        Encode one protein using Conjoint Triad.
        '''
        try:
            features = CalculateConjointTriad(s)
        except:
            print(f'Some problems with {s}.. Array of 0\'s will be returned')
            features = np.zeros((343, ))
        return features


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        '''
        Returns (drug index, target index, label) or (features, label) 
        depending on current mode.
        '''
        # Return (drugs, targets, labels)
        if self.return_type == 'ind':
            return self.d_ind[idx], self.t_ind[idx], self.i_ind[idx]

        # Return (features, labels)
        elif self.return_type == 'encoding':
            # if features (embeddings and features) not saved -- save them in self.feat_enc
            if self.feat_enc is not None:
                pass
            else:
                d = np.stack(self.dataset['drug_encoding'].values)
                t = np.stack(self.dataset['target_encoding'].values)
                i = self.dataset[self.label_col].values
                i = torch.tensor(i, dtype=torch.float32)

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

            return self.feat_enc[idx, :], self.i_ind[idx]