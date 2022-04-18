#
from torch_geometric.data import Dataset
# ДЛЯ КОДИРОВАНИЯ ЛЕКАРСТВ 
# rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import DataStructs
# ДЛЯ КОДИРОВАНИЯ БЕЛКОВ 
from pybiomed_helper import CalculateConjointTriad

import numpy as np
import pandas as pd

import torch





class LoadDavis(Dataset):
    def __init__(self, mode='train', return_type = 'ind'):
        '''
        Пока не придумал как сделать по-человечески:
        mode -- должна давать на выход что-то из train/val/test
        return_type -- что возвращать (индексы или же encoding, для разных моделей нужно разное)

        Наверное сейчас многовато всего в self хранится
        '''
        dataFolder = 'https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/DAVIS/'
        davis_train = pd.read_csv(dataFolder + '/train.csv', index_col=0)
        davis_test = pd.read_csv(dataFolder + '/test.csv', index_col=0)
        davis_val = pd.read_csv(dataFolder + '/val.csv', index_col=0)
        davis = davis_train.append(davis_test, ignore_index=True).append(davis_val, ignore_index=True).drop_duplicates(ignore_index=True)

        self.dataset = davis                                          # полный датасет
        self.drug_col = 'SMILES'
        self.target_col = 'Target Sequence'
        self.label_col = 'Label'

        self.n_relations = len(self.dataset)                        # число связей
        self.n_drugs = len(self.dataset[self.drug_col].unique())      # число лекарств
        self.n_prots = len(self.dataset[self.target_col].unique())    # число белков
        self.n_entities = self.n_drugs + self.n_prots               # число узлов (лекарств + белков)

        # кодирование по индексам для моделей типа DistMult
        self.entity_to_ind, self.ind_to_entity = self._encode_by_ind()
        # закодировать лекарства и белки
        self._encode_drugs()
        self._encode_targets()

        # что возвращать: train/val/test
        self.mode = mode

        # создавать лоадер из индексов или энкодингов
        self.return_type = return_type

        # TODO: деление датасета ('Drug_ID' тоже наверное в переменную)
        d = self.dataset['Drug_IND'].values
        t = self.dataset['Targ_IND'].values
        i = self.dataset[self.label_col].values
        self.d_ind = torch.tensor(d, dtype=torch.float32)
        self.t_ind = torch.tensor(t, dtype=torch.float32)
        self.i_ind = torch.tensor(i, dtype=torch.float32)

        # застаканные (в будущем) тензоры для возврата энкодингов+фичей
        self.feat_enc = None


    def add_feature(self, feat_name, feat_values):
        '''
        Добавить в датасет колонку с фичами (сделано для добавления эмбеддингов)
        '''
        self.dataset[feat_name] = feat_values


    def get_entity_by_ind(self, i):
        '''
        Возвращает SMILES лекарства/Target Sequence белка по его индексу
        '''
        return self.ind_to_entity.get(i, None)


    def get_ind_by_sequence(self, s):
        '''
        Возвращает индекс белка/лекарства по его SMILES/Target Sequence
        '''
        return self.entity_to_ind.get(s, None)


    def _encode_by_ind(self):
        '''
        Кодирование лекарств и белков 
        '''
        # кодируем
        drug_encoding = {d:i for d,i in zip(self.dataset[self.drug_col].unique(), range(self.n_drugs))}
        prot_encoding = {p:i for p,i in zip(self.dataset[self.target_col].unique(), range(self.n_drugs, self.n_drugs+self.n_prots))}
        ind_encoding = {**drug_encoding, **prot_encoding}  # merge two dicts
        reversed_encoding = {v: k for k, v in ind_encoding.items()}

        # добавляем новую колонку ('Drug_ID' тоже наверное в переменную)
        self.dataset['Drug_IND'] = [drug_encoding[i] for i in self.dataset[self.drug_col]]
        self.dataset['Targ_IND'] = [prot_encoding[i] for i in self.dataset[self.target_col]]

        return ind_encoding, reversed_encoding


    def _encode_drugs(self):
        '''
        Закодировать все лекарства
        '''
        unique_drugs = pd.Series(self.dataset[self.drug_col].unique()).apply(self._smiles2morgan)
        unique_drugs_dict = dict(zip(self.dataset[self.drug_col].unique(), unique_drugs))
        self.dataset['drug_encoding'] = [unique_drugs_dict[i] for i in self.dataset[self.drug_col]]


    def _encode_targets(self):
        '''
        Закодировать все белки
        '''
        unique_prot = pd.Series(self.dataset[self.target_col].unique()).apply(self._target2ct)
        unique_prots_dict = dict(zip(self.dataset[self.target_col].unique(), unique_prot))
        self.dataset['target_encoding'] = [unique_prots_dict[i] for i in self.dataset[self.target_col]]


    def _smiles2morgan(self, s):
        '''
        Кодирование одного лекарства фингерпринтом моргана
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
        Кодирование одного белка
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
        # вернуть (drugs, targets, labels)
        if self.return_type == 'ind':
            return self.d_ind[idx], self.t_ind[idx], self.i_ind[idx]

        # вернуть (features, labels)
        elif self.return_type == 'encoding':
            if self.feat_enc is not None:
                pass
            else:
                d = np.stack(self.dataset['drug_encoding'].values)
                t = np.stack(self.dataset['target_encoding'].values)
                i = self.dataset[self.label_col].values
                i = torch.tensor(i, dtype=torch.float32)

                # как-то тут надо по-умному, чтобы не зависеть от названия колонки...
                if 'embeddings' in self.dataset.columns:
                    if 'embeddings_w_feats' not in self.dataset.columns:
                        e = np.stack(self.dataset.embeddings.values)
                        feats_and_embs = torch.tensor(np.concatenate([d, t, e], axis=1), dtype=torch.float32)
                        self.dataset['embeddings_w_feats'] = feats_and_embs.tolist()  # сохраняем в столбец чтобы потом не мучиться с массивами каждый раз
                    else:
                        feats_and_embs = torch.tensor(np.stack(self.dataset['embeddings_w_feats']), dtype=torch.float32)
                else:
                    print('Embeddings not found')
                    return None
                self.feat_enc = feats_and_embs

            return self.feat_enc[idx, :], self.i_ind[idx]