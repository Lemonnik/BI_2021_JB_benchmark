import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from pybiomed_helper import CalculateConjointTriad


def _encode_by_ind(self):
    """
    Drug and Proteins encoding by ID's.
    """
    # Encode
    drug_encoding = {d: i for d, i in zip(self.dataset[self.drug_col].unique(), range(self.n_drugs))}
    prot_encoding = {p:i for p,i in zip(self.dataset[self.target_col].unique(), range(self.n_drugs, self.n_drugs+self.n_prots))}
    ind_encoding = {**drug_encoding, **prot_encoding}  # merge two dicts
    reversed_encoding = {v: k for k, v in ind_encoding.items()}

    # Add a new column to dataset
    self.dataset['Drug_IND'] = [drug_encoding[i] for i in self.dataset[self.drug_col]]
    self.dataset['Targ_IND'] = [prot_encoding[i] for i in self.dataset[self.target_col]]

    return ind_encoding, reversed_encoding


def encode_drugs(drugs, enc_type='morgan'):
    """
    Encode all drugs.

    Parameters
    ----------
    drugs : list
        SMILES string corresponding to particular drug.
    enc_type : {'morgan'}
        Encoding type.

    Returns
    -------
    features: list
        Drugs encoded by enc_type.
    """
    unique_drugs = pd.Series(drugs).unique()
    unique_drugs_encoded = pd.Series(unique_drugs).apply(_smiles2morgan)
    unique_drugs_encoded_dict = dict(zip(unique_drugs, unique_drugs_encoded))
    
    all_drugs_encoded = [unique_drugs_encoded_dict[i] for i in drugs]
    return all_drugs_encoded


def encode_targets(prots, enc_type='conj_triad'):
    """
    Encode all proteins.

    Parameters
    ----------
    prots : list
        List of protein sequences.
    enc_type : {'conj_triad'}
        Encoding type.

    Returns
    -------
    features: array[1024]
        Proteins encoded by enc_type.
    """
    unique_prots = pd.Series(prots).unique()
    unique_prots_encoded = pd.Series(unique_prots).apply(_target2ct)
    unique_prots_encoded_dict = dict(zip(unique_prots, unique_prots_encoded))
    
    all_prots_encoded = [unique_prots_encoded_dict[i] for i in prots]
    return all_prots_encoded


def _smiles2morgan(s):
    """
    Converts SMILES to Morgan Fingerprint.

    Parameters
    ----------
    s : str
        SMILES string corresponding to particular drug.

    Returns
    -------
    features: array[1024]
        Drug encoded by Morgan's Fingerprint.
    """
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print(f'Some problems with {s}.. Array of 0\'s will be returned')
        features = np.zeros((1024, ))
    return features


def _target2ct(s):
    """
    Converts Protein Sequence using Conjoint Triad.

    Parameters
    ----------
    s : str
        Protein amino acid sequence.

    Returns
    -------
    features: array[343]
        Protein encoded using Conjoint Triad.
    """
    try:
        features = CalculateConjointTriad(s)
    except:
        print(f'Some problems with {s}.. Array of 0\'s will be returned')
        features = np.zeros((343, ))
    return features


def get_dataset_path(df):
    paths = {'davis': 'https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/DAVIS/',
             'bindingDB': 'https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/BindingDB/'
    }
    return paths[df]