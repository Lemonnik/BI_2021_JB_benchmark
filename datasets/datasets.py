from urllib.error import URLError
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from datasets.DTI_dataset import DTI_dataset
import torch


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('logger')


class Davis(DTI_dataset):

    _download_link = "https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/DAVIS/"

    def __init__(self, *args, **kwars) -> None:
        self.label_encoder = LabelEncoder()
        super().__init__(link=self._download_link, *args, **kwars)

    
    def _load_data(self) -> None:
        logger.debug("Loading data...")
        # Get SMILES and TargetSequences encodings
        self._encode_by_ind()
        # Now load data into self.features
        df = pd.read_csv(os.path.join(self.raw_folder, self.filenames['full']), index_col=0)

        self.add_feature(feat_name="SMILES", feat_values=df["SMILES"].values.tolist())
        self.add_feature(feat_name="Sequence", feat_values=df["Target Sequence"].values.tolist())
        self.add_feature(feat_name="Label", feat_values=df["Label"].values.tolist())

        self.add_feature(feat_name="DrugName", feat_values=self.entity_to_ind(self.features["SMILES"]))
        self.add_feature(feat_name="DrugInd", feat_values=self.entity_to_ind(self.features["SMILES"]))
        self.add_feature(feat_name="ProtName", feat_values=self.entity_to_ind(self.features["Sequence"]))
        self.add_feature(feat_name="ProtInd", feat_values=self.entity_to_ind(self.features["Sequence"]))

        logger.debug("LOADED!")

    def _load_processed_data(self) -> None:
        """Load processed data (if exist) and store features in self.features dictionary."""
        # Not implemented yet
    
    def _update_processed_data(self) -> None:
        """Update processed data (rewrite files train/val/test.csv"""
        # Not implemented yet

    def _save_processed_data(self) -> None:
        """Save processed data in processed_folder (rewrite files train/val/test.csv)."""
        pass
        # with open(os.path.join(self.processed_folder, ), "") as :

    def download(self, link: str) -> None:
        """Download the data if it doesn't exist already."""

        if self._check_exists(self.raw_folder):
            logger.debug("Raw dataset already downloaded.")
            return
        
        os.makedirs(self.raw_folder, exist_ok=True)

        # download and save all raw datasets (train/val/test)
        dataset = pd.DataFrame()
        for filename in self.filenames.values():
            url = f"{link}{filename}"
            try:
                print(f"Downloading {url}")
                df = pd.read_csv(url, index_col=0)
                dataset = dataset.append(df[["SMILES","Target Sequence", "Label"]], ignore_index=True).drop_duplicates(ignore_index=True)
            except URLError as error:
                print(f"Failed to download {filename}:\n{error}")
                continue
            finally:
                dataset.to_csv(os.path.join(self.raw_folder, self.filenames['full']))
                print()

    def _encode_by_ind(self) -> None:
        """
        Encode drugs and targets by Index
        """
        # If LabelEncoder already exists -- load encodings
        label_encoder_path = os.path.join(self.raw_folder, "label_encoder.pkl")
        if os.path.exists(label_encoder_path):
            logger.debug("Loading existing label encoder...")
            with open(label_encoder_path, 'rb') as le_dump_file:
                self.label_encoder = pickle.load(le_dump_file)
            logger.debug("Loaded succesfully.\n")
            #attribute
            self._n_entities = len(self.label_encoder.classes_)
            return

        # If not:
        # load raw file with full dataset and encode all unique drugs/prots by some IND
        dataset = pd.read_csv(os.path.join(self.raw_folder, self.filenames['full']))
        drugs = dataset["SMILES"].unique()
        prots = dataset["Target Sequence"].unique()
        entities = np.append(drugs, prots)           # all unique drugs and proteins
        
        # save encodings 
        self.label_encoder.fit(entities)
        with open(label_encoder_path, "wb") as le_dump_file:
            pickle.dump(self.label_encoder, le_dump_file)
        logger.debug("Drugs and Proteins were encoded by IND's...")
        # ...now we have encodings stored in label_encoder.pkl and in self.label_encoder

        # attributes
        # TODO: move _unique_proteins/_unique_drugs setting into ``download()`` function 
        self._unique_drugs = self.entity_to_ind(drugs)
        self._unique_proteins = self.entity_to_ind(prots)
        self._n_entities = len(entities)

    def ind_to_entity(self, ind: list) -> List:
        """
        Gets list of drugs/proteins IND's.
        Returns SMILES strings/Target Sequences.
        """
        # return [self.label_encoder.classes_[i] for i in ind]
        return self.label_encoder.inverse_transform(ind).tolist()

    def entity_to_ind(self, s: list) -> List:
        """
        Gets list of SMILES strings/Target Sequences.
        Returns their indexes (IND's).
        """
        return self.label_encoder.transform(s).tolist()

    def __len__(self) -> int:
        # return len(self.features['Label'])
        # TODO: Normal train/test/val split
        ratio = 0.75
        n = int(ratio * len(self.features['Label']))
        if self.mode == 'train':
            return len(self.features['Label'][:n])
        else:
            return len(self.features['Label'][n:])

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        index : int
            Index.

        Returns
        -------
        Tuple
            Tuple of features returned. ``return_type`` parameter allows to
            define what features should be returned.
        """
        feats_to_return = []

        ratio = 0.75
        n = int(ratio * len(self.features['Label']))
        if self.mode == 'test':
            idx = idx + n

        for feat in self._return_type:
            feat_i = self.features[feat][idx]
            # try:
            #     feat_i = torch.tensor(feat_i, dtype=torch.float32)
            # except:
            #     pass
            feats_to_return.append(feat_i)

        return tuple(feats_to_return)