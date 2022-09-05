import logging
import os
import pickle
from abc import ABC
from typing import List, Tuple, Optional

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from datasets.DtiDataset import DtiDataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('logger')


class DatasetWithLabelEncoder(DtiDataset, ABC):
    _label_encoder_filename = "label_encoder.pkl"

    def __init__(self,
                 root: str,
                 download_link: Optional[str] = None,
                 mode: str = 'train',
                 force_download: bool = False,
                 return_type: list = None,
                 load_from_raw: bool = False) -> None:
        if return_type is None:
            return_type = ['DrugInd', 'ProtInd', 'Label']
        self.label_encoder = LabelEncoder()
        self._label_encoder_path = None
        super().__init__(root, download_link, mode, force_download, return_type, load_from_raw)

    def _encode_by_ind(self, drugs, proteins) -> None:
        """
        Encode drugs and targets by Index.
        """
        entities = np.append(drugs, proteins)
        self.label_encoder.fit(entities)
        self._n_entities = len(self.label_encoder.classes_)
        self._label_encoder_path = os.path.join(self.raw_folder, self._label_encoder_filename)

        with open(self._label_encoder_path, "wb") as le_dump_file:
            pickle.dump(self.label_encoder, le_dump_file)
        logger.debug("Drugs and Proteins were encoded by IND's...")

        self._unique_drugs = self.entity_to_ind(drugs)
        self._unique_proteins = self.entity_to_ind(proteins)
        self._n_entities = len(entities)

    def _load_label_encoder(self) -> None:
        # Todo: add checks and errors
        # TODO: do we need individual LabelEncoder for processed data folder?
        self._label_encoder_path = os.path.join(self.raw_folder, self._label_encoder_filename)
        if os.path.exists(self._label_encoder_path):
            logger.debug("Loading existing label encoder...")
            with open(self._label_encoder_path, 'rb') as le_dump_file:
                self.label_encoder = pickle.load(le_dump_file)
            logger.debug("Loaded successfully.\n")
            # attribute
            self._n_entities = len(self.label_encoder.classes_)

    def ind_to_entity(self, ind: list) -> list:
        """
        Converts list of indexes (labels) into list of entities (SMILES or Protein sequences).

        Parameters
        ----------
        ind : list
            List of drugs/proteins indexes (labels).

        Returns
        -------
        list
            SMILES strings/Target Sequences.
        """
        # return [self.label_encoder.classes_[i] for i in ind]
        return self.label_encoder.inverse_transform(ind).tolist()

    def entity_to_ind(self, s: list) -> list:
        """
        Converts list of entities (SMILES or Protein sequences) into their indexes (labels).

        Parameters
        ----------
        s : list
            List of SMILES strings/Target Sequences.

        Returns
        -------
        list
            Indexes (labels)
        """
        return self.label_encoder.transform(s).tolist()

    def __len__(self) -> int:
        """
        Length of the dataset (number of drug-target pairs for which there is a Label).
        If 'self.mode' in train/test/val then length of the chosen (train/test/val) part will be returned.
        """
        # return len(self.features['Label'])
        # TODO: Normal train/test/val split
        ratio = 0.75
        n = int(ratio * len(self.features['Label']))
        if self.mode == 'all':
            return len(self.features['Label'])
        elif self.mode == 'train':
            return len(self.features['Label'][:n])
        else:
            return len(self.features['Label'][n:])

    def __getitem__(self, idx) -> tuple:
        """
        Parameters
        ----------
        idx : int
            Index.

        Returns
        -------
        Tuple
            Tuple of features returned.
            ``return_type`` parameter allows
            defining what features should be returned.
        """
        feats_to_return = []

        ratio = 0.75
        n = int(ratio * len(self.features['Label']))
        if self.mode == 'test':
            idx = idx + n

        for feat in self._return_type:
            # print(feat, idx, self.features[feat])
            feat_i = self.features[feat][idx]
            try:
                feat_i = torch.tensor(feat_i, dtype=torch.float32)
            except:
                pass
            feats_to_return.append(feat_i)

        return tuple(feats_to_return)
