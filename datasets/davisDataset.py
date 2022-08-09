import logging
import os
import pickle
from typing import List
from urllib.error import URLError

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets.datasetWithLabelEncoder import DatasetWithLabelEncoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('logger')


class Davis(DatasetWithLabelEncoder):
    _download_link = "https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/DAVIS/"
    _label_encoder_filename = "label_encoder.pkl"
    _files_to_download = {
        'train': 'train.csv',
        'test': 'test.csv',
        'val': 'val.csv'}
    _stored_files = {
        'full': 'full.csv'
    }

    def __init__(self,
                 root: str,
                 mode: str = 'train',
                 force_download: bool = False,
                 return_type=None) -> None:
        if return_type is None:
            return_type = ['DrugInd', 'ProtInd', 'Label']
        self.label_encoder = LabelEncoder()
        self._label_encoder_path = None
        super().__init__(root, self._download_link, mode, force_download, return_type)

    def _load_raw_data(self) -> None:
        logger.debug("Loading data...")

        dataset = pd.read_csv(os.path.join(self.raw_folder, self._stored_files['full']))
        drugs = dataset["SMILES"].unique()
        prots = dataset["Target Sequence"].unique()

        self._encode_by_ind(drugs, prots)

        self.add_feature(feat_name="SMILES", feat_values=dataset["SMILES"].values.tolist())
        self.add_feature(feat_name="Sequence", feat_values=dataset["Target Sequence"].values.tolist())
        self.add_feature(feat_name="Label", feat_values=dataset["Label"].values.tolist())

        self.add_feature(feat_name="DrugName", feat_values=self.entity_to_ind(self.features["SMILES"]))
        self.add_feature(feat_name="DrugInd", feat_values=self.entity_to_ind(self.features["SMILES"]))
        self.add_feature(feat_name="ProtName", feat_values=self.entity_to_ind(self.features["Sequence"]))
        self.add_feature(feat_name="ProtInd", feat_values=self.entity_to_ind(self.features["Sequence"]))

        logger.debug("LOADED!")

    def _load_processed_data(self) -> None:
        """Load processed data (if exist) and store features in `self.features` dictionary."""

        # load features
        self.features = {}
        df = pd.read_csv(os.path.join(self.processed_folder, self._stored_files['full']), index_col=0)
        for col in df.columns:
            self.add_feature(feat_name=col, feat_values=df[col].values.tolist())

        # load encodings
        self._load_label_encoder()

    # I guess this function is unnecessary
    #
    # def _update_processed_data(self) -> None:
    #     """Update processed data (rewrite files train/val/test.csv"""
    #     raise NotImplementedError()

    def _save_processed_data(self) -> None:
        """Save processed data in processed_folder (rewrites file every time)."""
        os.makedirs(self.processed_folder, exist_ok=True)

        df = pd.DataFrame.from_dict(self.features)
        df.to_csv(os.path.join(self.processed_folder, self._stored_files['full']))

    def download_from_url(self) -> None:
        """Download the data if it doesn't exist already."""
        os.makedirs(self.raw_folder, exist_ok=True)

        # download and save all raw datasets (train/val/test)
        dataset = pd.DataFrame()
        for filename in self._files_to_download.values():
            url = f"{self._download_link}{filename}"
            try:
                print(f"Downloading {url}")
                df = pd.read_csv(url, index_col=0)
                dataset = dataset.append(df[["SMILES", "Target Sequence", "Label"]], ignore_index=True).drop_duplicates(
                    ignore_index=True)
            except URLError as error:
                print(f"Failed to download {filename}:\n{error}")
                continue
        dataset.to_csv(os.path.join(self.raw_folder, self._stored_files['full']))

    @property
    def all_drugs(self) -> List[str]:
        raise NotImplementedError

    @property
    def all_proteins(self) -> List[str]:
        raise NotImplementedError
