import gzip
import logging
import os
from io import BytesIO
from urllib.error import URLError

import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

from datasets.datasetWithLabelEncoder import DatasetWithLabelEncoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('logger')


class Davis(DatasetWithLabelEncoder):
    _download_link = "https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/DAVIS/"
    _label_encoder_filename = "label_encoder.pkl"
    filenames = {
        'train': 'train.csv',
        'test': 'test.csv',
        'val': 'val.csv',
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

        dataset = pd.read_csv(os.path.join(self.raw_folder, self.filenames['full']))
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
        # """Load processed data (if exist) and store features in self.features dictionary."""
        # if os.path.exists(self._label_encoder_path) or self.download_from_url:
        #     logger.debug("Loading existing label encoder...")
        #     with open(self._label_encoder_path, 'rb') as le_dump_file:
        #         self.label_encoder = pickle.load(le_dump_file)
        #     logger.debug("Loaded successfully.\n")
        #     # attribute
        #     self._n_entities = len(self.label_encoder.classes_)
        # else:
        #     self._encode_by_ind(entities=)
        raise NotImplementedError()
    
    def _update_processed_data(self) -> None:
        """Update processed data (rewrite files train/val/test.csv"""
        raise NotImplementedError()

    def _save_processed_data(self) -> None:
        """Save processed data in processed_folder (rewrite files train/val/test.csv)."""
        # with open(os.path.join(self.processed_folder, ), "") as :
        pass

    def download_from_url(self) -> None:
        """Download the data if it doesn't exist already."""
        os.makedirs(self.raw_folder, exist_ok=True)

        # download and save all raw datasets (train/val/test)
        dataset = pd.DataFrame()
        for filename in self.filenames.values():
            url = f"{self._download_link}{filename}"
            try:
                print(f"Downloading {url}")
                df = pd.read_csv(url, index_col=0)
                dataset = dataset.append(df[["SMILES", "Target Sequence", "Label"]], ignore_index=True).drop_duplicates(
                    ignore_index=True)
            except URLError as error:
                print(f"Failed to download {filename}:\n{error}")
                continue
            finally:
                dataset.to_csv(os.path.join(self.raw_folder, filename))
                print()


# TODO: Add encoding of smiles and prot seq from ID
class DtiMinor(DatasetWithLabelEncoder):
    _download_link = 'http://snap.stanford.edu/biodata/datasets/10002/files/ChG-Miner_miner-chem-gene.tsv.gz'
    _downloaded_file_name = "DTIMinor.tsv"
    _label_encoder_filename = "label_encoder.pkl"

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

    def download_from_url(self):
        """
        Download data if it's not presented on a device.
        """

        save_path = os.path.join(self.raw_folder, self._downloaded_file_name)

        if self.download_from_url:
            if not os.path.exists(self.raw_folder):
                os.makedirs(self.raw_folder, exist_ok=True)

            # download file
            resp = requests.get(self._download_link, allow_redirects=True)
            gzipfl = gzip.GzipFile(fileobj=BytesIO(resp.content))

            with open(save_path, 'w') as file:
                for row in gzipfl.readlines():
                    file.write(row.decode())

    def _load_raw_data(self) -> None:
        logger.debug("Loading data...")

        dataset = pd.read_csv(os.path.join(self.raw_folder, self._downloaded_file_name))
        drugs = dataset["SMILES"].unique()
        prots = dataset["Target Sequence"].unique()

        self._encode_by_ind(drugs, prots)

    def _load_processed_data(self) -> None:
        raise NotImplementedError()

    def _save_processed_data(self) -> None:
        raise NotImplementedError()

    def _update_processed_data(self) -> None:
        raise NotImplementedError()
