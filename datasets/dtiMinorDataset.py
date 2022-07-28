import gzip
import logging
import os
from io import BytesIO

import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

from datasets.datasetWithLabelEncoder import DatasetWithLabelEncoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('logger')


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
