import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def check_exists(folder_name, file_name) -> bool:
    """Checks file with specified name exists in folder_name"""
    return os.path.exists(os.path.join(folder_name, file_name))


class DtiDataset(Dataset, ABC):
    """
    Base Class for making datasets which are compatible with our DTI benchmark.
    It is necessary to override the ``__getitem__`` and ``__len__`` methods.

    Parameters
    ----------
    root : str
        Root directory of dataset.
    download_link : str
        Hyperlink from which dataset can be downloaded.
    mode : str
        Which part of data should be loaded in class instance.
        Possible variants are 'all'/'train'/'val'/'test'.
    force_download : bool
        Should the dataset be downloaded using [link] or not.
    return_type : list[str]
        Defines what features will be returned by ``__getitem__``.
        All features must be in ``self.features`` dictionary.
    """

    def __init__(self,
                 root: str,
                 download_link: Optional[str] = None,
                 mode: str = 'train',
                 force_download: bool = False,
                 return_type=None) -> None:
        if return_type is None:
            return_type = ['DrugInd', 'ProtInd', 'Label']
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self._n_entities = None
        self.root = root
        self.mode = mode
        self.features = {rt: None for rt in return_type}
        self._return_type = return_type
        self.link = download_link
        self.force_download = force_download

        if force_download:
            self.download_from_url()
            self._load_raw_data()
        else:
            # TODO change folder to preprocessed folder
            if os.path.exists(self.raw_folder):
                try:
                    self._load_processed_data()
                    return
                except (FileNotFoundError, IOError):
                    # Todo: Create class for loading error and change for log
                    print("Processed data not found. Loading from raw data.")
            if os.path.exists(self.raw_folder):
                try:
                    self._load_raw_data()
                    return
                except (FileNotFoundError, IOError):
                    # Todo: Create class for loading error
                    print("Processed data not found. Loading from raw data.")
            self.download_from_url()
            self._load_raw_data()

    def add_feature(self, feat_name: str, feat_values: list) -> None:
        """
        Adds new feature.

        Parameters
        ----------
        feat_name : str
            Name of the feature.
        feat_values : list
            Column values.
        """

        # TODO: add many features with one function call

        self.features[feat_name] = feat_values
        self._save_processed_data()

    @abstractmethod
    def _save_processed_data(self) -> None:
        """Save processed data in processed_folder."""
        pass

    @abstractmethod
    def _load_raw_data(self) -> None:
        """Load raw data (if exist) and store features in self.features dictionary."""
        ...

    @abstractmethod
    def _load_processed_data(self) -> None:
        """Load processed data (if exist) and store features in self.features dictionary."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple:
        """
        Parameters
        ----------
        index : int
            Index.

        Returns
        -------
        Tuple
            Tuple of features returned. ``return_type`` parameter allows
            defining what features should be returned.
        """
        ...

    @abstractmethod
    def download_from_url(self) -> None:
        ...

    @property
    @abstractmethod
    def all_drugs(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def all_proteins(self) -> List[str]:
        ...

    @property
    def n_entities(self) -> int:
        return self._n_entities

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def return_type(self) -> List[str]:
        return self._return_type

    # Todo: Add val checker
    @return_type.setter
    def return_type(self, val):
        if type(val) != list:
            raise ValueError(f"Expected val to be list of str. Got {type(val)}")
        for prop in val:
            if prop not in self.features.keys():
                raise ValueError(f"Expected all return values to be features. {prop} not in features")
        self._return_type = val

    @property
    def return_options(self) -> List[str]:
        return list(self.features.keys())
