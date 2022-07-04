from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import pickle
import torch

class DTI_dataset(Dataset, ABC):
    """
    Base Class for making datasets which are compatible with our DTI benchmark.
    It is necessary to override the ``__getitem__`` and ``__len__`` methods.

    Parameters
    ----------
    root : str
        Root directory of dataset.
    link : str
        Hyperlink from which dataset can be downloaded.
    mode : str
        Which part of data should be loaded in class instance.
        Possible variants are 'train'/'val'/'test'.
    download : bool
        Should the dataset be downloaded using [link] or not.
    return_type : list[str]
        Defines what features will be returned by ``__getitem__``.
        All features must be in ``self.features`` dictionary.

    Attributes
    ----------
    filenames : dict
        Name of files in which different parts (train/validation/test)
        of dataset will be stored.
    return_options : list[str]
        Allows to see ``features.keys()`` -- all possible features,
        that can be returned by ``__getitem__``.
    n_entities : int
        Nubmer of entities in full (train+val+test) dataset.
    """

    filenames = {
        'train': 'train.csv', 
        'test': 'test.csv', 
        'val': 'val.csv',
        'full': 'full.csv'
        }

    def __init__(
        self,
        root: str,
        link: Optional[str] = None,
        mode: str = 'train',
        download: bool = False,
        return_type: list = ['DrugInd', 'ProtInd', 'Label'],
        # *,
        ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self._n_entities = None
        self.root = root
        self.mode = mode
        self.features = {rt:None for rt in return_type}
        self._return_type = return_type

        # if preprocessed data already exist -- we need to load it and that's all
        if self._check_exists(self.processed_folder):
            self._load_processed_data()
            return

        # download dataset if needed
        if download:
            self.download(link)

        # if raw_folder does not contain all required files (train/val/test) -- raise error
        if not self._check_exists(self.raw_folder):
            raise RuntimeError(f"Dataset not found. You should download it or have it stored in {self.raw_folder}")

        # Load data from raw_folder. The required minimum is Drug Names, Protein Names and Labels (interactions)
        self._load_data()

    def _check_exists(self, folder_name) -> bool:
        """Checks if full.csv file exist in folder_name"""
        return os.path.exists(os.path.join(folder_name, 'full.csv'))

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
    def _load_data(self) -> None:
        """Load raw data (if exist) and store features in self.features dictionary."""
        ...

    @abstractmethod
    def _load_processed_data(self) -> None:
        """Load processed data (if exist) and store features in self.features dictionary."""
        ...

    @abstractmethod
    def _save_processed_data(self) -> None:
        """Save processed data in processed_folder."""
        ...

    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def all_drugs(self) -> List[str]:
        return self._unique_drugs
    
    @property
    def all_proteins(self) -> List[str]:
        return self._unique_proteins
    
    @property
    def n_entities(self) -> int:
        return self._n_entities
    
    @property
    def return_type(self) -> List[str]:
        return self._return_type

    @return_type.setter
    def return_type(self, val):
        self._return_type = val

    @property
    def return_options(self) -> List[str]:
        return list(self.features.keys())


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
            Tuple of features returned. ``return_type`` parameter allows to
            define what features should be returned.
        """
        ...