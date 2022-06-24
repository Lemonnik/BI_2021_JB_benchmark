from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

class DTI_dataset(Dataset, ABC):
    """
    Base Class For making datasets which are compatible with torchvision.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.

    Args:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    filenames = {
        'train': 'train.csv', 
        'test': 'test.csv', 
        'val': 'val.csv'
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
        self.mode = mode  # training/validation/test set
        self.features = {rt:None for rt in return_type}  # container for smiles, sequences, labels and all other features
        self.return_type = return_type  # what features from self.features Class should return using __getitem__ (see example)
        self.return_options = self.features.keys()

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
        """Checks if all files (train/val/test) exist in folder_name"""
        return all(
            os.path.exists(os.path.join(folder_name, filename))
            for filename in self.filenames.values()
        )

    def add_feature(self, feat_name: str, feat_values: list) -> None:
        '''
        Adds new feature.

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
        self.features[feat_name] = feat_values


    @abstractmethod
    def _load_data(self) -> None:
        """Load raw data (if exist) and store features in self.features dictionary."""
        ...

    @abstractmethod
    def _load_processed_data(self) -> None:
        """Load processed data (if exist) and store features in self.features dictionary."""
        ...

    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def get_drug_names(self) -> List[str]:
        return self.features['DrugName']
    
    @property
    def get_protein_names(self) -> List[str]:
        return self.features['ProtName']
    
    @property
    def get_labels(self) -> List[int]:
        return self.features['label']
    
    @property
    def n_entities(self) -> int:
        return self._n_entities
    
    @n_entities.setter
    def n_entities(self, val):
        self._n_entities = val


    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        ...