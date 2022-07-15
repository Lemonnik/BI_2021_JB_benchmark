import torch
from torch.nn import Module

from abc import ABC, abstractmethod
import os

class DTI_model(Module, ABC):
    """
    Base Class for making models which are compatible with our DTI benchmark.
    It is necessary to have ``return_type`` attribute and ``__call__`` method.

    Attributes
    ----------
    return_type: list
        Defines what features should be returned by ``__getitem__`` method of DTI dataset.
    """


    @property
    def return_type(self) -> list:
        return self._return_type

    @abstractmethod
    def __call__(self, data, train=True):
        """
        Should call ``forward`` method and return:
            loss in `train` mode
            correct_labels, predicted_labels, predicted_scores if not in `train` mode
        """
        ...