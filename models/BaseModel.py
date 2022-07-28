from abc import ABC

from torch.nn import Module


class DtiModel(Module, ABC):
    """
    Base Class for making models which are compatible with our DTI benchmark.
    It is necessary to have ``return_type`` attribute and ``__call__`` method.

    Attributes
    ----------
    return_type: list
        Defines what features should be returned by ``__getitem__`` method of DTI dataset.
        Read-only.
    """

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def return_type(self):
        return self._return_type
