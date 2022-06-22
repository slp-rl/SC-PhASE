from abc import ABC, abstractmethod
from typing import Tuple, Union, Any
from torch.utils.data import DataLoader


class DsetBuilder(ABC):

    @staticmethod
    @abstractmethod
    def get_tr_cv_tt_loaders(args: dict, model: Any) -> Tuple[DataLoader, DataLoader, DataLoader, Union[DataLoader, None]]:
        """
        this method should receive a dictionary of arguments and return train, valid, and test data loaders
        Optional: the 4th argument could be an additional dataloader used to generate samples at the end of
        each evaluation step; if not given - test dataloader would be used.
        """
        raise NotImplementedError()
