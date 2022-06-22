from typing import Dict, Tuple, Union, Any
from torch.utils.data import DataLoader
from dset_builders.dset_builder import DsetBuilder
from dset_builders.speech_enhancement.noisy_clean import NoisyCleanBuilder


class DsetBuilderFactory:
    valid_builders: Dict[str, DsetBuilder] = {
        "noisy_clean": NoisyCleanBuilder
    }

    @staticmethod
    def get_loaders(args: dict, model: Any) -> Tuple[DataLoader, DataLoader, DataLoader, Union[DataLoader, None]]:
        if args.dset_builder not in DsetBuilderFactory.valid_builders.keys():
            raise ValueError(f"DsetBuilder: {args.dset_builder} is not supported by DsetBuilderFactory.\nPlease make sure implementation is valid.")
        else:
            return DsetBuilderFactory.valid_builders[args.dset_builder].get_tr_cv_tt_loaders(args, model)