# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-MIT.txt file in the root directory of this source tree.
import json
import logging
import math
import os
import re
from abc import ABC
from typing import Tuple, Union

import torchaudio
from torch.utils.data import DataLoader

from dset_builders.dset_builder import DsetBuilder
from external_files import distrib
from external_files.dsp import convert_audio
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class Audioset:
    # This source code is licensed under the license found in the
    # LICENSE-META.txt file in the root directory of this source tree.
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, convert=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]
            if self.convert:
                out = convert_audio(out, sr, target_sr, target_channels)
            else:
                if sr != target_sr:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{target_sr}, but got {sr}")
                if out.shape[0] != target_channels:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{target_channels}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out


def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.

    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


class NoisyCleanSet:
    # This source code is licensed under the license found in the
    # LICENSE-META.txt file in the root directory of this source tree.
    def __init__(self, json_dir, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None, with_path=False):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate,
              "with_path": with_path}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)


class NoisyCleanBuilder(DsetBuilder, ABC):

    @staticmethod
    def get_tr_cv_tt_loaders(args: dict, model) -> Tuple[DataLoader, DataLoader, DataLoader, Union[DataLoader, None]]:
        """
        this method should receive a dictionary of arguments and return train, valid, and test data loaders
        Optional: the 4th argument could be an additional dataloader used to generate samples at the end of
        each evaluation step; if not given - test dataloader would be used.
        """

        length = int(args.dset.segment * args.dset.sample_rate)
        stride = int(args.dset.stride * args.dset.sample_rate)
        # Demucs requires a specific number of samples to avoid 0 padding during training
        if hasattr(model, 'valid_length'):
            length = model.valid_length(length)
        kwargs = {"matching": args.dset.matching, "sample_rate": args.dset.sample_rate}
        # Building datasets and loaders
        tr_dataset = NoisyCleanSet(
            args.dset.train, length=length, stride=stride, pad=args.dset.pad, **kwargs)
        tr_loader = distrib.loader(
            tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        if args.dset.valid and os.path.exists(args.dset.valid):
            cv_dataset = NoisyCleanSet(args.dset.valid, **kwargs)
            cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
        else:
            cv_loader = None
        if args.dset.test and os.path.exists(args.dset.test):
            tt_dataset = NoisyCleanSet(args.dset.test, **kwargs, with_path=True)
            tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
        else:
            tt_loader = None
        if args.dset.enh and os.path.exists(args.dset.enh):
            enh_dataset = NoisyCleanSet(args.dset.enh, **kwargs, with_path=True)
            enh_loader = distrib.loader(enh_dataset, batch_size=1, num_workers=args.num_workers)
        # elif args.dset.test and os.path.exists(args.dset.test):
        #     enh_dataset = NoisyCleanSet(args.dset.test, **kwargs, with_path=False)
        #     enh_loader = distrib.loader(enh_dataset, batch_size=1, num_workers=args.num_workers)
        else:
            enh_loader = None
        return tr_loader, cv_loader, tt_loader, enh_loader
