# Implementation was based on code from: https://github.com/facebookresearch/denoiser with the following license:
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import time

import torch as th
from torch import nn
from torch.nn import functional as F

from external_files.resample import upsample2, downsample2
from external_files.utils import capture_init
from models.dataclass_configurations.demucs_config import DemucsConfig
from models.dataclass_configurations.features_config import FeaturesConfig
from models.representation_models.ft_conditioning_modules.ft_conditioner import FtConditioner


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class PhoneticAwareDemucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.

    """
    def __init__(self, args):
        super().__init__()
        self._init_args_kwargs = (args, dict())
        self.demucs_config = DemucsConfig(**args.demucs)
        self. feature_config = FeaturesConfig(**args.features_config)

        if self.demucs_config.resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.sample_rate = args.dset.sample_rate
        self.include_ft = self.feature_config.include_ft and not self.feature_config.use_as_conditioning if \
            self.feature_config is not None else False
        self.ft_conditioning = self.feature_config is not None and self.feature_config.use_as_conditioning
        self.ft_conditioner = FtConditioner(self.feature_config) if self.ft_conditioning else None

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        activation = nn.GLU(1) if self.demucs_config.glu else nn.ReLU()
        ch_scale = 2 if self.demucs_config.glu else 1

        chin = self.demucs_config.chin
        hidden = self.demucs_config.hidden
        kernel_size = self.demucs_config.kernel_size
        stride = self.demucs_config.stride
        chout = self.demucs_config.chout
        self.floor = self.demucs_config.floor
        if isinstance(self.floor, tuple):
            self.floor = self.floor[0]

        for index in range(self.demucs_config.depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(self.demucs_config.growth * hidden), self.demucs_config.max_hidden)

        self.lstm = BLSTM(chin, bi=not self.demucs_config.causal)
        if self.demucs_config.rescale:
            rescale_module(self, reference=self.demucs_config.rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.demucs_config.resample)
        for idx in range(self.demucs_config.depth):
            length = math.ceil((length - self.demucs_config.kernel_size) / self.demucs_config.stride) + 1
            length = max(length, 1)
        for idx in range(self.demucs_config.depth):
            length = (length - 1) * self.demucs_config.stride + self.demucs_config.kernel_size
        length = int(math.ceil(length / self.demucs_config.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix, cond_features=None):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.demucs_config.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            factor = self.floor + std
            mix = mix / factor
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.demucs_config.resample == 2:
            x = upsample2(x)
        elif self.demucs_config.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        if self.ft_conditioning:
            x = self.ft_conditioner(x, cond_features)
        if self.include_ft and not self.feature_config.get_ft_after_lstm:
            features = x
        x, _ = self.lstm(x)
        if self.include_ft and self.feature_config.get_ft_after_lstm:
            features = x
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.demucs_config.resample == 2:
            x = downsample2(x)
        elif self.demucs_config.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        if self.include_ft:
            return std * x, features
        return std * x
