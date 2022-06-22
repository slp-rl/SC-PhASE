# This source code is licensed under the license found in the
# LICENSE-MIT.txt file in the root directory of this source tree.
import torch
import torch.nn as nn
import torchaudio.transforms
from torch.nn import functional as F
from models.dataclass_configurations.features_config import FeaturesConfig


class LearnedConditioning(nn.Module):

    def __init__(self, num_layers=13, device="cuda"):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(num_layers).float().to(device))

    def forward(self, features):

        # features: [Batch, Channels (20ms quantized time), Feature-dim, Layers]
        x = nn.Softmax()(self.w).expand_as(features) * features
        return torch.sum(x, dim=-1)


class FtConditioner(nn.Module):

    def __init__(self, ft_config: FeaturesConfig = None):
        super().__init__()
        if ft_config is not None:
            self.device = ft_config.device
            self.use_as_conditioning = ft_config.use_as_conditioning
            self.include_ft = ft_config.include_ft
            self.proj = nn.Linear(ft_config.features_dim_for_conditioning + ft_config.features_dim,
                                  ft_config.features_dim_for_conditioning).to(ft_config.device)
            self.features_factor = ft_config.features_factor
            self.merge_method = ft_config.merge_method
            self.get_ft_after_lstm = ft_config.get_ft_after_lstm
            self.features_dim_for_conditioning = ft_config.features_dim_for_conditioning
            if self.use_as_conditioning:
                self.resampler = torchaudio.transforms.Resample(ft_config.features_dim_for_conditioning,
                                                                ft_config.features_dim) if \
                    ft_config.features_dim != ft_config.features_dim_for_conditioning else nn.Identity()
                self.learnable = ft_config.learnable if ft_config.feature_model.lower() == "hubert" and \
                                                        ft_config.layers is not None else False
            else:
                self.learnable = False
            vals = [1 / len(ft_config.layers)] * len(ft_config.layers) if ft_config.layers is not None else [1]
            self.attn = torch.Tensor(vals).float().to(ft_config.device)
            if self.learnable:
                self.ft_linear = LearnedConditioning(len(ft_config.layers), ft_config.device)
        else:
            self.include_ft = False

    def forward(self, x, features):
        if self.use_as_conditioning:
            if self.merge_method == 'inter':
                if self.learnable:
                    features = self.ft_linear(features.permute(1, 2, 3, 0))
                    x_res = F.interpolate(features.permute(0, 2, 1), x.shape[0]).permute(2, 0, 1)
                else:
                    if len(features.shape) == 3:
                        features = [features]
                    x_res = torch.zeros((x.shape[0], features[0].shape[0], features[0].shape[2]))
                    x_res = x_res.to(self.device)
                    factors = nn.Softmax()(self.attn).to(self.device) if self.attn.shape[0] > 1 else self.attn
                    for i, coef in enumerate(factors):
                        x_res = x_res + coef * F.interpolate(features[i].permute(0, 2, 1), x.shape[0]).permute(2, 0, 1)
                        x_res = x_res.to(self.device)
            else:
                raise ValueError("unsupported merge method was given")
            x = torch.cat([x, x_res], dim=-1)
            x = self.proj(x)
        return x
