# This source code is licensed under the license found in the
# LICENSE-MIT.txt file in the root directory of this source tree.
import fairseq
import torch
from omegaconf import ListConfig


class huBERT:
    def __init__(self, model_path, layer, device='cuda'):
        super().__init__()
        self.path = model_path
        self.layer = [la for la in layer] if isinstance(layer, ListConfig) else layer

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.path])
        self.model = models[0]
        self.model = self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_feats(self, x):
        if isinstance(self.layer, list):
            feats = torch.stack([self.model.extract_features(source=x.squeeze(1), padding_mask=None, mask=False,
                                                 output_layer=layer)[0].detach() for layer in self.layer], dim=0)
        else:
            feats, _ = self.model.extract_features(source=x.squeeze(1), padding_mask=None, mask=False,
                                                   output_layer=self.layer)
            feats = feats.detach()
        return feats