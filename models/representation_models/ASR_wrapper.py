# This source code is licensed under the license found in the
# LICENSE-MIT.txt file in the root directory of this source tree.
import torch
from speechbrain.pretrained import EncoderDecoderASR


class AsrFeatExtractor():
    def __init__(self, device='cuda', sr=16000):
        self.model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech",
            savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
            run_opts={'device': device}
        )
        self.sr = sr

    def extract_feats(self, x):
        """ x - audio of shape [B, T] """
        lengths = torch.FloatTensor([xi.shape[-1] for xi in x])
        x = self.model.audio_normalizer(x, self.sr)
        x = self.model.encode_batch(x, lengths)
        return x