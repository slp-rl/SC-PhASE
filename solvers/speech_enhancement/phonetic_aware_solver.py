import logging
import os

import omegaconf
import torch
import torch.nn.functional as F
import torchaudio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any

from custom_loss_functions.stft_loss import MultiResolutionSTFTLoss
from evaluation_metrics.speech_enhancement import get_pesq, get_stoi, get_snr
from external_files import augment, distrib
from external_files.utils import LogProgress, bold
from models.dataclass_configurations.features_config import FeaturesConfig
from solvers.base_solver import BaseSolver
from utils.load_lexical_model import load_lexical_model

logger = logging.getLogger(__name__)


class PhoneticAwareSolver(BaseSolver):
    def __init__(self, data, model, optimizer, args):
        super().__init__(data, model, optimizer, args)

        self.features_config = FeaturesConfig(**args.features_config)
        self.ft_conditioning = self.features_config.use_as_conditioning
        self.ft_regularization = self.features_config.include_ft and not self.ft_conditioning
        self.ft_supervision = self.features_config.use_as_supervision
        logger.info(f"cond: {self.ft_conditioning} | reg: {self.ft_regularization} | sup: {self.ft_supervision}")
        logger.info(f"layers: {args.features_config.layers} ({type(args.features_config.layers)})")
        if self.features_config.feature_model.lower() == "hubert" and self.features_config.use_as_conditioning:
            assert isinstance(self.features_config.layers, omegaconf.listconfig.ListConfig) and len(self.features_config.layers) > 0, \
                "feature_config.layers should be a non-empty list of layer indices"
            self.ft_model = load_lexical_model(self.features_config.feature_model,
                                               self.features_config.state_dict_path,
                                               device=args.device, sr=args.dset.sample_rate,
                                               layer=args.features_config.layers) if \
                self.features_config is not None else None
        else:
            self.ft_model = load_lexical_model(self.features_config.feature_model,
                                               self.features_config.state_dict_path,
                                               device=args.device, sr=args.dset.sample_rate) if \
                self.features_config is not None else None

        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)

        # data augment
        augments = []
        if args.dset.remix:
            augments.append(augment.Remix())
        if args.dset.bandmask:
            augments.append(augment.BandMask(args.dset.bandmask, sample_rate=args.dset.sample_rate))
        if args.dset.shift:
            augments.append(augment.Shift(args.dset.shift, args.dset.shift_same))
        if args.dset.revecho:
            augments.append(
                augment.RevEcho(args.dset.revecho))
        self.augment = torch.nn.Sequential(*augments)

    def get_best_loss_from_history(self):
        if self.history:
            best_loss = self.history[-1]["best"]
            logger.info(f"loaded best loss = {best_loss:.4f} from history")
            return best_loss
        else:
            logger.info("No history was loaded")
            return None

    def _get_features_loss(self, latent_signal, reference_signal):
        if not self.ft_regularization or latent_signal is None:
            return 0
        with torch.no_grad():
            # extract features from the reference signal
            features = self.ft_model.extract_feats(reference_signal)

        # stretch the latent signal to match the extracted features dims
        # -- stretch time dim
        latent_signal = F.interpolate(latent_signal, features.shape[-1], mode='linear').permute(1, 2, 0)
        # -- stretch channel dim
        latent_signal = F.interpolate(latent_signal, features.shape[-2], mode='linear').permute(0, 2, 1)

        # compare the loss
        return F.l1_loss(features, latent_signal) * self.features_config.features_factor

    def run_single_batch(self, batch_data, cross_valid=False):
        """
        this function should do forward propagation over the given batch
        and return the calculated loss
        """
        noisy, clean = [x.to(self.device) for x in batch_data]
        if not cross_valid:
            sources = torch.stack([noisy - clean, clean])
            sources = self.augment(sources)
            noise, clean = sources
            noisy = noise + clean
        cond_features = self.ft_model.extract_feats(noisy) if self.ft_conditioning else None
        if self.ft_regularization:
            estimate, features = self.dmodel(noisy)
        else:
            estimate = self.dmodel(noisy, cond_features)
            features = None

        if clean.shape[-1] > estimate.shape[-1]:
            clean = clean[..., :estimate.shape[-1]]
        if estimate.shape[-1] > clean.shape[-1]:
            estimate = estimate[..., :clean.shape[-1]]

        # apply a loss function after each layer
        with torch.autograd.set_detect_anomaly(True):
            if self.args.loss == 'l1':
                loss = F.l1_loss(clean, estimate)
            elif self.args.loss == 'l2':
                loss = F.mse_loss(clean, estimate)
            elif self.args.loss == 'huber':
                loss = F.smooth_l1_loss(clean, estimate)
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")
            # MultiResolution STFT loss
            if self.args.stft_loss:
                sc_loss, mag_loss = self.mrstftloss(estimate.squeeze(1), clean.squeeze(1))
                loss += sc_loss + mag_loss

            if self.ft_regularization:
                loss += self._get_features_loss(features, clean)

            if self.ft_supervision:
                loss += F.mse_loss(self.ft_model.extract_feats(estimate), self.ft_model.extract_feats(clean)) * \
                        self.features_config.supervision_factor
        return loss

    def evaluate(self, epoch) -> Dict[str, Any]:
        """
        This method performs evaluation over a custom set of metrics
        This should return a dictionary of the form: {'<metric_name>': <metric_mean_value>}
        """
        self.model.eval()
        self.dmodel.eval()
        total_pesq = 0
        total_stoi = 0
        total_snr = 0
        total_cnt = 0
        updates = 5
        files_to_log = []
        pendings = []
        with ProcessPoolExecutor(self.args.num_workers) as pool:
            with torch.no_grad():
                iterator = LogProgress(logger, self.tt_loader, name="Eval estimates")
                for i, data in enumerate(iterator):
                    # Get batch data
                    (noisy, noisy_path), (clean, clean_path) = data
                    filename = os.path.basename(clean_path[0]).rstrip('_clean.wav')
                    noisy = noisy.to(self.args.device)
                    clean = clean.to(self.args.device)
                    if self.args.wandb.n_files_to_log == -1 or len(files_to_log) < self.args.wandb.n_files_to_log:
                        files_to_log.append(filename)
                    # If device is CPU, we do parallel evaluation in each CPU worker.
                    if self.args.device == 'cpu':
                        pendings.append(
                            pool.submit(_estimate_and_run_metrics, clean, self.model, noisy, self.args, filename, self.ft_model,
                                        self.model.features_config.include_ft))
                    else:
                        estimate = get_estimate(self.model, noisy, self.args, self.ft_model, self.args.features_config.include_ft)
                        estimate = estimate.cpu()
                        clean = clean.cpu()
                        pendings.append(
                            pool.submit(_run_metrics, clean, estimate, self.args, self.args.dset.sample_rate, filename))
                    total_cnt += clean.shape[0]

            for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
                tmp = pending.result()
                pesq_i, stoi_i, snr_i, estimate_i, filename_i = tmp
                if filename_i in files_to_log:
                    tmp_metrics = {"pesq": pesq_i, "stoi": stoi_i, "snr": snr_i}
                    self.log_to_wandb(estimate_i, tmp_metrics, filename_i, epoch, self.args.dset.sample_rate)
                total_pesq += pesq_i
                total_stoi += stoi_i
                total_snr += snr_i

        metrics = [total_pesq, total_stoi, total_snr]
        pesq, stoi, snr = distrib.average([m / total_cnt for m in metrics], total_cnt)
        logger.info(bold(f'Test set performance: PESQ={pesq}, STOI={stoi}, SNR={snr}.'))
        return {"pesq": pesq, "stoi": stoi, "snr": snr}

    def enhance(self):
        """
        This method generates enhanced files and saves them to disk
        """
        self.model.eval()
        self.dmodel.eval()
        pendings = []
        with ProcessPoolExecutor(self.args.num_workers) as pool:
            iterator = LogProgress(logger, self.enh_loader, name="Generate enhanced files")
            for data in iterator:
                # Get batch data
                if len(data) == 2:
                    noisy, _ = data
                else:
                    noisy = data
                noisy_signals, filenames = noisy
                noisy_signals = noisy_signals.to(self.args.device)
                if self.args.device == 'cpu' and self.args.num_workers > 1:
                    pendings.append(
                        pool.submit(_estimate_and_save,
                                    self.model, noisy_signals, filenames, self.args.samples_dir, self.args, self.ft_model,
                                    self.model.features_config.include_ft))
                else:
                    # Forward
                    estimate = get_estimate(self.model, noisy_signals, self.args, self.ft_model,
                                            self.model.features_config.include_ft)
                    save_wavs(estimate, noisy_signals, filenames, self.args.samples_dir, sr=self.model.sample_rate)

            if pendings:
                print('Waiting for pending jobs...')
                for pending in LogProgress(logger, pendings, updates=5, name="Generate enhanced files"):
                    pending.result()


# --- external additional functions ---


def get_estimate(model, noisy, arguments, ft_model=None, ft_reg=False):
    torch.set_num_threads(1)
    with torch.no_grad():
        features = ft_model.extract_feats(noisy) if ft_model is not None else None
        estimate = model(noisy, features)[0] if ft_reg else model(noisy, features)
        if noisy.shape[-1] > estimate.shape[-1]:
            noisy = noisy[..., :estimate.shape[-1]]
        elif estimate.shape[-1] > noisy.shape[-1]:
            estimate = estimate[..., :noisy.shape[-1]]
        estimate = (1 - arguments.dry) * estimate + arguments.dry * noisy
    return estimate


def _estimate_and_save(model, noisy_signals, filenames, out_dir, args, ft_model=None, ft_reg=False):
    estimate = get_estimate(model, noisy_signals, args, ft_model, ft_reg)
    save_wavs(estimate, noisy_signals, filenames, out_dir, sr=model.sample_rate)


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def save_wavs(estimates, noisy_sigs, filenames, out_dir, sr=16_000):
    # Write result
    os.makedirs(out_dir, exist_ok=True)
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        write(noisy, filename + "_noisy.wav", sr=sr)
        write(estimate, filename + "_enhanced.wav", sr=sr)


def _run_metrics(clean, estimate, args, sr, filename):
    pesq_i, stoi_i, snr = get_metrics(clean, estimate, sr, True)
    return pesq_i, stoi_i, snr, estimate, filename


def get_metrics(clean, estimate, sr, return_pesq=True):
    if clean.shape[-1] > estimate.shape[-1]:
        clean = clean[..., :estimate.shape[-1]]
    elif estimate.shape[-1] > clean.shape[-1]:
        estimate = estimate[..., :clean.shape[-1]]

    estimate_numpy = estimate.squeeze(1).numpy()
    clean_numpy = clean.squeeze(1).numpy()
    pesq = get_pesq(clean_numpy, estimate_numpy, sr=sr) if return_pesq else 0
    stoi = get_stoi(clean_numpy, estimate_numpy, sr=sr)
    snr = get_snr(estimate, estimate - clean).item()
    return pesq, stoi, snr


def _estimate_and_run_metrics(clean, model, noisy, args, filename, ft_model=None, ft_reg=False):
    estimate = get_estimate(model, noisy, args, ft_model, ft_reg)
    return _run_metrics(clean, estimate, args, sr=model.sample_rate, filename=filename)
