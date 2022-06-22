# Implementation was based on code from: https://github.com/facebookresearch/denoiser with the following license:
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-MIT.txt file in the root directory of this source tree.

import json
import logging
from abc import abstractmethod, ABC
from pathlib import Path
import os
import time
from typing import Any, Dict

import torch
import wandb
from torchaudio.transforms import Spectrogram

from external_files import distrib
from external_files.utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from utils.visualization import convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)


class BaseSolver(object):

    """
    Base Solver class.
    Each new custom solver should extend this class.
    """

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.enh_loader = data['enh_loader']
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer

        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.include_pretraining = args.include_pretraining
        self.args = args
        self._reset()

    def _serialize(self):
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        model = package['model']
        model['state'] = self.best_state
        tmp_path = str(self.best_file) + ".tmp"
        torch.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

    def log_loss(self, loss, train=True):
        return f'{"Train" if train else "Valid"} Loss {loss:.5f}'

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = False

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['model']['state'])
            if 'optimizer' in package and not load_best:
                self.optimizer.load_state_dict(package['optimizer'])
            if keep_history:
                self.history = package['history']
            self.best_state = package['best_state']

    def capture_loss_for_metrics(self, train_loss, valid_loss, best_loss):
        return {'train': train_loss, 'valid': valid_loss, 'best': best_loss}

    def check_if_model_improved_and_save_best(self, valid_loss, best_loss):
        flag = best_loss is None or valid_loss < best_loss
        if flag:
            logger.info(bold('New best valid loss %.4f'), valid_loss)
            self.best_state = copy_state(self.model.state_dict())
            best_loss = valid_loss
        return flag, best_loss

    def eval_over_test_set(self, epoch, model_improved):
        if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
            # Evaluate on the testset
            logger.info('-' * 70)
            logger.info('Evaluating on the test set...')
            # We switch to the best known model for testing
            with swap_state(self.model, self.best_state):
                evaluated_metrics = self.evaluate(epoch)
                # enhance some samples
                if model_improved and (not hasattr(self.args, "do_enhance") or
                                       hasattr(self.args, "do_enhance") and self.args.do_enhance):
                    logger.info('Enhance and save samples...')
                    self.enhance()
                else:
                    logger.info("Best validation loss hasn't change, skipping enhance.")
            return evaluated_metrics
        return None

    def train(self):
        if self.args.save_again:
            self._serialize()
            return
        # reload model if weights are found in ./outputs/... dir
        if self.history:
            logger.info("Replaying metrics from previous run")
        elif self.include_pretraining:
            self.run_pretraining()
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        if (self.epochs > len(self.history)):
            logger.info("Training...")

        best_loss = self.get_best_loss_from_history()
        model_improved = False
        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            logger.info(f"Epoch: {epoch + 1}")
            self.model.train()
            start = time.time()
            train_loss = self.run_single_epoch(epoch)
            end = time.time()
            logger.info("------")
            logger.info(
                bold(f'Train Summary | End of Epoch {epoch + 1} | '
                     f'Time {end - start:.2f}s | {self.log_loss(train_loss)}'))

            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    valid_loss = self.run_single_epoch(epoch, cross_valid=True)
                val_end = time.time()
                logger.info("------")
                logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                         f'Time {val_end - end:.2f}s | {self.log_loss(valid_loss, False)}'))
            else:
                valid_loss = 0

            # Save the best model
            model_has_improved, best_loss = self.check_if_model_improved_and_save_best(valid_loss, best_loss)
            if model_has_improved:
                model_improved = True

            metrics = self.capture_loss_for_metrics(train_loss, valid_loss, best_loss)

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            evaluated_metrics = self.eval_over_test_set(epoch, model_improved)
            if evaluated_metrics is not None:
                metrics.update(evaluated_metrics)
                model_improved = False
            # update wandb and history files
            wandb.log(metrics, step=epoch)
            self.history.append(metrics)
            # log to stdout
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
        if self.args.log_results:
            self.log_results()

# ========================= Commonly used methods to override =========================

    def log_information_at_beggining_of_training(self):
        """
        Method to log information pre-training
        """
        logger.info("Trainable Params:")
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        mb = n_params * 4 / 2 ** 20
        logger.info(f"{self.args.model}: parameters: {n_params}, size: {mb} MB")

    def optimize(self, loss):
        """
        Method that handles all back-propagation.
        The following is a simple suggestion
        """
        with torch.autograd.set_detect_anomaly(True):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def accumulate_loss(self, total_loss, loss):
        if total_loss is None:
            return loss.item()
        return total_loss + loss.item()

    def log_prog(self, logprog, total_loss, i):
        logprog.update(loss=format(total_loss / (i + 1), ".5f"))

    def retval_for_run_single_epoch(self, total_loss, i):
        return distrib.average([total_loss / (i + 1)], i + 1)[0]

    def run_single_epoch(self, epoch, cross_valid=False):
        """
        This function is the main function per epoch that should include:
        1. forward propagation
        2. calculate loss
        3. back propagate gradients

        Then following is a simple suggestion, when overriding try to modify this method to make things simple
        """
        total_loss = None
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        self.model.train()
        self.dmodel.train()
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):

            loss = self.run_single_batch(data, cross_valid)
            if not cross_valid:
                self.optimize(loss)

            total_loss = self.accumulate_loss(total_loss, loss)
            self.log_prog(logprog, total_loss, i)
        return self.retval_for_run_single_epoch(total_loss, i)

    def log_results(self):
        """
        (optional) This method logs results at the end of the train process.
        does nothing by default
        """
        return

    @staticmethod
    def log_to_wandb(signal, metrics, filename, epoch, sr):
        """
        optional implementation, this function should be used in each enhance step
        """
        spectrogram_transform = Spectrogram()
        enhanced_spectrogram = spectrogram_transform(signal).log2()[0, :, :].numpy()
        enhanced_spectrogram_wandb_image = wandb.Image(convert_spectrogram_to_heatmap(enhanced_spectrogram),
                                                       caption=filename)
        enhanced_wandb_audio = wandb.Audio(signal.squeeze().numpy(), sample_rate=sr, caption=filename)
        log_dict = {f'test samples/{filename}/{metric}': metric_value for metric, metric_value in metrics.items()}
        log_dict.update({f'test samples/{filename}/spectrogram': enhanced_spectrogram_wandb_image,
                         f'test samples/{filename}/audio': enhanced_wandb_audio})
        wandb.log(log_dict, step=epoch)

# TODO: ========================= Abstract methods to override =========================

    def run_pretraining(self):  # optional function
        pass

    @abstractmethod
    def run_single_batch(self, batch_data, cross_valid=False):
        """
        this function should do forward propagation over the given batch
        and return the calculated loss
        :param data:
        :param cross_valid:
        :return:
        """
        raise NotImplementedError("Implement this function. See docstring")

    @abstractmethod
    def evaluate(self, epoch) -> Dict[str, Any]:
        """
        This method performs evaluation over a custom set of metrics
        This should return a dictionary of the form: {'<metric_name>': <metric_mean_value>}
        """
        raise NotImplementedError()

    @abstractmethod
    def enhance(self):
        """
        This method generates enhanced files and saves them to disk
        """
        # TODO: use self.log_to_wandb(signal, metrics, filename, epoch, sr) to log tested files to wandb in each eval step.
        raise NotImplementedError()

    @abstractmethod
    def get_best_loss_from_history(self):
        pass

