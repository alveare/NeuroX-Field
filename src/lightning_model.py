from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os
import os.path
import json
import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback

from src.models.conditioner.base import BaseConditioner
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


def _denorm_from_metadata(
    x_b1ct: torch.Tensor, metadata,
) -> Optional[torch.Tensor]:
    """Inverse of the dataset normalization: recover raw-uV via x * std + mean."""
    if not (isinstance(metadata, (list, tuple)) and len(metadata) > 0
            and isinstance(metadata[0], dict) and 'norm_params' in metadata[0]):
        return None
    B, _, C, _T = x_b1ct.shape
    means, stds = [], []
    for m in metadata:
        p = m.get('norm_params') if isinstance(m, dict) else None
        if not isinstance(p, dict) or p.get('std', None) is None:
            return None
        means.append(torch.as_tensor(p.get('mean', 0.0), dtype=x_b1ct.dtype, device=x_b1ct.device))
        stds.append(torch.as_tensor(p.get('std'), dtype=x_b1ct.dtype, device=x_b1ct.device))

    def _stack_b1ct(vals):
        v = torch.stack([t.reshape(-1) for t in vals], dim=0)
        if v.shape[1] == 1:
            return v.view(B, 1, 1, 1)
        if v.shape[1] == C:
            return v.view(B, 1, C, 1)
        return None

    mean_t = _stack_b1ct(means)
    std_t = _stack_b1ct(stds)
    if mean_t is None or std_t is None:
        return None
    return x_b1ct * std_t + mean_t


class LightningModel(pl.LightningModule):
    def __init__(self,
                 conditioner: BaseConditioner,
                 denoiser: nn.Module,
                 diffusion_trainer: BaseTrainer,
                 diffusion_sampler: BaseSampler,
                 ema_tracker: SimpleEMA=None,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 eval_original_model: bool = False,
                 debug: bool = False,
                 num_val_ensemble: int = 1,
                 clf_ckpt_path: Optional[str] = None,
                 clf_input_norm: str = "none",
                 clf_input_scale: float = 100.0,
                 ):
        super().__init__()
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model
        self.debug = debug
        self.num_val_ensemble = num_val_ensemble

        self.clf_ckpt_path = clf_ckpt_path
        self.clf_input_norm = clf_input_norm
        self.clf_input_scale = clf_input_scale
        self._clf = None
        self._clf_cfg = None
        self._clf_unavailable = False

        self._strict_loading = False

    def _get_classifier(self):
        """
        Load the downstream EEGNet classifier.
        """
        if self._clf is not None:
            return self._clf, self._clf_cfg
        if self._clf_unavailable:
            return None, None
        if not self.clf_ckpt_path:
            self._clf_unavailable = True
            return None, None
        if not os.path.isfile(self.clf_ckpt_path):
            print(f"[clf] checkpoint not found: {self.clf_ckpt_path} — "
                  f"skipping downstream accuracy metrics.")
            self._clf_unavailable = True
            return None, None
        from src.eval.eegnet import load_encoder
        try:
            model, cfg, ckpt = load_encoder(self.clf_ckpt_path, self.device)
        except Exception as e:
            print(f"[clf] failed to load {self.clf_ckpt_path} "
                  f"({type(e).__name__}: {e}) — skipping downstream accuracy metrics.")
            self._clf_unavailable = True
            return None, None
        model.eval()
        self._clf, self._clf_cfg = model, cfg
        print(f"[clf] loaded EEGNet ckpt {self.clf_ckpt_path} "
              f"(nb_classes={cfg['nb_classes']} Chans={cfg['Chans']} "
              f"Samples={cfg['Samples']} "
              f"trained val_acc={ckpt.get('val_accuracy', float('nan')):.4f})")
        return self._clf, self._clf_cfg

    @torch.no_grad()
    def _clf_accuracy(self, x_b1ct, labels, model, cfg, metadata):
        """
        Classification accuracy.
        x_b1ct: (B, 1, C, T) in the model's normalized output space.
        labels: (B,) long tensor
        """
        x = x_b1ct.float()
        if self.clf_input_norm == "scale":
            x = x * self.clf_input_scale
        elif self.clf_input_norm == "uv":
            x_uv = _denorm_from_metadata(x, metadata)
            if x_uv is None:
                return None
            x = x_uv
        elif self.clf_input_norm == "zscore":
            m = x.mean(dim=-1, keepdim=True)
            s = x.std(dim=-1, keepdim=True)
            s = torch.where(s == 0, torch.ones_like(s), s)
            x = (x - m) / s
        if x.shape[2] != cfg["Chans"] or x.shape[3] != cfg["Samples"]:
            return None
        preds = model(x.to(self.device)).argmax(1)
        return (preds == labels.to(preds.device)).float().mean().item()

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        no_grad(self.conditioner)
        # no_grad(self.diffusion_sampler)
        no_grad(self.ema_denoiser)

        # torch.compile
        if not self.debug:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 64
            )
            self.denoiser.compile()
            self.ema_denoiser.compile()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        params_sampler = filter_nograd_tensors(self.diffusion_sampler.parameters())
        param_groups = [
            {"params": params_denoiser, },
            {"params": params_trainer,},
            {"params": params_sampler, "lr": 1e-3},
        ]
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.predict_metrics = {}

    def on_predict_epoch_end(self) -> None:
        if not (hasattr(self, 'predict_metrics') and self.predict_metrics):
            return
        summary: Dict[str, Dict[str, float]] = {}
        for k, vals in self.predict_metrics.items():
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            summary[k] = {'mean': float(mean), 'std': float(var ** 0.5), 'n': n}

        print("\n" + "=" * 40)
        print("Predict Epoch End Metrics:")
        for k, s in summary.items():
            print(f"{k}: {s['mean']:.6f} ± {s['std']:.6f}  (n={s['n']})")
        print("=" * 40 + "\n")

        target_dir = os.path.join(self.trainer.default_root_dir, "val", "predict")

        os.makedirs(target_dir, exist_ok=True)
        json_path = os.path.join(target_dir, 'predict_metrics.json')
        with open(json_path, 'w') as f:
            json.dump({'metrics': summary}, f, indent=2)
        print(f"Saved predict metrics to {json_path}")

    # sanity check before training start
    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y, metadata)
        
        loss = self.diffusion_trainer(self.denoiser, self.ema_denoiser, self.diffusion_sampler, x, condition, uncondition, metadata)
        self.log_dict(loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False)

        return loss["loss"]

    def predict_step(self, batch, batch_idx, num_ensemble=None):
        is_predict = self.trainer.predicting
        print("Stage: ", "predict" if is_predict else "validation")
        xT, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        if self.eval_original_model:
            model = self.denoiser
        else:
            model = self.ema_denoiser

        if num_ensemble is None:
            num_ensemble = self.num_val_ensemble
            
        if num_ensemble > 1:
            ensemble_samples = []
            for i in range(num_ensemble):
                current_noise = torch.randn_like(xT)
                sample = self.diffusion_sampler(model, current_noise, condition, uncondition, metadata)
                ensemble_samples.append(sample)
            
            # Average samples
            ensemble_stack = torch.stack(ensemble_samples)
            samples = torch.mean(ensemble_stack, dim=0)
            
            # Uncertainty Quantification
            uncertainty = torch.std(ensemble_stack, dim=0)
            
            if isinstance(metadata, (list, tuple)):
                for i, m in enumerate(metadata):
                    if isinstance(m, dict):
                        m['uncertainty'] = uncertainty[i]
        else:
            samples = self.diffusion_sampler(model, xT, condition, uncondition, metadata)

        gt_eeg = None
        if isinstance(metadata, (list, tuple)) and len(metadata) > 0:
            if isinstance(metadata[0], dict) and 'gt_eeg' in metadata[0]:
                gt_list = [m['gt_eeg'] for m in metadata]
                gt_eeg = torch.stack(gt_list, dim=0).to(samples.device)
        elif isinstance(metadata, dict) and 'gt_eeg' in metadata:
            gt_eeg = metadata['gt_eeg'].to(samples.device)

        # Class labels for the downstream classifier
        labels = None
        if isinstance(metadata, (list, tuple)) and len(metadata) > 0:
            if isinstance(metadata[0], dict) and 'label' in metadata[0]:
                labels = torch.stack([torch.as_tensor(m['label']) for m in metadata]
                                     ).reshape(-1).long().to(samples.device)
        elif isinstance(metadata, dict) and 'label' in metadata:
            labels = torch.as_tensor(metadata['label']).reshape(-1).long().to(samples.device)

        # Calculate metrics if ground truth is available
        if gt_eeg is not None:
            if samples.shape[2] > gt_eeg.shape[2]:
                n_gt_ch = gt_eeg.shape[2]
                samples_for_metric = samples[:, :, :n_gt_ch, :]
            else:
                samples_for_metric = samples
            
            mse_all = torch.nn.functional.mse_loss(samples_for_metric, gt_eeg)
            print(f"Validation MSE: {mse_all.item()}")
            if not is_predict:
                self.log('val/mse', mse_all.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.predict_metrics.setdefault('val/mse', []).append(mse_all.item())

            # Extract observed_mask for masked-only metrics
            observed_mask = None
            if isinstance(metadata, (list, tuple)) and len(metadata) > 0:
                if isinstance(metadata[0], dict) and 'observed_mask' in metadata[0]:
                    obs_list = [m['observed_mask'] for m in metadata]
                    observed_mask = torch.stack(obs_list, dim=0).to(samples.device)
            elif isinstance(metadata, dict) and 'observed_mask' in metadata:
                observed_mask = metadata['observed_mask'].to(samples.device)
            # Build sliced tensors for masked (unobserved) channels
            if observed_mask is not None:
                masked_ch_idx = (observed_mask[0, 0] == 0).nonzero(as_tuple=True)[0]  
                pred_masked = samples_for_metric[:, :, masked_ch_idx, :] 
                gt_masked = gt_eeg[:, :, masked_ch_idx, :]               

            # NMSE (all)
            B = samples_for_metric.shape[0]
            per_sample_mse = (samples_for_metric - gt_eeg).pow(2).flatten(1).mean(dim=1) 
            per_sample_var = gt_eeg.flatten(1).var(dim=1)  
            nmse_ps = (per_sample_mse / (per_sample_var + 1e-12)).mean()
            print(f"Validation NMSE (all): {nmse_ps.item()}")
            if not is_predict:
                self.log('val/nmse', nmse_ps.detach(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.predict_metrics.setdefault('val/nmse', []).append(nmse_ps.item())

            # NMSE (masked channels only)
            if observed_mask is not None:
                m_mse_ps = (pred_masked - gt_masked).pow(2).flatten(1).mean(dim=1) 
                m_var_ps = gt_masked.flatten(1).var(dim=1)  
                nmse_masked = (m_mse_ps / (m_var_ps + 1e-12)).mean()
                print(f"Validation NMSE (masked): {nmse_masked.item()}")
                if not is_predict:
                    self.log('val/nmse_masked', nmse_masked.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                else:
                    self.predict_metrics.setdefault('val/nmse_masked', []).append(nmse_masked.item())


            # PCC (all)
            pred_ch = samples_for_metric.squeeze(1).float() 
            gt_ch = gt_eeg.squeeze(1).float()
            pred_centered = pred_ch - pred_ch.mean(dim=-1, keepdim=True)
            gt_centered = gt_ch - gt_ch.mean(dim=-1, keepdim=True)
            numer = (pred_centered * gt_centered).sum(dim=-1)
            denom = torch.sqrt(pred_centered.pow(2).sum(dim=-1) * gt_centered.pow(2).sum(dim=-1)) + 1e-12
            pcc_ps = (numer / denom).mean()
            print(f"Validation PCC (all): {pcc_ps.item()}")
            if not is_predict:
                self.log('val/pcc', pcc_ps.detach(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.predict_metrics.setdefault('val/pcc', []).append(pcc_ps.item())

            # PCC (masked channels only)
            if observed_mask is not None:
                pred_m_ch = pred_masked.squeeze(1).float()  # (B, C_masked, T)
                gt_m_ch = gt_masked.squeeze(1).float()
                pred_m_centered = pred_m_ch - pred_m_ch.mean(dim=-1, keepdim=True)
                gt_m_centered = gt_m_ch - gt_m_ch.mean(dim=-1, keepdim=True)
                numer_m = (pred_m_centered * gt_m_centered).sum(dim=-1)  # (B, C_masked)
                denom_m = torch.sqrt(pred_m_centered.pow(2).sum(dim=-1) * gt_m_centered.pow(2).sum(dim=-1)) + 1e-12
                pcc_masked = (numer_m / denom_m).mean()
                print(f"Validation PCC (masked): {pcc_masked.item()}")
                if not is_predict:
                    self.log('val/pcc_masked', pcc_masked.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                else:
                    self.predict_metrics.setdefault('val/pcc_masked', []).append(pcc_masked.item())

            # SNR (all)
            noise = samples_for_metric - gt_eeg
            per_sample_signal = gt_eeg.flatten(1).pow(2).mean(dim=1)  # (B,)
            per_sample_noise = noise.flatten(1).pow(2).mean(dim=1)    # (B,)
            snr_ps = (10.0 * torch.log10(per_sample_signal / (per_sample_noise + 1e-12))).mean()
            print(f"Validation SNR (all): {snr_ps.item()}")
            if not is_predict:
                self.log('val/snr_db', snr_ps.detach(), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.predict_metrics.setdefault('val/snr_db', []).append(snr_ps.item())

            # SNR (masked channels only)
            if observed_mask is not None:
                m_noise = pred_masked - gt_masked
                m_signal_power = gt_masked.flatten(1).pow(2).mean(dim=1)
                m_noise_power = m_noise.flatten(1).pow(2).mean(dim=1)
                snr_masked = (10.0 * torch.log10(m_signal_power / (m_noise_power + 1e-12))).mean()
                print(f"Validation SNR (masked): {snr_masked.item()}")
                if not is_predict:
                    self.log('val/snr_db_masked', snr_masked.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                else:
                    self.predict_metrics.setdefault('val/snr_db_masked', []).append(snr_masked.item())

            # Downstream classification accuracy
            clf_model, clf_cfg = self._get_classifier()
            if clf_model is not None and labels is not None:
                y = labels
                nb = clf_cfg["nb_classes"]
                if int(y.min()) == 1 and int(y.max()) == nb:
                    y = y - 1
                acc_gen = self._clf_accuracy(samples_for_metric, y, clf_model, clf_cfg, metadata)
                acc_gt = self._clf_accuracy(gt_eeg, y, clf_model, clf_cfg, metadata)
                for key, acc in (('val/clf_acc_gen', acc_gen), ('val/clf_acc_gt', acc_gt)):
                    if acc is None:
                        continue
                    print(f"Validation {key}: {acc}")
                    if not is_predict:
                        self.log(key, torch.tensor(acc, device=samples.device),
                                 prog_bar=(key == 'val/clf_acc_gen'),
                                 on_step=False, on_epoch=True, sync_dist=True)
                    else:
                        self.predict_metrics.setdefault(key, []).append(acc)

        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix+"denoiser.",
            keep_vars=keep_vars)
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix+"ema_denoiser.",
            keep_vars=keep_vars)
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix+"diffusion_trainer.",
            keep_vars=keep_vars)
        return destination