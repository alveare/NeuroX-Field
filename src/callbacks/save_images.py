import lightning.pytorch as pl
from lightning.pytorch import Callback

import torch
import os.path
import numpy
import time
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor

from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info


class SaveImagesHook(Callback):
    def __init__(self, save_dir="val", save_compressed=True, num_channels_to_plot=None, max_save_samples=None):
        self.save_dir = save_dir
        self.save_compressed = save_compressed
        self.num_channels_to_plot = num_channels_to_plot
        self.max_save_samples = max_save_samples

    _NPZ_TENSOR_KEYS = ['gt_eeg', 'observed_mask', 'label']

    _FLUSH_EVERY = 8000  # flush to disk every N samples to avoid OOM

    def save_start(self, target_dir):
        self.samples = []
        self._sample_names = []   # per-sample string keys
        self._meta_tensors = {}   # key -> list of numpy arrays (per-sample)
        self._meta_const = None   # constant metadata (saved once from first batch)
        self._chunk_idx = 0       # chunk counter for flushed files
        self._mem_collected = 0   # samples held in memory (reset on flush)
        self.target_dir = target_dir
        self.executor_pool = ThreadPoolExecutor(max_workers=8)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir, exist_ok=True)
        else:
            if os.listdir(target_dir) and "debug" not in str(target_dir):
                raise FileExistsError(f'{self.target_dir} already exists and not empty!')
        rank_zero_info(f"Save images to {self.target_dir}")
        self._saved_num = 0
        self._npz_collected = 0

    def process_batch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: STEP_OUTPUT,
        batch: Any,
    ) -> None:
        xT, y, metadata = batch
        b = samples.shape[0]
        if not self.save_compressed or self._saved_num < 10:
            self._saved_num += b
        
        # Save local samples + metadata on rank 0 only
        if not trainer.is_global_zero:
            return
        
        if samples.ndim == 4:  # reconstructed signal: (B, N, C, T)
            local_samples = samples.permute(0, 2, 3, 1).cpu().numpy()  # (B, N, C, T) -> (B, C, T, N)
        else:                  # embedding: (B, N, D)
            local_samples = samples.cpu().numpy()

        if self.max_save_samples is not None:
            remaining = self.max_save_samples - self._npz_collected
            if remaining <= 0:
                return
            local_samples = local_samples[:remaining]
            metadata = metadata[:remaining]
            self._npz_collected += local_samples.shape[0]

        self.samples.append(local_samples)
        self._sample_names.extend(m.get('sample_name', '') for m in metadata)
        self._mem_collected += local_samples.shape[0]

        # save tensor metadata (per-sample)
        for key in self._NPZ_TENSOR_KEYS:
            vals = [m[key].cpu().numpy() for m in metadata if key in m]
            if vals:
                self._meta_tensors.setdefault(key, []).append(numpy.stack(vals))

        # save normalization statistics for denormalization
        norm_mean_batch = []
        norm_std_batch = []
        for m in metadata:
            np_params = m.get('norm_params')
            if not isinstance(np_params, dict):
                continue
            nm = np_params.get('mean', None)
            ns = np_params.get('std', None)
            if nm is None or ns is None:
                continue
            nm_np = nm.cpu().numpy() if isinstance(nm, torch.Tensor) else numpy.asarray(nm, dtype=numpy.float64)
            ns_np = ns.cpu().numpy() if isinstance(ns, torch.Tensor) else numpy.asarray(ns, dtype=numpy.float64)
            norm_mean_batch.append(nm_np)
            norm_std_batch.append(ns_np)
        if len(norm_mean_batch) == len(metadata) and norm_mean_batch:
            self._meta_tensors.setdefault('normalization_mean', []).append(numpy.stack(norm_mean_batch))
            self._meta_tensors.setdefault('normalization_std', []).append(numpy.stack(norm_std_batch))

        # save constant metadata (only once)
        if self._meta_const is None:
            m0 = metadata[0]
            self._meta_const = {}
            if 'ch_names' in m0:
                self._meta_const['ch_names'] = numpy.array(m0['ch_names'], dtype=object)
            if 'coords' in m0 and isinstance(m0['coords'], torch.Tensor):
                self._meta_const['coords'] = m0['coords'].cpu().numpy()

        # Flush to disk periodically to avoid OOM
        if self._mem_collected >= self._FLUSH_EVERY:
            self._flush_chunk()

    def _flush_chunk(self):
        """Write accumulated samples to a chunk file and free memory."""
        if not self.samples:
            return
        save_dict = {'samples': numpy.concatenate(self.samples)}
        for k, v_list in self._meta_tensors.items():
            save_dict[k] = numpy.concatenate(v_list)
        if self._sample_names:
            save_dict['sample_names'] = numpy.array(self._sample_names, dtype=object)
        if self._meta_const:
            save_dict.update(self._meta_const)
        numpy.savez(f'{self.target_dir}/output_{self._chunk_idx:04d}.npz', **save_dict)
        rank_zero_info(f"Flushed chunk {self._chunk_idx} ({save_dict['samples'].shape[0]} samples)")
        self._chunk_idx += 1
        self.samples.clear()
        self._sample_names.clear()
        self._meta_tensors.clear()
        self._mem_collected = 0

    def save_end(self):
        if self.save_compressed:
            # Flush remaining samples
            self._flush_chunk()
            if self._chunk_idx == 1:
                src = f'{self.target_dir}/output_0000.npz'
                dst = f'{self.target_dir}/output.npz'
                if os.path.exists(src):
                    os.rename(src, dst)
        self.executor_pool.shutdown(wait=True)
        self.target_dir = None
        self.executor_pool = None
        self.samples = []
        self._sample_names = []
        self._meta_tensors = {}
        self._meta_const = None

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, f"iter_{trainer.global_step}")
        self.save_start(target_dir)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, outputs, batch)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        n_save = "all" if self.max_save_samples is None else str(self.max_save_samples)
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, "predict", f"test_{n_save}_{timestamp}")
        self.save_start(target_dir)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, samples, batch)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end()

    def state_dict(self) -> Dict[str, Any]:
        return dict()