from typing import Any
import torch
import copy
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, IterableDataset


class PerBatchRatioBatchSampler(torch.utils.data.Sampler):
    """
    Groups indices from a base sampler into batches and attaches a single spatial 
    mask ratio (drawn uniformly from ``ratio_range``) to every index in each batch.
    """

    def __init__(self, sampler, batch_size, ratio_range, drop_last=False, seed=42):
        self.sampler = sampler
        self.batch_size = batch_size
        self.ratio_range = ratio_range
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        if callable(getattr(self.sampler, "set_epoch", None)):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self._epoch)
        lo, hi = self.ratio_range
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                r = rng.uniform(lo, hi)
                yield [(i, r) for i in batch]
                batch = []
        if batch and not self.drop_last:
            r = rng.uniform(lo, hi)
            yield [(i, r) for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def mirco_batch_collate_fn(batch):
    batch = copy.deepcopy(batch)
    new_batch = []
    for micro_batch in batch:
        new_batch.extend(micro_batch)
    x, y, metadata = list(zip(*new_batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except:
            pass
    x = torch.stack(x, dim=0)
    
    if isinstance(y[0], torch.Tensor):
        y = torch.stack(y, dim=0)

    return x, y, stacked_metadata

def collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except:
            pass
    x = torch.stack(x, dim=0)
    
    if isinstance(y[0], torch.Tensor):
        y = torch.stack(y, dim=0)

    return x, y, stacked_metadata

def eval_collate_fn(batch):
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    x = torch.stack(x, dim=0)
    
    if isinstance(y[0], torch.Tensor):
        y = torch.stack(y, dim=0)

    return x, y, metadata


class DataModule(pl.LightningDataModule):
    def __init__(self,
                train_dataset:Dataset=None,
                eval_dataset:Dataset=None,
                pred_dataset:Dataset=None,
                train_batch_size=64,
                train_num_workers=16,
                train_prefetch_factor=8,
                eval_batch_size=32,
                eval_num_workers=4,
                pred_batch_size=32,
                pred_num_workers=4,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.pred_dataset = pred_dataset
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor

        self.eval_batch_size = eval_batch_size
        self.pred_batch_size = pred_batch_size

        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers

        self._train_dataloader = None

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        micro_batch_size = getattr(self.train_dataset, "micro_batch_size", None)
        if micro_batch_size is not None:
            assert self.train_batch_size % micro_batch_size == 0
            dataloader_batch_size = self.train_batch_size // micro_batch_size
            train_collate_fn = mirco_batch_collate_fn
        else:
            dataloader_batch_size = self.train_batch_size
            train_collate_fn = collate_fn
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size

        # build dataloader sampler
        if not isinstance(self.train_dataset, IterableDataset):
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=world_size, rank=global_rank)
        else:
            sampler = None
        
        # dynamic masking ratio
        ratio_range = getattr(self.train_dataset, "spatial_mask_ratio_range", None)
        batch_sampler = None
        if ratio_range is not None and micro_batch_size is None:
            base_sampler = sampler if sampler is not None else torch.utils.data.RandomSampler(self.train_dataset)
            batch_sampler = PerBatchRatioBatchSampler(
                base_sampler, dataloader_batch_size, ratio_range, drop_last=False
            )

        prefetch_factor = self.train_prefetch_factor if self.train_num_workers > 0 else None
        timeout = 6000 if self.train_num_workers > 0 else 0
        if batch_sampler is not None:
            self._train_dataloader = DataLoader(
                self.train_dataset,
                timeout=timeout,
                num_workers=self.train_num_workers,
                persistent_workers=True if self.train_num_workers > 0 else False,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                collate_fn=train_collate_fn,
                batch_sampler=batch_sampler,
            )
        else:
            self._train_dataloader = DataLoader(
                self.train_dataset,
                dataloader_batch_size,
                timeout=timeout,
                num_workers=self.train_num_workers,
                persistent_workers=True if self.train_num_workers > 0 else False,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                collate_fn=train_collate_fn,
                sampler=sampler,
            )
        
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.eval_dataset is None:
            return []
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        eval_prefetch = 2 if self.eval_num_workers > 0 else None

        return DataLoader(self.eval_dataset, self.eval_batch_size,
                          num_workers=self.eval_num_workers,
                          persistent_workers=True if self.eval_num_workers > 0 else False,
                          pin_memory=True, 
                          prefetch_factor=eval_prefetch,
                          sampler=sampler,
                          collate_fn=eval_collate_fn
                )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.pred_dataset is None:
            return []
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        pred_prefetch = 4 if self.pred_num_workers > 0 else None

        return DataLoader(self.pred_dataset, batch_size=self.pred_batch_size,
                          num_workers=self.pred_num_workers,
                          persistent_workers=True if self.pred_num_workers > 0 else False,
                          prefetch_factor=pred_prefetch,
                          sampler=sampler,
                          collate_fn=eval_collate_fn
               )
