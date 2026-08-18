import os
import lmdb
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import mne
import random
import numpy as np
import pandas as pd
from scipy.signal import resample_poly


class EEGDataset(Dataset):
    """Dataset for EEG Spatial Super-Resolution using Continuous 3D Coordinates."""
    def __init__(
        self,
        data_root,
        dataset='seed',
        eeg_channels=62,
        eeg_timepoints=800,
        limit_samples=None,
        downsample_factor=1.0,
        spatial_mask_mode='2x',
        spatial_mask_layout=0,
        spatial_mask_ratio=0.0,
        normalization_mode='scale',
        scale_const=0.01,
        normalize_coords=False,
        visible_only=False,
        return_orig_gt=False,
        split=None,
        seed=42,
    ):
        self.data_root = data_root
        self.dataset = dataset.lower()
        if self.dataset not in ('seed', 'physionet'):
            raise ValueError(f"Unsupported dataset: '{dataset}'. Expected 'seed' or 'physionet'.")
        self.eeg_channels = eeg_channels
        self.eeg_timepoints = eeg_timepoints
        self.downsample_factor = downsample_factor
        self.spatial_mask_mode = spatial_mask_mode
        self.spatial_mask_layout = spatial_mask_layout
        if isinstance(spatial_mask_ratio, (tuple, list)):
            if len(spatial_mask_ratio) != 2:
                raise ValueError(f"spatial_mask_ratio tuple must have 2 elements, got {len(spatial_mask_ratio)}")
            lo, hi = float(spatial_mask_ratio[0]), float(spatial_mask_ratio[1])
            if lo > hi:
                raise ValueError(f"spatial_mask_ratio range invalid: ({lo}, {hi})")
            self.spatial_mask_ratio_range = (lo, hi)
            self.spatial_mask_ratio = hi
            print(f"spatial_mask_ratio: random in range [{lo}, {hi}] per sample")
        else:
            self.spatial_mask_ratio_range = None
            self.spatial_mask_ratio = float(spatial_mask_ratio)
        self.visible_only = visible_only
        self.normalization_mode = normalization_mode
        self.scale_const = float(scale_const)
        self.normalize_coords = normalize_coords
        self.return_orig_gt = return_orig_gt
        self.split = split
        self.seed = seed

        # Open LMDB
        self._db = lmdb.open(self.data_root, readonly=True, lock=False, readahead=True, meminit=False)
        with self._db.begin(write=False) as txn:
            self.keys_dict = pickle.loads(txn.get('__keys__'.encode()))
        self._db.close()
        self._db = None

        # Split dataset
        if split is not None:
            if split not in self.keys_dict:
                raise ValueError(f"Split '{split}' not found in keys. Available splits: {list(self.keys_dict.keys())}")
            
            self.keys = self.keys_dict[split]
            split_name = split.upper()
            
            # Limit samples (applied after selecting split)
            if limit_samples is not None and limit_samples > 0:
                self.keys = self.keys[:limit_samples]
                print(f"[{split_name}] Limiting to first {limit_samples} keys.")

            n_total = sum(len(v) for v in self.keys_dict.values())
            n_current = len(self.keys)
            print(f"[{split_name}] {n_current} keys (total in DB: {n_total}).")
        else:
            raise ValueError(f"Split must be provided.")
        
        self.samples = []
        for key in self.keys:
            self.samples.append({'key': key})
        print(f"Prepared {len(self.samples)} samples.")
        
        self._prepare_montage()
        self._prepare_spatial_mask_layout()
        self._precompute_coords()
        self._preload_metadata()
    
    def _prepare_montage(self):
        """
        Load standard 10-05 montage and prepare coordinate mapping.
        """
        montage = mne.channels.make_standard_montage('standard_1005')
        self.pos_dict = montage.get_positions()['ch_pos']
        self.pos_dict_lower = {k.lower(): v for k, v in self.pos_dict.items()}
        aliases = {'t3': 't7', 't4': 't8', 't5': 'p7', 't6': 'p8'}
        for old, new in aliases.items():
            if new in self.pos_dict_lower:
                self.pos_dict_lower[old] = self.pos_dict_lower[new]

    def _prepare_spatial_mask_layout(self):
        """
        Dispatch to dataset-specific layout preparation.
        """
        if self.dataset == 'seed':
            self._prepare_spatial_mask_layout_seed()
        elif self.dataset == 'physionet':
            self._prepare_spatial_mask_layout_physionet()
        else:
            raise ValueError(f"Unsupported dataset: '{self.dataset}'. Expected 'seed' or 'physionet'.")

        print(f"Layout {self.spatial_mask_layout} number of visible channels: {len(self.visible_channel_idx)}")
        print(f"Layout {self.spatial_mask_layout} visible channels: {self.visible_channels}")
        if self.visible_only:
            print(f"Training on visible channels only ({len(self.visible_channel_idx)} channels); unseen channels are excluded from the sample.")

    def _prepare_spatial_mask_layout_seed(self):
        """SEED 62-channel layout."""
        channelsMapping = pd.read_excel(os.path.join(os.path.dirname(self.data_root), 'channel-order.xlsx'), header=None, names=['channels'])
        channelsMapping = channelsMapping.replace({'CB1': 'I1', 'CB2': 'I2'})  # to conform to the 10-5 system convention
        channel_names = channelsMapping['channels'].tolist()
        self.channel_names = channel_names

        if self.spatial_mask_mode == '2x':
            if self.spatial_mask_layout == 0:
                self.visible_channel_idx = channelsMapping[~channelsMapping['channels'].str.endswith(('1', '5', '2', '6'))].index.tolist() 
            elif self.spatial_mask_layout == 1:
                self.visible_channel_idx = channelsMapping[~channelsMapping['channels'].str.endswith(('3', '7', '4', '8', 'Z'))].index.tolist() 
            elif self.spatial_mask_layout == 2:
                self.visible_channel_idx = channelsMapping[~channelsMapping['channels'].str.startswith(('AF', 'FT', 'FC', 'TP', 'CP', 'PO', 'I'))].index.tolist() 
            elif self.spatial_mask_layout == 3:
                self.visible_channel_idx = channelsMapping[channelsMapping['channels'].str.startswith(('AF', 'FT', 'FC', 'TP', 'CP', 'PO', 'I'))].index.tolist()
            self.visible_channels = channelsMapping.iloc[self.visible_channel_idx]['channels'].tolist()
        elif self.spatial_mask_mode == '4x':
            if self.spatial_mask_layout == 0:
                self.visible_channels = ['FPZ', 'AF3', 'AF4', 'FC5', 'FC1', 'FC2', 'FC6', 'CP5', 'CP1', 'CP2', 'CP6', 'PO3', 'PO4', 'I1', 'I2', 'OZ'] 
            elif self.spatial_mask_layout == 1:
                self.visible_channels = ['FP1', 'FP2', 'F5', 'FZ', 'F6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P5', 'PZ', 'P6', 'O1', 'O2'] 
            elif self.spatial_mask_layout == 2:
                self.visible_channels = ['FPZ', 'AF3', 'AF4', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'PO7', 'PO8', 'OZ']
            elif self.spatial_mask_layout == 3:
                self.visible_channels = ['FP1', 'FP2', 'F7', 'F3', 'F4', 'F8', 'C5', 'C1', 'C2', 'C6', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2'] 
            self.visible_channel_idx = [channel_names.index(ch) for ch in self.visible_channels if ch in channel_names]
        elif self.spatial_mask_mode == '8x':
            if self.spatial_mask_layout == 0:
                self.visible_channels = ['AF3', 'AF4', 'FC5', 'FC6', 'CP5', 'CP6', 'PO5', 'PO6']
            elif self.spatial_mask_layout == 1:
                self.visible_channels = ['FP1', 'FP2', 'FZ', 'T7', 'T8', 'PZ', 'O1', 'O2']
            elif self.spatial_mask_layout == 2:
                self.visible_channels = ['AF3', 'AF4', 'FC3', 'FC4', 'P3', 'P4', 'PO7', 'PO8']
            elif self.spatial_mask_layout == 3:
                self.visible_channels = ['FPZ', 'F5', 'F6', 'C3', 'C4', 'P5', 'P6', 'OZ']
            self.visible_channel_idx = [channel_names.index(ch) for ch in self.visible_channels if ch in channel_names]
        elif self.spatial_mask_mode == 'random':
            self.visible_channel_idx = list(range(len(channel_names)))
            self.visible_channels = channel_names
        else:
            raise ValueError(f"Unsupported spatial_mask_mode: '{self.spatial_mask_mode}'. Expected '2x', '4x', '8x', or 'random'.")

    def _prepare_spatial_mask_layout_physionet(self):
        """PhysioNet 64-channel layout."""
        db = lmdb.open(self.data_root, readonly=True, lock=False)
        with db.begin(write=False) as txn:
            any_keys = next(iter(self.keys_dict.values()))
            first = pickle.loads(txn.get(any_keys[0].encode()))
            channel_names = list(first['data_info']['channel_names'])
        db.close()
        self.channel_names = channel_names
        channelsMapping = pd.DataFrame({'channels': channel_names})

        SEED_TO_PN_SUB = {'I1': 'O1', 'I2': 'O2', 'PO5': 'PO3', 'PO6': 'PO4'}
        def _sub(seed_list):
            return [SEED_TO_PN_SUB.get(ch, ch) for ch in seed_list]

        if self.spatial_mask_mode == '2x':
            if self.spatial_mask_layout == 0:
                visible = [ch for ch in channelsMapping[~channelsMapping['channels'].str.endswith(('1', '5', '9', '2', '6', '10'))]['channels'].tolist()]
            elif self.spatial_mask_layout == 1:
                visible = channelsMapping[~channelsMapping['channels'].str.endswith(('3', '7', '4', '8', 'Z'))]['channels'].tolist()
            elif self.spatial_mask_layout == 2:
                visible = channelsMapping[~channelsMapping['channels'].str.startswith(('AF', 'FT', 'FC', 'TP', 'CP', 'PO', 'I'))]['channels'].tolist()
            elif self.spatial_mask_layout == 3:
                visible = channelsMapping[channelsMapping['channels'].str.startswith(('AF', 'FT', 'FC', 'TP', 'CP', 'PO', 'I'))]['channels'].tolist()
            else:
                raise ValueError(f"Unsupported spatial_mask_layout for PhysioNet 2x: {self.spatial_mask_layout}. Expected 0-3.")
            self.visible_channels = visible
        elif self.spatial_mask_mode == '4x':
            seed_lists = [
                ['FPZ', 'AF3', 'AF4', 'FC5', 'FC1', 'FC2', 'FC6', 'CP5', 'CP1', 'CP2', 'CP6', 'PO3', 'PO4', 'I1', 'I2', 'OZ'], 
                ['FP1', 'FP2', 'F5', 'FZ', 'F6', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P5', 'PZ', 'P6', 'O1', 'O2'],  
                ['FPZ', 'AF3', 'AF4', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'PO7', 'PO8', 'OZ'], 
                ['FP1', 'FP2', 'F7', 'F3', 'F4', 'F8', 'C5', 'C1', 'C2', 'C6', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2'], 
            ]
            self.visible_channels = _sub(seed_lists[self.spatial_mask_layout])
        elif self.spatial_mask_mode == '8x':
            seed_lists = [
                ['AF3', 'AF4', 'FC5', 'FC6', 'CP5', 'CP6', 'PO5', 'PO6'],
                ['FP1', 'FP2', 'FZ', 'T7', 'T8', 'PZ', 'O1', 'O2'],
                ['AF3', 'AF4', 'FC3', 'FC4', 'P3', 'P4', 'PO7', 'PO8'],
                ['FPZ', 'F5', 'F6', 'C3', 'C4', 'P5', 'P6', 'OZ'],
            ]
            self.visible_channels = _sub(seed_lists[self.spatial_mask_layout])
        else:
            raise ValueError(f"Unsupported spatial_mask_mode for PhysioNet: '{self.spatial_mask_mode}'. Expected '2x', '4x', or '8x'.")

        missing = [ch for ch in self.visible_channels if ch not in channel_names]
        if missing:
            raise RuntimeError(f"PhysioNet layout {self.spatial_mask_mode}/{self.spatial_mask_layout} requested channels missing from data: {missing}")
        self.visible_channel_idx = [channel_names.index(ch) for ch in self.visible_channels]
    
    def _get_coords(self, ch_names):
        """
        Get coordinates for the given channel names.
        Returns:
            coords: (C, 3)
            valid_mask: (C,) boolean array (True if channel found)
        """
        coords = []
        valid_mask = []
        for ch in ch_names:
            ch_clean = ch.replace('.', '').strip().lower()
            if ch_clean in self.pos_dict_lower:
                coords.append(self.pos_dict_lower[ch_clean])
                valid_mask.append(True)
            else:
                coords.append([0.0, 0.0, 0.0])
                valid_mask.append(False)
        coords_3d = np.array(coords, dtype=np.float32)

        return coords_3d, np.array(valid_mask)

    def _precompute_coords(self):
        """
        Pre-compute coordinate tensors once since channel layout is fixed across all samples.
        When normalize_coords=True, applies per-axis min-max normalization to [0, 1].
        """
        if self.visible_only:
            visible_ch_names = [self.channel_names[i] for i in self.visible_channel_idx]
            coords, valid_mask = self._get_coords(visible_ch_names)
        else:
            coords, valid_mask = self._get_coords(self.channel_names)

        if self.normalize_coords and coords is not None:
            ref = np.array(list(self.pos_dict.values()), dtype=np.float32)
            mn = ref.min(axis=0, keepdims=True)
            mx = ref.max(axis=0, keepdims=True)
            denom = np.maximum(mx - mn, 1e-8)
            coords = (coords - mn) / denom
            print(f"[normalize_coords] montage axis ranges: min={mn.squeeze().tolist()}, max={mx.squeeze().tolist()}")

        self._cached_coords = torch.from_numpy(coords).float() if coords is not None else None
        self._cached_valid_mask = valid_mask if valid_mask is not None else None

    def _preload_metadata(self):
        """Cache ch_names, label, sample_idx for each sample from LMDB."""
        print("Preloading metadata...")
        db = lmdb.open(self.data_root, readonly=True, lock=False, readahead=True, meminit=False)
        with db.begin(write=False) as txn:
            for idx, sample in enumerate(self.samples):
                pair = pickle.loads(txn.get(sample['key'].encode()))
                data_info = pair['data_info']
                ch_names = data_info['channel_names']
                if ch_names and len(ch_names) > self.eeg_channels:
                    ch_names = ch_names[:self.eeg_channels]
                sample['ch_names'] = ch_names or [f"Ch{i}" for i in range(self.eeg_channels)]
                sample['label'] = pair.get('label', None)
                sample['sample_idx'] = idx
                if self.normalization_mode == 'preloaded':
                    sample['norm_params'] = pair['norm_params']
        db.close()
        print(f"Preloaded metadata for {len(self.samples)} samples.")

    def _load_and_extract_segment(self, key):
        if self._db is None:
            self._db = lmdb.open(self.data_root, readonly=True, lock=False, readahead=True, meminit=False)
        with self._db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        self._db.close()
        self._db = None
        data = pair['sample']
        label = pair['label']
        data_info = pair.get('data_info', {})
        ch_names = data_info.get('channel_names', None)
        
        data = data.reshape(data.shape[0], -1)
        n_channels, n_timepoints = data.shape
        if n_timepoints >= self.eeg_timepoints:
            segment = data[:, :self.eeg_timepoints]
        else:
            raise ValueError(f"EEG timepoints {n_timepoints} is less than requested {self.eeg_timepoints}")
        if n_channels >= self.eeg_channels:
            segment = segment[:self.eeg_channels, :]
            ch_names = ch_names[:self.eeg_channels] if ch_names else None
        else:
            raise ValueError(f"EEG channels {n_channels} is less than requested {self.eeg_channels}")

        # VISIBLE_ONLY mode: keep only visible channels as defined by spatial_mask_mode
        # At eval/test time `visible_only` is False, so all channels are returned
        if self.visible_only and hasattr(self, 'visible_channel_idx'):
            segment = segment[self.visible_channel_idx, :]  # (len(visible), T)
            ch_names = [ch_names[i] for i in self.visible_channel_idx]

        orig_segment = segment

        if self.downsample_factor > 1.0:
            down = int(round(self.downsample_factor))
            segment = resample_poly(segment, up=1, down=down, axis=-1).astype(np.float32)

        return segment, ch_names, label, orig_segment

    def _normalize_sample(self, segment: torch.Tensor, observed_mask: torch.Tensor = None) -> tuple:
        """
        Normalize EEG segment based on normalization_mode.
        Args:
            segment: (C, T) tensor
            observed_mask: (C,) tensor, 1=observed 0=masked. If None, use all channels.
        Returns:
            normalized segment: (C, T) tensor
            norm_params: dict with normalization parameters for inverse transform
        """
        if self.normalization_mode == 'preloaded':
            return segment, {'type': 'preloaded'}
        elif self.normalization_mode == 'scale':
            segment = segment * self.scale_const
            return segment, {'type': 'scale', 'scale': self.scale_const,
                             'mean': 0.0, 'std': 1.0 / self.scale_const}
        elif self.normalization_mode == 'per_sample':
            all_channels_mean = float(segment.mean())
            all_channels_std = float(segment.std())
            if observed_mask is not None:
                obs_idx = (observed_mask > 0).nonzero(as_tuple=True)[0]
                obs_data = segment[obs_idx]
                mean = float(obs_data.mean())
                std = float(obs_data.std())
            else:
                mean = all_channels_mean
                std = all_channels_std
            if std == 0:
                print("All samples are the same value; std is 0")
                std = 1.0
            segment = (segment - mean) / std
            return segment, {
                'type': 'per_sample',
                'mean': mean, 'std': std,
                'all_channels_mean': all_channels_mean, 'all_channels_std': all_channels_std,
            }
        elif self.normalization_mode == 'per_channel':
            mean = segment.mean(dim=-1, keepdim=True) 
            std = segment.std(dim=-1, keepdim=True)
            std[std == 0] = 1.0
            segment = (segment - mean) / std
            return segment, {'type': 'per_channel', 'mean': mean, 'std': std}
        else:
            raise ValueError(f"Unsupported normalization_mode: '{self.normalization_mode}'")

    def _sample_mask_ratio(self, override=None):
        """Return the mask ratio for the current sample."""
        if override is not None:
            return float(override)
        if self.spatial_mask_ratio_range is not None:
            lo, hi = self.spatial_mask_ratio_range
            return random.uniform(lo, hi)
        return self.spatial_mask_ratio

    def _apply_random_mask(self, condition_tensor, valid_mask=None, ratio_override=None):
        """
        Apply spatial masking to condition tensor.
        Args:
            condition_tensor: (C, T)
        Returns:
            masked_condition_tensor (C, T)
            mask_vector (C,): 1=observed, 0=masked
        """
        ratio = self._sample_mask_ratio(ratio_override)
        if ratio <= 0.0:
            return condition_tensor, torch.ones(condition_tensor.shape[0], dtype=torch.float32)

        C = condition_tensor.shape[0]

        if valid_mask is not None:
            candidate_indices = np.where(valid_mask)[0]
        else:
            candidate_indices = np.arange(C)

        num_mask = int(len(candidate_indices) * ratio)
        mask_vector = torch.ones(C, dtype=torch.float32)

        if num_mask > 0:
            mask_indices = random.sample(list(candidate_indices), num_mask)
            masked_condition = condition_tensor.clone()
            masked_condition[mask_indices, :] = 0.0
            mask_vector[mask_indices] = 0.0
            return masked_condition, mask_vector

        return condition_tensor, mask_vector

    def _apply_fixed_mask(self, condition_tensor):
        """
        Fixed 2x/4x/8x downsampling mask: keep visible channels.
        Args:
            condition_tensor: (C, T)
        Returns:
            masked_condition_tensor (C, T)
            mask_vector (C,): 1=observed, 0=masked
        """
        C = condition_tensor.shape[0]
        mask_vector = torch.zeros(C, dtype=torch.float32)
        mask_vector[self.visible_channel_idx] = 1.0

        masked_condition = condition_tensor.clone()
        masked_condition[mask_vector == 0, :] = 0.0

        return masked_condition, mask_vector

    def __getitem__(self, idx):
        ratio_override = None
        if isinstance(idx, (tuple, list)):
            idx, ratio_override = idx

        sample_info = self.samples[idx]
        key         = sample_info['key']
        ch_names    = sample_info['ch_names']
        label       = sample_info['label']
        sample_idx  = sample_info['sample_idx']

        # 1) Load raw segment
        segment, ch_names, label, orig_segment = self._load_and_extract_segment(key)
        cond_data = segment
        eeg_tensor = torch.from_numpy(segment).float()          # (C, T)
        condition_tensor = torch.from_numpy(cond_data).float()  # (C, T)

        # 2) Get Coordinates (pre-computed)
        coords_tensor = self._cached_coords
        valid_mask = self._cached_valid_mask

        # 3) Apply spatial masking (mask_vector: (C,) 1 for observed, 0 for masked)
        # VISIBLE_ONLY mode: segment is already sliced to visible channels only, so apply random masking on that subset.
        if self.visible_only:
            condition_tensor, mask_vector = self._apply_random_mask(condition_tensor, valid_mask, ratio_override)
        elif self.spatial_mask_mode in ('2x', '4x', '8x'):
            condition_tensor, mask_vector = self._apply_fixed_mask(condition_tensor)
        elif self.spatial_mask_mode == 'random':
            condition_tensor, mask_vector = self._apply_random_mask(condition_tensor, valid_mask, ratio_override)
        else:
            raise ValueError(f"Unsupported spatial_mask_mode: '{self.spatial_mask_mode}'. Expected '2x', '4x', '8x', or 'random'.")

        # 4) Normalize using observed channels only
        eeg_tensor, norm_params = self._normalize_sample(eeg_tensor, observed_mask=mask_vector)
        condition, _ = self._normalize_sample(condition_tensor, observed_mask=mask_vector)
        if self.normalization_mode == 'preloaded':
            preloaded_norm_params = sample_info['norm_params']
            norm_params['mean'] = preloaded_norm_params.get('means', None)
            norm_params['std'] = preloaded_norm_params.get('stds', None)
        # Re-zero masked channels
        condition[mask_vector == 0] = 0.0

        # 5) Add channel dim
        eeg_tensor = eeg_tensor.unsqueeze(0)
        condition = condition.unsqueeze(0)

        # Optional pre-downsample GT (for temporal SR test)
        gt_eeg_full = None
        if self.return_orig_gt:
            orig_t = torch.from_numpy(orig_segment).float()
            orig_t, _ = self._normalize_sample(orig_t, observed_mask=mask_vector)
            gt_eeg_full = orig_t.unsqueeze(0)

        if self.split == 'train':
            eeg_input = eeg_tensor
        else:
            # Val/Test: Random noise input
            torch.manual_seed(self.seed)
            eeg_input = torch.randn_like(condition, dtype=torch.float32)

        # Expand mask to (1, C) for compatibility
        mask_tensor = mask_vector.unsqueeze(0)

        metadata = {
            "sample_name": key,
            "sample_idx": int(sample_idx),
            "eeg_shape": eeg_input.shape,
            "ch_names": ch_names,
            "coords": coords_tensor,
            "observed_mask": mask_tensor,
            "gt_eeg": eeg_tensor,
            "label": torch.tensor(label, dtype=torch.long) + 1,
            "norm_params": norm_params,
        }

        if gt_eeg_full is not None:
            metadata["gt_eeg_full"] = gt_eeg_full

        return eeg_input, condition, metadata
    
    def __len__(self):
        return len(self.keys)
