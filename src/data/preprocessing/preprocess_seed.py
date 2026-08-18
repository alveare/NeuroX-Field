import scipy
import os
import lmdb
import pickle
import numpy as np
import pandas as pd
from natsort import natsorted
import mne
import random


# Path settings
DATA_DIR     = '/data/SEED/SEED_EEG/Preprocessed_EEG'
LABEL_FILE   = '/data/SEED/SEED_EEG/label.mat'
CHANNEL_FILE = '/data/SEED/channel-order.xlsx'
SAVE_DIR     = '/data/SEED/seed_processed_lmdb'

# Signal processing settings
BANDPASS_WINDOW  = (0.5, 75.0) 
CLIP_UV          = 200
SFREQ            = 200.0 
SEGMENT_SEC      = 4.0 
TARGET_SEGMENT   = int(SFREQ * SEGMENT_SEC)
NORMALIZATION    = 'none'
FALSE_DATA       = []              
VAL_RATIO        = 0.1
VAL_SEED         = 42


def normalize_segment(segment: np.ndarray) -> tuple:
    """
    Normalize a segment based on NORMALIZATION setting.
    Args:
        segment: (C, T) numpy array
    Returns:
        normalized segment: (C, T) numpy array
        norm_params: dict with normalization parameters for inverse transform
    """
    if NORMALIZATION == 'none':
        return segment, {'type': 'none'}
    elif NORMALIZATION == 'per_sample':
        mean = float(segment.mean())
        std = float(segment.std())
        segment = (segment - mean) / std
        return segment, {'type': 'per_sample', 'means': [mean], 'stds': [std]}
    elif NORMALIZATION == 'per_channel':
        means = segment.mean(axis=-1)    # (C,)
        stds = segment.std(axis=-1)      # (C,)
        stds[stds == 0] = 1.0
        segment = (segment - means[:, None]) / stds[:, None]
        return segment, {'type': 'per_channel', 'means': means.tolist(), 'stds': stds.tolist()}
    elif NORMALIZATION == 'global':
        return segment, {'type': 'global'}
    else:
        raise ValueError(f"Unsupported NORMALIZATION: '{NORMALIZATION}'")


def load_xyz_from_elc(elc_path: str,
                      want_channels: list[str]) -> np.ndarray:
    """
    Load channel positions from ELC file (mm units).
    If you want to match the units with MNE montage, divide by 1000.

    Args:
        elc_path: path to ELC file
        want_channels: list of channel names
    Returns:
        np.ndarray: channel positions (n_channels, 3)
    """
    # Convert channel names to uppercase for case-insensitive matching
    want_up = [ch.upper() for ch in want_channels]

    # Skip first 4 lines and read file
    with open(elc_path, 'r') as f:
        lines = f.readlines()[4:]

    positions_start = labels_start = None
    for i, ln in enumerate(lines):
        ll = ln.strip().lower()
        if ll == "positions":
            positions_start = i + 1
        elif ll == "labels":
            labels_start = i + 1
            break
    if positions_start is None or labels_start is None:
        raise RuntimeError("ELC file does not have Positions/Labels section.")

    # Read electrode positions
    positions = []
    for ln in lines[positions_start:labels_start-1]:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        xyz = [float(p) for p in ln.split()[:3]]
        positions.append(np.array(xyz, dtype=float))

    # Read labels
    labels = [ln.strip().upper() for ln in lines[labels_start:]
              if ln.strip() and not ln.startswith('#')]

    if len(labels) != len(positions):
        raise RuntimeError("Number of labels and positions do not match.")

    # Mapping (maintain want order)
    xyz_list = []
    for ch in want_up:
        if ch in labels:
            idx = labels.index(ch)
            xyz_list.append(positions[idx])
        else:
            print(f"[ELC] Warning: {ch} not found; NaN inserted")
            xyz_list.append(np.full(3, np.nan))
    
    # shape: (len(want), 3)
    return np.vstack(xyz_list)

def load_xyz_from_montage(want_channels: list) -> np.ndarray:
    """
    Load channel positions from MNE standard_1005 montage (m units).
    
    Args:
        want_channels: list of channel names
    Returns:
        np.ndarray: channel positions (n_channels, 3)
    """
    montage = mne.channels.make_standard_montage('standard_1005')
    ch_pos = montage.get_positions()['ch_pos']
    ch_pos_lower = {k.lower(): v for k, v in ch_pos.items()}

    xyz_list = []
    for ch in want_channels:
        key = ch.lower()
        if key in ch_pos_lower:
            xyz_list.append(ch_pos_lower[key])
        else:
            print(f"[montage] Warning: {ch} not found; NaN inserted")
            xyz_list.append(np.full(3, np.nan))

    return np.vstack(xyz_list).astype(np.float32)  # (n_ch, 3), m units

def process_trial(eeg_trial: np.ndarray) -> np.ndarray:
    """
    Process a trial of EEG data.
    (62, T_orig @ 200Hz) → (N, 62, 800)
    bandpass → segment
    """
    # Bandpass filter
    if BANDPASS_WINDOW is not None:
        l_freq, h_freq = float(BANDPASS_WINDOW[0]), float(BANDPASS_WINDOW[1])
        eeg_trial = mne.filter.filter_data(
            eeg_trial.astype(np.float64, copy=False),
            sfreq=SFREQ,
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=False,
        ).astype(np.float32, copy=False)
        print(f"Bandpass filtered: {l_freq} Hz - {h_freq} Hz")

    # Segment into 4 seconds (800 timepoints)
    n_channels, n_timepoints = eeg_trial.shape
    n_segments = n_timepoints // TARGET_SEGMENT

    segments = eeg_trial.reshape(n_channels, n_segments, TARGET_SEGMENT).transpose(1, 0, 2)
    print(f"segments.shape: {segments.shape}")

    return segments


def main():
    # Load and clean up channel names
    channel_df    = pd.read_excel(CHANNEL_FILE, header=None)
    channel_names = [str(ch).strip() for ch in channel_df.iloc[:, 0].tolist()]
    print(f"Loaded {len(channel_names)} channels: {channel_names}")

    # Replace CB1, CB2 with I1, I2
    channel_names = [ch.replace('CB1', 'I1').replace('CB2', 'I2') for ch in channel_names]
    print(f"After replacement: {len(channel_names)} channels: {channel_names}")
    channels_filtered = channel_names

    # Make file list
    files = [file for file in os.listdir(DATA_DIR)]
    files = natsorted(files)

    # Train/Val/Test split
    train_files = files[:36]
    test_files = files[36:]
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")
    files_dict = {
        'train': train_files, # subject 1~12 (val is carved out of train after processing)
        'test': test_files,   # subject 13~15
    }
    dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }

    # Load labels
    labels = scipy.io.loadmat(LABEL_FILE)['label'][0]
    print("label.shape: ", labels.shape)

    # Load xyz_array
    xyz_array = load_xyz_from_montage(channels_filtered)
    print(f"xyz_array.shape: {xyz_array.shape}")
    
    # Create LMDB database
    db = lmdb.open(SAVE_DIR, map_size=20 * 1024 * 1024 * 1024)

    # Clip statistics
    clip_total_values = 0      # all (channel, timepoint) values that reach the clip step
    clip_clipped_values = 0    # of those, how many exceed ±CLIP_UV
    clip_total_segments = 0
    clip_clipped_segments = 0  # segments with >=1 clipped value

    # Process each key (train, val, test)
    for files_key in files_dict.keys():
        print("=" * 60)
        print(f"Processing {files_key}...")
        print("=" * 60)
        # Process each file
        for file in files_dict[files_key]:
            print(f"Processing {file} for {files_key}...")
            file_samples = 0

            session_id = file.split('.')[0]

            # Skip anomalous files
            if session_id in FALSE_DATA:
                print(f"Skipping anomalous file: {file}")
                continue
            
            # Load mat data
            try:
                mat_data = scipy.io.loadmat(os.path.join(DATA_DIR, file))
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            trial_keys = [k for k in mat_data.keys() if 'eeg' in k]
            
            # Process each trial
            for key, label in zip(trial_keys, labels):
                trial_samples = 0
                if 'eeg' not in key or key.startswith('__'):
                    print(f"Skipping {key} in {file}: not containing 'eeg' or starts with '__'")
                    continue
                
                eeg_trial = mat_data[key]
                
                # Ensure it's 2D array (channels x time)
                if eeg_trial.ndim != 2:
                    print(f"Skipping {key} in {file}: unexpected shape {eeg_trial.shape}")
                    continue
                
                # Segment handling
                n_channels, n_timepoints = eeg_trial.shape
                print(f"[{key}] n_channels: {n_channels}, n_timepoints: {n_timepoints}")
                n_segments = n_timepoints // TARGET_SEGMENT
                # Skip if too short
                if n_segments == 0:
                    print(f"Skipping {key} in {file}: too short ({n_timepoints})")
                    continue
                # Truncate to multiple of TARGET_SEGMENT
                eeg_trial = eeg_trial[:, :n_segments * TARGET_SEGMENT]

                eeg_segments = process_trial(eeg_trial)

                # Trial ID
                trial_num = key.split('eeg')[1]

                # Save each segment
                with db.begin(write=True) as txn:
                    for i, segment in enumerate(eeg_segments):
                        sample_key = f'{session_id}_trial{trial_num}_{i}'

                        if segment.mean() > 500:
                            print(f"Skipping {sample_key} in {file}: mean {segment.mean()} > 500 uV")
                            continue

                        # Clip the signal to ±CLIP_UV (skip entirely when CLIP_UV is None)
                        if CLIP_UV is not None:
                            # Track how much of this segment would be clipped
                            n_over = int((np.abs(segment) > CLIP_UV).sum())
                            clip_total_values += segment.size
                            clip_clipped_values += n_over
                            clip_total_segments += 1
                            clip_clipped_segments += int(n_over > 0)

                            segment = np.clip(segment, -CLIP_UV, CLIP_UV)

                        # Normalize
                        segment, norm_params = normalize_segment(segment)

                        # Prepare data dict
                        data_dict = {
                            'sample': segment.astype(np.float32),
                            'label': label,
                            'norm_params': norm_params,
                            'data_info': {
                                'Dataset': 'SEED',
                                'modality': 'EEG',
                                'release': None,
                                'session_id': session_id,
                                'trial_num': int(trial_num),
                                'task': 'Emotion Recognition',
                                'sampling_rate': SFREQ,
                                'segment_idx': i,
                                'start_time': i*SEGMENT_SEC,
                                'channel_names': channels_filtered,
                                'xyz_id': xyz_array,
                                }
                        }
                        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                        dataset[files_key].append(sample_key)
                        trial_samples += 1
                        file_samples += 1
                print(f" → {trial_samples} samples for trial {trial_num} in {file}") # trial end
            print(f" → {file_samples} samples for {file}") # session file end
            print("\n" + "=" * 60)
        print(f"Completed: {len(dataset[files_key])} samples for {files_key}") # key end

    # Randomly sample VAL_RATIO of train samples as validation set
    rng = random.Random(VAL_SEED)
    n_val = int(round(len(dataset['train']) * VAL_RATIO))
    val_keys = set(rng.sample(dataset['train'], n_val))
    dataset['val'] = [k for k in dataset['train'] if k in val_keys]
    dataset['train'] = [k for k in dataset['train'] if k not in val_keys]
    print(f"Train/Val split: train={len(dataset['train'])}, val={len(dataset['val'])}")

    # Compute global statistics
    all_keys = [k for split_keys in dataset.values() for k in split_keys]
    print(f"Computing global statistics over {len(all_keys)} samples...")
    sum_x, sum_sq_x, total_elements = 0.0, 0.0, 0

    with db.begin(write=False) as txn:
        for sample_key in all_keys:
            data_dict = pickle.loads(txn.get(sample_key.encode()))
            s = data_dict['sample']
            sum_x += s.sum()
            sum_sq_x += (s ** 2).sum()
            total_elements += s.size

    global_mean = sum_x / total_elements
    global_std = np.sqrt(sum_sq_x / total_elements - global_mean ** 2)
    print(f"Global mean: {global_mean:.6f}, std: {global_std:.6f}")

    if NORMALIZATION == 'global':
        with db.begin(write=True) as txn:
            for sample_key in all_keys:
                data_dict = pickle.loads(txn.get(sample_key.encode()))
                data_dict['sample'] = ((data_dict['sample'] - global_mean) / global_std).astype(np.float32)
                data_dict['norm_params'] = {
                    'type': 'global',
                    'mean': float(global_mean),
                    'std': float(global_std),
                }
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        print("Global normalization applied.")

    # Report clip statistics
    print("=" * 60)
    if CLIP_UV is None:
        print("Clip statistics: clipping disabled (CLIP_UV is None)")
    elif clip_total_values > 0:
        print(f"Clip statistics (threshold ±{CLIP_UV:g} uV)")
        print("-" * 60)
        print(f"Total segments : {clip_total_segments:,}")
        print(f"Total values   : {clip_total_values:,}")
        print(f"Value-level clipped   : {clip_clipped_values:,} "
              f"({100.0 * clip_clipped_values / clip_total_values:.4f}% of all values)")
        print(f"Segment-level clipped : {clip_clipped_segments:,} "
              f"({100.0 * clip_clipped_segments / clip_total_segments:.4f}% of segments have >=1 clipped value)")
    else:
        print("No values processed.")
    print("=" * 60)

    # Save keys index to lmdb
    print("Saving keys index...")
    with db.begin(write=True) as txn:
        txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))

    db.close()
    print("Done.")

if __name__ == "__main__":
    main()