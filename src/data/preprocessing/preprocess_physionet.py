import os
import lmdb
import pickle
import random
import numpy as np
import mne


# ============================================
# Path settings
# ============================================
DATA_DIR = '/data/physionet.org/files/eegmmidb/1.0.0'
SAVE_DIR = '/data/PhysioNet/physionet_processed_lmdb'

# ============================================
# Subject split settings
# ============================================
SUBJECT_TRAIN_END = 88    # S001..S088 → train (10% then carved to val)
SUBJECT_TOTAL = 109       # S089..S109 → test
VAL_RATIO = 0.1
VAL_SEED = 42
FALSE_DATA = {}

# ============================================
# Run selection (PhysioNet EEG-MMIDB has R01..R14 per subject)
#   R01: baseline eyes open                   R02: baseline eyes closed
#   R03/R07/R11: open & close left/right fist (real)
#   R04/R08/R12: imagine open & close L/R fist
#   R05/R09/R13: open & close both fists/feet (real)
#   R06/R10/R14: imagine open & close both fists/feet
# Default: include every run for SEED-style continuous segmentation.
# ============================================
RUNS = [f"{i:02d}" for i in range(3, 15)]

# Task class mapping by run number:
#   0: real, L/R fist           (R03/R07/R11)
#   1: imagine, L/R fist        (R04/R08/R12)
#   2: real, both fists/feet    (R05/R09/R13)
#   3: imagine, both fists/feet (R06/R10/R14)
RUN_TO_TASK = {
    3: 0, 7: 0, 11: 0,
    4: 1, 8: 1, 12: 1,
    5: 2, 9: 2, 13: 2,
    6: 3, 10: 3, 14: 3,
}

# ============================================
# Signal processing settings
# ============================================
BANDPASS_WINDOW = (0.5, 75.0) 
NOTCH_FREQS     = [60.0]
AVERAGE_REF     = True
CLIP_UV         = 200.0
SFREQ           = 160.0
SEGMENT_SEC     = 4.0
TARGET_SEGMENT  = int(SFREQ * SEGMENT_SEC)
NORMALIZATION   = 'none'


def clean_channel_name(edf_ch_name: str) -> str:
    """EDF spelling (e.g. 'Fc5.', 'C3..', 'Iz..') → clean uppercase ('FC5', 'C3', 'IZ')."""
    return edf_ch_name.rstrip('.').upper()

def normalize_segment(segment: np.ndarray) -> tuple:
    if NORMALIZATION == 'none':
        return segment, {'type': 'none'}
    elif NORMALIZATION == 'per_sample':
        mean = float(segment.mean())
        std = float(segment.std())
        segment = (segment - mean) / std
        return segment, {'type': 'per_sample', 'means': [mean], 'stds': [std]}
    elif NORMALIZATION == 'per_channel':
        means = segment.mean(axis=-1)
        stds = segment.std(axis=-1)
        stds[stds == 0] = 1.0
        segment = (segment - means[:, None]) / stds[:, None]
        return segment, {'type': 'per_channel', 'means': means.tolist(), 'stds': stds.tolist()}
    elif NORMALIZATION == 'global':
        return segment, {'type': 'global'}
    else:
        raise ValueError(f"Unsupported NORMALIZATION: '{NORMALIZATION}'")


def load_xyz_from_montage(want_channels: list) -> np.ndarray:
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
    return np.vstack(xyz_list).astype(np.float32)


def process_run(raw: mne.io.BaseRaw, edf_picks: list[str]) -> np.ndarray:
    """
    Run-level signal processing (in-place on `raw`):
      pick channels (in fixed order) → interpolate bads → average ref →
      bandpass → notch → resample → return data in uV (n_channels, T).
    """
    raw.pick(edf_picks)

    if len(raw.info['bads']) > 0:
        print(f"  interpolating bads: {raw.info['bads']}")
        raw.interpolate_bads()

    if AVERAGE_REF:
        raw.set_eeg_reference(ref_channels='average', verbose=False)

    if BANDPASS_WINDOW is not None:
        l_freq, h_freq = float(BANDPASS_WINDOW[0]), float(BANDPASS_WINDOW[1])
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    if NOTCH_FREQS:
        raw.notch_filter(NOTCH_FREQS, verbose=False)

    return raw.get_data(units='uV').astype(np.float32)


def main():
    # Subject IDs
    train_subjects = [f"S{i:03d}" for i in range(1, SUBJECT_TRAIN_END + 1)]
    test_subjects  = [f"S{i:03d}" for i in range(SUBJECT_TRAIN_END + 1, SUBJECT_TOTAL + 1)]
    print(f"Train subjects: {len(train_subjects)} ({train_subjects[0]}..{train_subjects[-1]})")
    print(f"Test  subjects: {len(test_subjects)} ({test_subjects[0]}..{test_subjects[-1]})")
    print(f"Runs per subject: {RUNS}")

    subjects_dict = {'train': train_subjects, 'test': test_subjects}
    dataset = {'train': [], 'val': [], 'test': []}

    # ---- Discover canonical channel order from the first available EDF ----
    edf_picks = None
    clean_channels = None
    for subj in train_subjects + test_subjects:
        for run in RUNS:
            ref_path = os.path.join(DATA_DIR, subj, f"{subj}R{run}.edf")
            if os.path.exists(ref_path):
                ref_raw = mne.io.read_raw_edf(ref_path, preload=False, verbose=False)
                edf_picks = list(ref_raw.ch_names)
                clean_channels = [clean_channel_name(n) for n in edf_picks]
                print(f"Discovered {len(clean_channels)} channels from {subj}R{run}: {clean_channels}")
                break
        if edf_picks is not None:
            break
    if edf_picks is None:
        raise RuntimeError("No EDF files found under DATA_DIR — cannot infer channel layout.")

    xyz_array = load_xyz_from_montage(clean_channels)
    print(f"xyz_array.shape: {xyz_array.shape}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    db = lmdb.open(SAVE_DIR, map_size=20 * 1024 * 1024 * 1024)

    # Clip statistics
    clip_total_values = 0      # all (channel, timepoint) values that reach the clip step
    clip_clipped_values = 0    # of those, how many exceed ±CLIP_UV
    clip_total_segments = 0
    clip_clipped_segments = 0  # segments with >=1 clipped value

    for split_key in ['train', 'test']:
        print("=" * 60)
        print(f"Processing {split_key}...")
        print("=" * 60)

        for subject in subjects_dict[split_key]:
            # Skip anomalous subjects (see FALSE_DATA)
            if subject in FALSE_DATA.get(split_key, []):
                print(f"[skip] {subject}: anomalous subject excluded from {split_key}")
                continue

            subj_dir = os.path.join(DATA_DIR, subject)
            if not os.path.isdir(subj_dir):
                print(f"[skip] {subject}: directory not found")
                continue

            subject_samples = 0
            for run in RUNS:
                edf_path = os.path.join(subj_dir, f"{subject}R{run}.edf")
                if not os.path.exists(edf_path):
                    print(f"[skip] {subject}R{run}: file not found")
                    continue

                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                except Exception as e:
                    print(f"[skip] {subject}R{run}: load error {e}")
                    continue

                missing = [n for n in edf_picks if n not in raw.ch_names]
                if missing:
                    print(f"[skip] {subject}R{run}: missing channels {missing[:3]}{'...' if len(missing) > 3 else ''}")
                    continue

                try:
                    eeg_data = process_run(raw, edf_picks)
                except Exception as e:
                    print(f"[skip] {subject}R{run}: processing error {e}")
                    continue

                # Segment into 4-second blocks
                n_channels, n_timepoints = eeg_data.shape
                n_segments = n_timepoints // TARGET_SEGMENT
                if n_segments == 0:
                    print(f"[skip] {subject}R{run}: too short ({n_timepoints} timepoints)")
                    continue
                eeg_data = eeg_data[:, :n_segments * TARGET_SEGMENT]
                segments = eeg_data.reshape(n_channels, n_segments, TARGET_SEGMENT).transpose(1, 0, 2)

                run_int = int(run)
                task_label = RUN_TO_TASK[run_int]

                with db.begin(write=True) as txn:
                    run_samples = 0
                    for i, segment in enumerate(segments):
                        sample_key = f"{subject}_R{run}_{i}"

                        if abs(segment.mean()) > 500:
                            print(f"  [skip] {sample_key}: |mean| {segment.mean():.2f} > 500 uV")
                            continue

                        # Clip the signal to ±CLIP_UV (skip entirely when CLIP_UV is None)
                        if CLIP_UV is not None:
                            # Track how much of this segment would be clipped (pre-clip)
                            n_over = int((np.abs(segment) > CLIP_UV).sum())
                            clip_total_values += segment.size
                            clip_clipped_values += n_over
                            clip_total_segments += 1
                            clip_clipped_segments += int(n_over > 0)

                            segment = np.clip(segment, -CLIP_UV, CLIP_UV)
                        segment, norm_params = normalize_segment(segment)

                        data_dict = {
                            'sample': segment.astype(np.float32),
                            'label': task_label,
                            'norm_params': norm_params,
                            'data_info': {
                                'Dataset': 'PhysioNet-MMI',
                                'modality': 'EEG',
                                'release': '1.0.0',
                                'session_id': subject,
                                'trial_num': run_int,
                                'task': 'Motor Movement & Imagery',
                                'sampling_rate': SFREQ,
                                'segment_idx': i,
                                'start_time': i * SEGMENT_SEC,
                                'channel_names': clean_channels,
                                'xyz_id': xyz_array,
                            },
                        }
                        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                        dataset[split_key].append(sample_key)
                        subject_samples += 1
                        run_samples += 1
                print(f"  {subject}R{run}: {run_samples} samples")
            print(f"  → {subject}: {subject_samples} total samples")
        print(f"Completed: {len(dataset[split_key])} samples for {split_key}")

    # Carve val from train
    rng = random.Random(VAL_SEED)
    n_val = int(round(len(dataset['train']) * VAL_RATIO))
    val_keys = set(rng.sample(dataset['train'], n_val))
    dataset['val'] = [k for k in dataset['train'] if k in val_keys]
    dataset['train'] = [k for k in dataset['train'] if k not in val_keys]
    print(f"Train/Val split: train={len(dataset['train'])}, val={len(dataset['val'])}")

    # Compute global statistics
    all_keys = [k for ks in dataset.values() for k in ks]
    print(f"Computing global statistics over {len(all_keys)} samples...")
    sum_x, sum_sq_x, total_elements = 0.0, 0.0, 0
    with db.begin(write=False) as txn:
        for sample_key in all_keys:
            data_dict = pickle.loads(txn.get(sample_key.encode()))
            s = data_dict['sample']
            sum_x += s.sum()
            sum_sq_x += (s ** 2).sum()
            total_elements += s.size

    if total_elements > 0:
        global_mean = sum_x / total_elements
        global_std = float(np.sqrt(max(sum_sq_x / total_elements - global_mean ** 2, 0.0)))
        print(f"Global mean: {global_mean:.6f}, std: {global_std:.6f}")
    else:
        global_mean, global_std = 0.0, 1.0

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
