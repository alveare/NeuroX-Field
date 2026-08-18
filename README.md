# NeuroXField: Arbitrary Scale EEG Predictive Super-Resolution with Neural Field Diffusion

---

## 1. Environment

```bash
conda create -n neuroxfield python=3.11
conda activate neuroxfield
pip install -r requirements.txt
```

The configs use `devices: auto`, so multi-GPU runs work without changes.

---

## 2. Data

Both datasets are preprocessed into LMDB stores once.

| Dataset | Source | Channels | Rate | Segment |
|---|---|---|---|---|
| SEED | [SEED](https://bcmi.sjtu.edu.cn/home/seed/) | 62 | 200 Hz | 800 samples (4 s) |
| PhysioNet-MM/MI | [physionet.org](https://physionet.org/content/eegmmidb/1.0.0/) | 64 | 160 Hz | 640 samples (4 s) |

### 2.1 Edit the paths

The two preprocessing scripts hold their paths as constants at the top of the
file. Point them at your copies:

```python
# src/data/preprocessing/preprocess_seed.py
DATA_DIR     = '/data/SEED/SEED_EEG/Preprocessed_EEG'
LABEL_FILE   = '/data/SEED/SEED_EEG/label.mat'
CHANNEL_FILE = '/data/SEED/channel-order.xlsx'
SAVE_DIR     = '/data/SEED/seed_processed_lmdb'

# src/data/preprocessing/preprocess_physionet.py
DATA_DIR = '/data/physionet.org/files/eegmmidb/1.0.0'
SAVE_DIR = '/data/PhysioNet/physionet_processed_lmdb'
```

### 2.2 Build the LMDB stores

```bash
python src/data/preprocessing/preprocess_seed.py
python src/data/preprocessing/preprocess_physionet.py
```

### 2.3 Point the configs at the stores

```yaml
# configs/neuroxfield_seed_d512.yaml
data_root: &data_root /data/SEED/seed_processed_lmdb
```

---

## 3. Training

```bash
./train_neuroxfield.sh
```

Edit the variables at the top of the script to choose the setting:

```bash
DATASET="seed"    # seed | physionet
MODE="2x"         # 2x | 4x | 8x   — how many electrodes stay observed
LAYOUT="0"        # 0-3            — which electrode layout within that mode
```

Or invoke the CLI directly:

```bash
python main.py fit -c configs/neuroxfield_seed_d512.yaml \
    --data.train_dataset.init_args.spatial_mask_mode 4x \
    --data.train_dataset.init_args.spatial_mask_layout 0
```

Checkpoints, logs, and TensorBoard events land under `workdirs/<EXP_NAME>/`.

### What `MODE` controls

`spatial_mask_mode` fixes the *visible layout* — the electrode subset the model
is allowed to see. 

With `visible_only: true` (the training datasets), electrodes outside the layout
are dropped from the sample entirely, so the model never observes them. 
At eval time `visible_only: false` returns the full montage, making prediction at
the unseen positions.

---

## 4. Evaluation

```bash
./test_neuroxfield.sh
```

Predictions are written as `.npz` under `workdirs/<EXP_NAME>/predict/`.

---

## 5. Repository layout

```
main.py                          LightningCLI entry point
configs/                         YAML files for experiments
src/
  lightning_model.py             training/validation/predict steps, metrics
  lightning_data.py              dataloaders, per-batch mask-ratio sampler
  models/
    neuroxfield.py               the model (encoder / cross-attn / NF decoder)
    layers/                      RMSNorm, SwiGLU, patch + timestep embedding
    conditioner/                 wraps the observed-electrode conditioning
  diffusion/
    base/                        scheduler / trainer / sampler interfaces
    flow_matching/               linear-path scheduler, trainer, Euler sampler
  data/
    eeg_dataset.py               LMDB reader, montage coords, spatial masking
    preprocessing/               SEED and PhysioNet → LMDB
  eval/eegnet.py                 frozen EEGNet for the downstream metric
```
