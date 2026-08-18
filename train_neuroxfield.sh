#!/bin/bash

# ============================================
# Experiment Configuration
# ============================================
# CHANGE THESE VARIABLES TO UPDATE EXP NAME AND WORKDIR AUTOMATICALLY
DATASET="seed"    # seed / physionet 
MODE="2x"         # 2x / 4x / 8x
LAYOUT="0"        # 0-3
EXP_SUFFIX="neuroxfield_d512"
EXP_NAME="${DATASET}_${EXP_SUFFIX}_${MODE}${LAYOUT}"
CONFIG_PATH="configs/neuroxfield_${DATASET}_d512.yaml"

# ============================================
# Setup
# ============================================
base_dir=$(cd "$(dirname "$0")" && pwd)
cd "$base_dir"

# rm -f "$HOME/.mne/mne-python.json.lock"

mkdir -p workdirs/logs
current_time=$(date "+%Y%m%d_%H%M")
log_path="workdirs/logs/${EXP_NAME}_${current_time}.txt"
exec > >(tee -a "$log_path") 2>&1

conda activate neuroxfield

# ============================================
# Run Training
# ============================================
export EXP_NAME
export WORKDIR_ROOT=workdirs

# Per-dataset overrides (channels/timepoints/patch_size_t/data_root)
DATASET_ARGS=()
if [ "$DATASET" = "physionet" ]; then
    DATASET_ARGS=(
        --data.train_dataset.init_args.dataset physionet
        --data.train_dataset.init_args.data_root /data/PhysioNet/physionet_processed_lmdb
        --data.train_dataset.init_args.eeg_channels 64
        --data.train_dataset.init_args.eeg_timepoints 640
        --data.eval_dataset.init_args.dataset physionet
        --data.eval_dataset.init_args.data_root /data/PhysioNet/physionet_processed_lmdb
        --data.eval_dataset.init_args.eeg_channels 64
        --data.eval_dataset.init_args.eeg_timepoints 640
        --model.denoiser.init_args.patch_size_t 80
        --model.clf_ckpt_path downstream/eegnet_physionet.pt
    )
fi

python main.py fit -c $CONFIG_PATH \
    --data.train_dataset.init_args.spatial_mask_mode ${MODE} \
    --data.train_dataset.init_args.spatial_mask_layout ${LAYOUT} \
    --data.eval_dataset.init_args.spatial_mask_mode ${MODE} \
    --data.eval_dataset.init_args.spatial_mask_layout ${LAYOUT} \
    "${DATASET_ARGS[@]}"

echo "Training completed."
echo "Total time: $SECONDS seconds"