#!/bin/bash

# ============================================
# Experiment Configuration
# ============================================
# CHANGE THESE VARIABLES TO UPDATE EXP NAME AND WORKDIR AUTOMATICALLY
DATASET="seed"    # seed / physionet
MODE="2x"         # 2x / 4x / 8x
LAYOUT=0          # 0-3
VAL_ITER=30000
EXP_SUFFIX="neuroxfield_d512"
EXP_NAME="${DATASET}_${EXP_SUFFIX}_${MODE}${LAYOUT}"
TEST_NAME="test"
CONFIG_PATH="configs/neuroxfield_${DATASET}_d512.yaml"

# ============================================
# Setup
# ============================================
base_dir=$(cd "$(dirname "$0")" && pwd)
cd "$base_dir"

mkdir -p workdirs/logs
current_time=$(date "+%Y%m%d_%H%M")
log_path="workdirs/logs/${EXP_NAME}_${TEST_NAME}_${current_time}.txt"
exec > >(tee -a "$log_path") 2>&1

conda activate neuroxfield

# ============================================
# Run Training
# ============================================
export EXP_NAME
export WORKDIR_ROOT=workdirs

# Load the best checkpoint
CKPT_DIR="${WORKDIR_ROOT}/${EXP_NAME}/exp_${EXP_NAME}"
CKPT_GLOB="${CKPT_DIR}/best-*-${VAL_ITER}-val_nmse_masked*.ckpt"
CKPT_MATCHES=( $CKPT_GLOB )
if [ ! -e "${CKPT_MATCHES[0]}" ]; then
    echo "ERROR: no checkpoint matched ${CKPT_GLOB}" >&2
    exit 1
fi
if [ ${#CKPT_MATCHES[@]} -gt 1 ]; then
    echo "WARNING: multiple checkpoints matched ${CKPT_GLOB}; using first:" >&2
    printf '  %s\n' "${CKPT_MATCHES[@]}" >&2
fi
CKPT_PATH="${CKPT_MATCHES[0]}"
echo "Using ckpt: ${CKPT_PATH}"

DATASET_ARGS=()
if [ "$DATASET" = "physionet" ]; then
    DATASET_ARGS=(
        --data.pred_dataset.init_args.dataset physionet
        --data.pred_dataset.init_args.data_root /data/PhysioNet/physionet_processed_lmdb
        --data.pred_dataset.init_args.eeg_channels 64
        --data.pred_dataset.init_args.eeg_timepoints 640
        --model.denoiser.init_args.patch_size_t 80
        --model.clf_ckpt_path downstream/eegnet_physionet.pt
    )
fi

python main.py predict -c $CONFIG_PATH \
    --data.pred_dataset.init_args.spatial_mask_mode ${MODE} \
    --data.pred_dataset.init_args.spatial_mask_layout ${LAYOUT} \
    "${DATASET_ARGS[@]}" \
    --ckpt_path "${CKPT_PATH}"

echo "Testing completed."
echo "Total time: $SECONDS seconds"