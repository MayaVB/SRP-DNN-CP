#!/bin/bash
#SBATCH --job-name=srpdnn_static_inference
#SBATCH --account=ug_gannot
#SBATCH --partition=H200-4h
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --chdir=/home/dsi/mayavb/PythonProjects/SRP-DNN

set -euo pipefail
mkdir -p logs

echo "=== SLURM JOB INFO ==="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "GPUs Device allocated (SLURM_JOB_GPUS): ${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES (host): ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Node:     $(hostname)"
echo "CWD:      ${PWD}"
echo "User:     $(whoami)"
echo "======================"

DockerName="slurm-job-${SLURM_JOB_ID}"

# >>>> SET THESE <<<<
TIME_ID="01081556"              # exp/<TIME_ID>/best_model.tar   01081556 or 01132213
GPU_ID="0"                      # with 1 allocated GPU, inside container it should be 0
EVAL_MODE="some"                # "some" or "all"
LOCALIZE_MODE="IDL unkNum 2"    # has spaces
SOURCES="1 2"                   # has spaces
# <<<<<<<<<<<<<<<<<<<

# Host paths
HOST_PROJ="/home/dsi/mayavb/PythonProjects/SRP-DNN"
HOST_EXP="${HOST_PROJ}/exp"
HOST_LS="${HOST_PROJ}/data/SrcSig/LibriSpeech"

echo "=== HOST PATH CHECK ==="
echo "Expect checkpoint: ${HOST_EXP}/${TIME_ID}/best_model.tar"
echo "Expect dataset dir: ${HOST_LS}/train-clean-100"

test -d "${HOST_PROJ}" || { echo "ERROR: missing ${HOST_PROJ}"; exit 10; }
test -d "${HOST_LS}/train-clean-100" || { echo "ERROR: missing ${HOST_LS}/train-clean-100"; exit 11; }
test -f "${HOST_EXP}/${TIME_ID}/best_model.tar" || {
  echo "ERROR: missing ${HOST_EXP}/${TIME_ID}/best_model.tar"
  ls -la "${HOST_EXP}" | head -n 200 || true
  exit 12
}

echo "OK: host paths exist."
echo "======================"

# Use Slurm’s masking if present, else fall back to SLURM_JOB_GPUS
GPU_VIS="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
echo "GPU_VIS (to pass into container) = ${GPU_VIS:-<empty>}"

docker run --name "${DockerName}" --rm \
  --gpus "device=${GPU_VIS}" \
  --ipc=host \
  -w /workspace/SRP-DNN \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HOME=/home/mayavb \
  -e XDG_CACHE_HOME=/home/mayavb/.cache \
  -e MPLCONFIGDIR=/home/mayavb/.cache/matplotlib \
  -e TORCH_HOME=/home/mayavb/.cache/torch \
  -e CUDA_VISIBLE_DEVICES="${GPU_VIS}" \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_VIS}" \
  -e TIME_ID="${TIME_ID}" \
  -e GPU_ID="${GPU_ID}" \
  -e EVAL_MODE="${EVAL_MODE}" \
  -e LOCALIZE_MODE="${LOCALIZE_MODE}" \
  -e SOURCES="${SOURCES}" \
  -v "${HOST_PROJ}:/workspace/SRP-DNN" \
  -v "${HOST_PROJ}:/home/mayavb/SRP-DNN" \
  -v "${HOST_EXP}:/workspace/SRP-DNN/exp" \
  -v "${HOST_EXP}:/home/mayavb/SRP-DNN/exp" \
  -v /private:/dataset_folder \
  mayavb/srpdnn:b200-public-v1 \
  bash -lc '
    set -euo pipefail
    set -x
    trap "st=\$?; echo CONTAINER_ERROR line=\$LINENO status=\$st cmd=\"\$BASH_COMMAND\"; exit \$st" ERR

    echo "=== INSIDE CONTAINER ==="
    echo "PWD:      $(pwd)"
    echo "USER:     $(whoami)"
    echo "HOST:     $(hostname)"
    echo "TIME_ID:  $TIME_ID"
    echo "GPU_ID:   $GPU_ID"
    echo "EVAL_MODE:$EVAL_MODE"
    echo "LOCALIZE_MODE: $LOCALIZE_MODE"
    echo "SOURCES:  $SOURCES"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
    echo "========================"

    mkdir -p /home/mayavb/.cache/matplotlib /home/mayavb/.cache/torch

    # Choose python
    if command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
      PY=python
    else
      PY=python3
    fi
    echo "PY=$PY"

    echo "=== GPU CHECK ==="
    nvidia-smi || true

    # IMPORTANT: do NOT escape $PY here
    $PY - <<'"'"'PY'"'"'
import os, sys, torch
print("sys.executable:", sys.executable)
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
print("cuda available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

    echo "=== CHECK PATHS THAT THE CODE USES ==="
    ls -la /home/mayavb/SRP-DNN/data/SrcSig/LibriSpeech/train-clean-100 | head -n 30 || true
    ls -la "/home/mayavb/SRP-DNN/exp/$TIME_ID/best_model.tar"
    ls -la "/workspace/SRP-DNN/exp/$TIME_ID/best_model.tar"

    echo "=== Ensure tensorboard exists (needed by code import) ==="
    export PYTHONUSERBASE=/tmp/pyuser
    export PATH=/tmp/pyuser/bin:$PATH
    $PY -m pip install --user --no-cache-dir -q tensorboard

    echo "=== RUN SRP-DNN INFERENCE (SIMULATE, STATIC) ==="
    read -r -a LOCALIZE_ARR <<< "$LOCALIZE_MODE"
    read -r -a SOURCES_ARR  <<< "$SOURCES"

    $PY /workspace/SRP-DNN/code/RunSRPDNN.py \
      --test \
      --gpu-id "$GPU_ID" \
      --use-amp \
      --time "$TIME_ID" \
      --source-state static \
      --gen-on-the-fly \
      --eval-mode simulate "$EVAL_MODE" \
      --localize-mode "${LOCALIZE_ARR[@]}" \
      --sources "${SOURCES_ARR[@]}"

    echo "=== DONE ==="
  ' 2>&1

echo "=== JOB COMPLETED ==="
