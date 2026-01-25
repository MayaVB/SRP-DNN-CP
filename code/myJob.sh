#!/bin/bash
#SBATCH --job-name=srpdnn_static
#SBATCH --account=ug_gannot
#SBATCH --partition=H200-12h
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=docker_job-%j.out
#SBATCH --error=docker_job-%j.err

echo "=== SLURM JOB INFO ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job GPUs: $SLURM_JOB_GPUS"
echo "Node:   $(hostname)"
echo "CWD:    $PWD"
echo "====================="

DockerName="slurm-job-${SLURM_JOB_ID}"

docker run --name "$DockerName" --rm \
  --gpus "device=${SLURM_JOB_GPUS}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/dsi/mayavb/PythonProjects/SRP-DNN:/workspace/SRP-DNN \
  -v /private:/dataset_folder \
  mayavb/srpdnn:b200-public-v1 \
  bash -lc '
    set -e

    echo "=== INSIDE CONTAINER ==="
    echo "PWD: $(pwd)"
    echo "USER: $(whoami)"
    echo "========================"

    export SRPDNN_WORKDIR=/workspace
    echo "SRPDNN_WORKDIR=$SRPDNN_WORKDIR"
    echo "HOME=$(echo ~)"

    echo "=== Choose python that has torch ==="
    if command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
      PY=python
    else
      PY=python3
    fi

    echo "PY=$PY"
    $PY - <<PY
import sys, torch
print("sys.executable:", sys.executable)
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

    echo "=== Ensure tensorboard in SAME env ==="
    $PY -m pip install --no-cache-dir tensorboard

    echo "=== tensorboard check ==="
    $PY - <<PY
import tensorboard
print("tensorboard:", tensorboard.__version__)
PY

    echo "=== Dataset link ==="
    mkdir -p /workspace/SRP-DNN/data/SrcSig
    ln -sfn /src/data/SrcSig/LibriSpeech /workspace/SRP-DNN/data/SrcSig/LibriSpeech
    ls -la /workspace/SRP-DNN/data/SrcSig/LibriSpeech/train-clean-100 | head -20

    echo "=== RUN SRP-DNN ==="
    $PY /workspace/SRP-DNN/code/RunSRPDNN.py \
      --train --gen-on-the-fly --use-amp --gpu-id 0 --source-state mobile
  '

echo "=== JOB COMPLETED ==="
