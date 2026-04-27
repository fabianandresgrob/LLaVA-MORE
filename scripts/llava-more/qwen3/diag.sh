#!/bin/bash
#SBATCH --job-name=diag-env
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm
#SBATCH --output=%x_%j.out

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "=== CUDA-related env vars (before venv) ==="
env | grep -iE 'cuda|nvcc|nv_|ebrootcuda' | sort

echo ""
echo "=== nvcc location ==="
which nvcc 2>/dev/null || echo "nvcc not in PATH"
find /usr /opt /software 2>/dev/null -name nvcc -type f | head -5

echo ""
echo "=== Activating venv ==="
source "${VENV_PATH}/activate.sh"

echo ""
echo "=== CUDA-related env vars (after venv) ==="
env | grep -iE 'cuda|nvcc|nv_|ebrootcuda' | sort

echo ""
echo "=== nvcc after venv ==="
which nvcc 2>/dev/null || echo "nvcc still not in PATH"

echo ""
echo "=== torch CUDA_HOME ==="
python3 -c "
from torch.utils.cpp_extension import CUDA_HOME
import torch
print('torch.version.cuda:', torch.version.cuda)
print('torch CUDA_HOME:   ', CUDA_HOME)
print('cuda available:    ', torch.cuda.is_available())
"

echo ""
echo "=== deepspeed CUDA check ==="
python3 -c "
import os, sys
print('CUDA_HOME in env:', os.environ.get('CUDA_HOME', '(not set)'))
try:
    from deepspeed.ops.op_builder.builder import installed_cuda_version
    major, minor = installed_cuda_version()
    print(f'deepspeed installed_cuda_version: {major}.{minor}')
except Exception as e:
    print(f'deepspeed CUDA check failed: {e}')
"

echo ""
echo "=== Done ==="
