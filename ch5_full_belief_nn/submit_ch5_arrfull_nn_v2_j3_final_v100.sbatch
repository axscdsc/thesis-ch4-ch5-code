#!/bin/bash
#SBATCH --job-name=nnv2j3final
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/nnv2j3final_%j.out
#SBATCH --error=logs/nnv2j3final_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs_ch5_arrfull/j3_final_v2_v100

source /home/spet6348/venv_q2/bin/activate
export PYTHONUNBUFFERED=1

echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "Python: $(which python)"

nvidia-smi

python -u -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python -u scripts/exp55_train_compare_arrfull_nn_point_v2_ctmconly.py \
  --q 0.6 --gamma 6.0 \
  --J 3 \
  --preset final \
  --device cuda \
  --chapter4_outputs /home/spet6348/market_making_fiplug_clean_v3/market_making_full_arrival/outputs_full_arrival \
  --out_root outputs_ch5_arrfull/j3_final_v2_v100

echo "End time: $(date)"
