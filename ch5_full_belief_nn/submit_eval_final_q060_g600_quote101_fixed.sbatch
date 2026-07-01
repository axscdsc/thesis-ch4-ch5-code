#!/bin/bash
#SBATCH --job-name=evalq101f
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/evalq101f_%j.out
#SBATCH --error=logs/evalq101f_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /home/spet6348/venv_q2/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start time: $(date)"
echo "Node: $(hostname)"
nvidia-smi

python -u scripts/exp55_train_compare_arrfull_nn_point_v2_ctmconly.py \
  --q 0.6 --gamma 6.0 \
  --J 3 \
  --preset final \
  --device cuda \
  --skip_training \
  --batch_paths 50 \
  --chapter4_outputs /home/spet6348/market_making_fiplug_clean_v3/market_making_full_arrival/outputs_full_arrival \
  --out_root outputs_ch5_arrfull/j3_final_v2_v100_quote101

echo "End time: $(date)"
