#!/bin/bash
#SBATCH --job-name=misg6H100
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_sku:H100
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/misg6H100_%j.out
#SBATCH --error=logs/misg6H100_%j.err

set -euo pipefail

cd ~/market_making_fiplug_clean_v3/market_making_ch5_arrfull_redesign_20260526
source /home/spet6348/venv_q2/bin/activate
export PYTHONUNBUFFERED=1

nvidia-smi

python -u scripts/exp55_misspec_eval_baseline_nn.py \
  --q 0.6 \
  --gamma 6.0 \
  --model_q 0.6 \
  --model_gamma 6.0 \
  --J 3 \
  --model_root outputs_ch5_arrfull/j3_final_v2_v100 \
  --out_root outputs_ch5_arrfull/misspec_baseline_nn_retry_H100_J3 \
  --chapter4_outputs outputs_ch5_arrfull/j3_final_v2_v100 \
  --preset quick \
  --device cuda
