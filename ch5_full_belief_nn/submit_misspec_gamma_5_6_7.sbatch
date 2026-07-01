#!/bin/bash
#SBATCH --job-name=misg567
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=logs/misg567_%A_%a.out
#SBATCH --error=logs/misg567_%A_%a.err

set -euo pipefail

cd ~/market_making_fiplug_clean_v3/market_making_ch5_arrfull_redesign_20260526
source /home/spet6348/venv_q2/bin/activate
export PYTHONUNBUFFERED=1

GAMMAS=(5.0 6.0 7.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

echo "Testing gamma=${GAMMA} using trained model gamma=6.0"

python -u scripts/exp55_misspec_eval_baseline_nn.py \
  --q 0.6 \
  --gamma ${GAMMA} \
  --model_q 0.6 \
  --model_gamma 6.0 \
  --model_root outputs_ch5_arrfull/j3_final_v2_v100 \
  --out_root outputs_ch5_arrfull/misspec_baseline_nn \
  --chapter4_outputs outputs_ch5_arrfull/j3_final_v2_v100 \
  --preset quick \
  --device cuda
