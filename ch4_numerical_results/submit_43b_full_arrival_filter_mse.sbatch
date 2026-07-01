#!/bin/bash
#SBATCH --job-name=fa43b
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/fa43b_%j.out
#SBATCH --error=logs/fa43b_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs_full_arrival

python exp43b_full_arrival_filter_mse.py \
  --q_points 11 \
  --gamma_points 13 \
  --gamma_max 3 \
  --sims 10000 \
  --Nt 1000 \
  --J 7 \
  --quote_policy zero_fiplug \
  --outdir outputs_full_arrival/43b_filter_mse_J7
