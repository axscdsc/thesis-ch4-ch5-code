#!/bin/bash
#SBATCH --job-name=fig54v2
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/fig54v2_%j.out
#SBATCH --error=logs/fig54v2_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs figures_full_arrival_final

source /home/spet6348/venv_q2/bin/activate

python exp54_arrfull_filter_sample_paths_baseline.py \
  --psi_csv outputs_full_arrival/43b_filter_mse_gamma10_J7/psi_values.csv \
  --out_png figures_full_arrival_final/fig48_arrfull_filter_sample_paths_baseline_v2.png \
  --out_json figures_full_arrival_final/fig48_arrfull_filter_sample_paths_baseline_v2.json \
  --Nt 1000 \
  --T 1.0 \
  --J 7 \
  --seed_start 20260500 \
  --n_seed_trials 2000
