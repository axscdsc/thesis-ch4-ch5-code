#!/bin/bash
#SBATCH --job-name=fa43c
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/fa43c_%j.out
#SBATCH --error=logs/fa43c_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs_full_arrival

python exp43c_arrfull_discretisation_check.py \
  --points 0.5:2,0.6:1,0.6:3,0.9:1 \
  --J_values 3,5,7,15 \
  --grids equidistant,equal_probability \
  --sims 10000 \
  --Nt 1000 \
  --quote_policy zero_fiplug \
  --outdir outputs_full_arrival/43c_arrfull_discretisation_check
