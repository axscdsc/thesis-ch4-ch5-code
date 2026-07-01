#!/bin/bash
#SBATCH --job-name=fa45c
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/fa45c_%j.out
#SBATCH --error=logs/fa45c_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs_full_arrival

python exp45c_full_arrival_fiplug_sweeps.py \
  --mode both \
  --q_values 0,0.2,0.4,0.6,0.8,1.0 \
  --gamma_values 0,1,2,3 \
  --q_for_gamma 0.6 \
  --gamma_for_q 1 \
  --sims 100000 \
  --Nt 1000 \
  --J 7 \
  --outdir outputs_full_arrival/45c_fiplug_sweeps_J7
