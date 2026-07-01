#!/bin/bash
#SBATCH --job-name=fa45e
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/fa45e_%j.out
#SBATCH --error=logs/fa45e_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs_full_arrival

python exp45e_arrfull_fiplug_heatmap_diagnostics.py \
  --diagnostics eta1_sigma1,eta10_sigma2,eta1_sigma2 \
  --q_values 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --gamma_values 0,1,2,3,4,5,6,7,8,9,10 \
  --sims 100000 \
  --Nt 1000 \
  --J 7 \
  --grid equidistant \
  --outdir outputs_full_arrival/45e_appendix_diagnostics
