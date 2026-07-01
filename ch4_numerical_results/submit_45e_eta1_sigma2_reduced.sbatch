#!/bin/bash
#SBATCH --job-name=fa45e12r
#SBATCH --partition=medium
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/fa45e12r_%j.out
#SBATCH --error=logs/fa45e12r_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs_full_arrival/45e_appendix_diagnostics_reduced

python exp45e_arrfull_fiplug_heatmap_diagnostics_reduced.py \
  --diagnostics eta1_sigma2 \
  --gamma_values 0,1,2,3,4,5,6,7 \
  --outdir outputs_full_arrival/45e_appendix_diagnostics_reduced
