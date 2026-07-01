#!/bin/bash
#SBATCH --job-name=cqalign
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/cqalign_%j.out
#SBATCH --error=logs/cqalign_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source /home/spet6348/venv_q2/bin/activate
export PYTHONUNBUFFERED=1

python -u scripts/eval_constant_quote_aligned.py \
  --sims 20000 \
  --Nt 1000 \
  --seed 36458647 \
  --outdir outputs_ch5_arrfull/constant_quote_aligned/q060_g600
