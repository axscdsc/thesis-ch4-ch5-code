#!/bin/bash
#SBATCH --job-name=nnpolicy
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/nnpolicy_%j.out
#SBATCH --error=logs/nnpolicy_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source /home/spet6348/venv_q2/bin/activate
export PYTHONUNBUFFERED=1

python -u scripts/plot_nn_policy_shapes.py \
  --model_dir outputs_ch5_arrfull/j3_final_v2_v100/points/q_0p60_gamma_6p00/model \
  --outdir outputs_ch5_arrfull/policy_shapes/q060_g600 \
  --device cuda \
  --t_value 0.5 \
  --q_min -20 \
  --q_max 20
