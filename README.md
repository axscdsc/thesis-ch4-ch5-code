# Code for Thesis Chapters 4 and 5

This repository contains the code used for the Chapter 4 and Chapter 5 experiments.

## Structure

- `ch4_numerical_results/`: numerical experiments and submission scripts for the Chapter 4 results.
- `ch5_full_belief_nn/`: full-belief neural-network and PDE experiments for the Chapter 5 results.

## Environment

The codebase is written in Python. The main dependencies are listed in `requirements.txt`.

Recommended:

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.

## Main dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `torch`
- `pandas`

## Notes

- Many scripts are intended to be run from inside their chapter directory.
- Several `.sbatch` files are included for cluster submission workflows.
- Some Chapter 5 experiments require a CUDA-enabled PyTorch installation if you want GPU training.

## Example entry points

Chapter 4:

- `ch4_numerical_results/exp43b_full_arrival_filter_mse.py`
- `ch4_numerical_results/exp45c_full_arrival_fiplug_sweeps.py`
- `ch4_numerical_results/exp45d_arrfull_fiplug_heatmap.py`
- `ch4_numerical_results/exp54_arrfull_filter_sample_paths_baseline.py`

Chapter 5:

- `ch5_full_belief_nn/exp55_train_compare_arrfull_nn_point_v2_ctmconly.py`
- `ch5_full_belief_nn/exp55_misspec_eval_baseline_nn.py`
- `ch5_full_belief_nn/exp56_refined_common_fiplug_benchmark.py`
- `ch5_full_belief_nn/arrfull_pde_j3_solver_v3_semilag.py`

## Reproducibility

If you plan to share this repository publicly, add the thesis title, a short abstract, and a citation block here so other readers can connect the code to the paper.
