# Market Making When Filtering Fads from Order Arrivals

Code for Chapters 4 and 5 of the dissertation *Market Making When Filtering Fads from Order Arrivals*.

Repository: `axscdsc/thesis-ch4-ch5-code`

## Repository Structure

- `ch4_numerical_results/`: numerical experiments and submission scripts for the Chapter 4 results.
- `ch5_full_belief_nn/`: full-belief neural-network and PDE experiments for the Chapter 5 results.

## Environment

The codebase is written in Python. The main dependencies are listed in `requirements.txt`.

Recommended:

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.

## Main Dependencies

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

## Example Entry Points

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

## Related Literature

Some of the theoretical and modeling background is closely related to work such as:

- Emilio Barucci, Adrien Mathieu, and Leandro Sanchez-Betancourt, *Market Making with Fads, Informed, and Uninformed Traders* (2025).
- Alvaro Cartea and Leandro Sanchez-Betancourt, *Brokers and Informed Traders: Dealing with Toxic Flow and Extracting Trading Signals* (2025).
