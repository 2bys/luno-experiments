# luno-experiments

Experiments for the paper "Linearization Turns Neural Operators into Function-Valued Gaussian Processes".

## About the repo

This repository contains a simplified and refactored version of the original code used to produce the results in the referred paper. It produces similar results.

## Quick Start

### Installation

Start by cloning the repository and the submodule `luno`, which contains the main gist of the LUNO method.

```bash
git clone --recurse-submodules https://github.com/2bys/luno-experiments.git
```

Then, install the dependencies using uv.

```bash
# Or install the package in development mode
uv pip install -e deps/luno
uv pip install -e .
```

### Running Experiments

The repository supports two main experiment types:

1. **Low Data Regime Experiments** (APEBENCH datasets):
   - `diff_lin_1`, `diff_ks_cons_1`, `diff_hyp_diff_1`, `diff_burgers_1`

2. **Out-of-Distribution Experiments** (Advection-Diffusion-Reaction):
   - `base_2`, `flip_2`, `pos_2`, `pos_neg_2`, `pos_neg_flip_2`

For running the experiments on our ML cloud, I am using the `submit` package [here](https://github.com/2bys/submit). This will also be the commands suggested here in the README. 

### Training

```bash
# Train models using the provided script
python3 submit/submit.py --mode slurm --script train \
  --data_name diff_lin_1 diff_ks_cons_1 diff_hyp_diff_1 diff_burgers_1 \
  --num_epochs 100 \
  --batch_size 5 \
  --num_train_samples 25 \
  --seed 0
```

### Evaluation

```bash
# Evaluate trained models
python3 submit/submit.py --mode slurm --script evaluate \
  --data_name <dataset_name>
```

## Project Structure

- `luno_experiments/` - Main package containing experiment code
  - `scripts/` - Training, evaluation, and data generation scripts
  - `data/` - Data loading and processing utilities
  - `nn/` - Neural network implementations
  - `uncertainty/` - Uncertainty quantification methods
  - `plotting/` - Visualization utilities
- `scripts/` - Shell scripts for running experiments
- `data/` - Dataset storage
- `results/` - Experiment results and outputs

We will add the original plotting code in the near future.

## Methods

The repository implements several uncertainty quantification methods:
- Input perturbations
- Ensemble methods
- Sampling-based approaches (ISO/LA)
- LUNO-based approaches (ISO/LA)

## Dependencies

Key dependencies include:
- `linox` - Linear Operator framework
- `laplax` - Laplace approximation
- `flax` - Neural network library
- `wandb` - Experiment tracking
- `apebench` - Benchmark datasets

I will add more detailed information on how to run the code in the near future. Until then, feel free to reach out for guidance on how to use the code, any related question, or the original code run/checkpoints.