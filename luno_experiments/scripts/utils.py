"""Utility functions for scripts."""

import random
import numpy as np
import torch
import operator
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
import wandb
from luno_experiments.enums import Data, Method
from luno_experiments.data import APEBenchScenarios
from luno_experiments.plotting.pipeline import plot_uncertainty_vs_target
import io
from PIL import Image


def str_to_bool(value: str) -> bool:
    """Convert a string representation of a boolean to a boolean value.

    Args:
        value: A string representation of a boolean ("True" or "False").

    Returns:
        bool: The corresponding boolean value.

    Raises:
        ValueError: If the string does not represent a valid boolean value.
    """
    valid_values = {"True": True, "False": False}
    if value not in valid_values:
        msg = "invalid string representation of a boolean value"
        raise ValueError(msg)
    return valid_values[value]


def load_with_pickle(obj_save_path):
    with open(obj_save_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_with_pickle(obj, obj_save_path):
    with open(obj_save_path, "wb") as f:
        pickle.dump(obj, f)


def fix_random_seed(seed: int):
    """Fix random seed in numpy, scipy and torch backend."""
    # Python built-in RNG
    random.seed(seed)
    # NumPy RNG (also covers SciPy)
    np.random.seed(seed)  # noqa: NPY002
    # PyTorch CPU RNG
    torch.manual_seed(seed)
    # PyTorch GPU RNG (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(
    experiment_name: str,
    data_name: Data,
    method_name: Method,
    results: dict,
    samples: dict,
    *,
    ood_results: dict | None = None,
    ood_samples: dict | None = None,
    trajectory_results: dict | None = None,
    log_to_wandb: bool = True,
    save_dir: Path = Path("./results"),
):
    # Create results directory
    results_dir = Path(save_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    save_with_pickle(results, results_dir / "results.pkl")

    # Save samples
    save_with_pickle(samples, results_dir / "samples.pkl")

    # Save OOD results
    if ood_results is not None:
        save_with_pickle(ood_results, results_dir / "ood_results.pkl")

    # Save OOD samples
    if ood_samples is not None:
        save_with_pickle(ood_samples, results_dir / "ood_samples.pkl")
            
    # Save trajectory results
    if trajectory_results is not None:
        save_with_pickle(trajectory_results, results_dir / "trajectory_results.pkl")

    # Save wandb results
    if log_to_wandb:
        wandb.log(jax.tree.map(jnp.mean, results))

        if ood_results is not None:
            wandb.log(jax.tree.map(jnp.mean, ood_results))

        if trajectory_results is not None:
            wandb.log(jax.tree.map(jnp.mean, trajectory_results))

        # Log sample plot
        buffer = plot_uncertainty_vs_target(
            jax.tree.map(operator.itemgetter(0), samples),
            dims=1 if data_name in APEBenchScenarios else 2,
            title=f"{method_name} - {data_name}",
        )
        
        # Convert buffer to PIL Image and then to wandb Image
        image = Image.open(io.BytesIO(buffer.getvalue()))
        wandb.log({f"{method_name} - {data_name}": wandb.Image(image)})
