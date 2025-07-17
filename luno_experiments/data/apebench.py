"""Wrapper for generating and handling APEBENCH data.

This module provides functionality to generate and handle data from the APEBENCH benchmark,
which includes various differential equation scenarios for testing machine learning models.
"""

import json

import apebench
import numpy as np

from luno_experiments.enums import Data

# Set of supported APEBENCH scenarios
APEBenchScenarios = {Data.DIFF_LIN_1, Data.DIFF_KS_CONS_1, Data.DIFF_HYP_DIFF_1, Data.DIFF_BURGERS_1}

# Configuration for each APEBENCH scenario
ScenarioConfigs = {
    k: {
        "scenario": "_".join(k.value.split("_")[:-1]),  # Extract scenario name without dimension suffix
        "num_spatial_dims": 1,                          # Number of spatial dimensions
        "num_points": 256,                              # Number of spatial points
        "train_temporal_horizon": 60,                   # Number of time steps for training
        "test_temporal_horizon": 60,                    # Number of time steps for testing
    }
    for k in APEBenchScenarios
}

def make_apebench_data(args):
    """
    Generate APEBENCH data for training, validation, and testing.

    This function generates data for the specified APEBENCH scenario and saves it in the
    appropriate format for each split (train/valid/test). It also saves metadata from
    the training split.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing:
        - data_name: Name of the APEBENCH scenario
        - data_root_dir: Root directory for saving the data
        - num_train_samples: Number of training samples
        - num_valid_samples: Number of validation samples
        - num_test_samples: Number of test samples
        - train_seed: Random seed for training set
        - valid_seed: Random seed for validation set
        - test_seed: Random seed for test set

    Returns
    -------
    None
        Saves the generated data and metadata to disk in the following structure:
        - {scenario}_{dim}/train_solutions.npz
        - {scenario}_{dim}/valid_solutions.npz
        - {scenario}_{dim}/test_solutions.npz
        - {scenario}_{dim}/metadata.json
    """
    # map splits to total counts and seeds
    splits = {
        "train": (args.num_train_samples, args.train_seed),
        "valid": (args.num_valid_samples, args.valid_seed),
        "test": (args.num_test_samples, args.test_seed),
    }

    # get scenario config
    cfg = ScenarioConfigs[args.data_name]

    # Initialize data generation
    out_dir = (args.data_root_dir / f"{cfg['scenario']}_{cfg['num_spatial_dims']}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = None
    # loop over each split
    for split, (total_samples, seed) in splits.items():
        if total_samples <= 0:
            continue

        # set samples and seed for current split
        cfg["num_train_samples"] = total_samples
        cfg["train_seed"] = seed
        cfg["num_test_samples"] = 0
        data, _, meta = apebench.scraper.scrape_data_and_metadata(**cfg)

        # save data
        out_dir.mkdir(exist_ok=True)
        np.savez(out_dir / f"{split}_solutions.npz", trajectories=data)

        # capture metadata from train split
        if split == "train":
            metadata = meta

    # write metadata once per scenario
    if metadata is not None:
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)