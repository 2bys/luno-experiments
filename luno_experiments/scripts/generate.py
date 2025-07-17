"""Data generation script for LUNO experiments.

This script provides functionality for generating training, validation, and test
datasets for both APEBENCH and advection-diffusion-reaction experiments. It
supports various data scenarios and allows for reproducible data generation
through seed control.
"""

import argparse
from pathlib import Path

from loguru import logger

from luno_experiments.data.apebench import APEBenchScenarios, make_apebench_data
from luno_experiments.scripts.utils import str_to_bool, fix_random_seed
from luno_experiments.enums import Data
from luno_experiments.data.adv_diff_react import AdvDiffReactScenarios, generate_adv_diff_react_data


def parse_args(args: argparse.Namespace | None = None) -> argparse.Namespace:
    """
    Parse command line arguments for data generation.

    This function sets up the argument parser with all necessary parameters for
    generating datasets, including data type, sample counts, and random seeds
    for reproducibility.

    Parameters
    ----------
    args : argparse.Namespace or None, optional
        Pre-parsed arguments to use instead of sys.argv, by default None

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - data_root_dir: Root directory for storing data
        - data_name: Name of the dataset to generate (from Data enum)
        - num_train_samples: Number of training samples
        - num_valid_samples: Number of validation samples
        - num_test_samples: Number of test samples
        - train_seed: Random seed for training data
        - valid_seed: Random seed for validation data
        - test_seed: Random seed for test data
        - seed: Global random seed for backend
        - verbose: Whether to print verbose output

    Notes
    -----
    The function provides default values for all parameters:
    - Default data root directory: "./data/"
    - Default dataset: Data.DIFF_LIN_1
    - Default sample counts: 100 train, 250 valid, 250 test
    - Default seeds: 0 (train), 12894 (valid), 59801 (test), 1025 (global)
    """
    p = argparse.ArgumentParser(
        description="Generate `apebench`-based PDE data for LUNO experiments."
    )
    p.add_argument(
        "--data_root_dir", 
        type=Path, 
        default="./data/",
        help="Root directory for storing all data.",
    )
    p.add_argument(
        "--data_name",
        type=Data,
        default=Data.DIFF_LIN_1,
        choices=list(Data),
        help="Name of the data to generate.",
    )
    p.add_argument(
        "--num_train_samples", 
        type=int, 
        default=100,
        help="Number of training samples to generate.",
    )
    p.add_argument(
        "--num_valid_samples", 
        type=int, 
        default=250,
        help="Number of validation samples to generate.",
    )
    p.add_argument(
        "--num_test_samples", 
        type=int, 
        default=250,
        help="Number of test samples to generate.",
    )

    p.add_argument(
        "--train_seed",
        type=int,
        default=0,
        help="Seed for training data generation.",
    )

    p.add_argument(
        "--valid_seed",
        type=int,
        default=12894,
        help="Seed for validation data generation.",
    )

    p.add_argument(
        "--test_seed",
        type=int,
        default=59801,
        help="Seed for test data generation.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1025,
        help="Random seed for backend."
    )
    
    p.add_argument(
        "--verbose", 
        type=str_to_bool, 
        default=False,
        help="Whether to print verbose output.",
    )

    return p.parse_args(args)


def main(args: argparse.Namespace | None = None) -> None:
    """
    Generate datasets for LUNO experiments.

    This function handles the complete data generation process:
    1. Sets global random seed for reproducibility
    2. Determines the appropriate data generation function based on data_name
    3. Generates the dataset with specified parameters
    4. Logs the generation progress

    Parameters
    ----------
    args : argparse.Namespace or None, optional
        Command line arguments parsed by parse_args(), by default None

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If data_name is not supported (not in APEBenchScenarios or AdvDiffReactScenarios)

    Notes
    -----
    The function supports two types of data generation:
    - APEBENCH data: Generated in ./data/apebench/
    - Advection-diffusion-reaction data: Generated in ./data/ood/
    """
    # Fix random seed
    fix_random_seed(args.seed)

    # Generate data
    if args.data_name in APEBenchScenarios:
        logger.info(f"Generating {args.data_name} data...")
        args.data_root_dir = args.data_root_dir / "apebench"
        make_apebench_data(args)
        logger.info("Data generation complete.")

    elif args.data_name in AdvDiffReactScenarios:
        logger.info(f"Generating {args.data_name} data...")
        args.data_root_dir = args.data_root_dir / "ood"
        generate_adv_diff_react_data(args)
        logger.info("Data generation complete.")

    else:
        raise ValueError(f"Invalid data name: {args.data_name}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
