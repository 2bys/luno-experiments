"""Training script for Fourier Neural Operator models.

This script provides functionality for training FNO models on various datasets,
including advection-diffusion-reaction and APEBENCH data. It handles model
initialization, data loading, training loop, and metric logging through
Weights & Biases or CSV files.
"""

import argparse
from luno_experiments.scripts.utils import fix_random_seed
from luno_experiments.nn.model import FNO
from luno_experiments.nn.trainer import Trainer
from luno_experiments.enums import Data
from luno_experiments.data.loaders import get_data_loaders
from pathlib import Path
from luno_experiments.data.adv_diff_react import AdvDiffReactScenarios
from luno_experiments.scripts.utils import str_to_bool
from luno_experiments.scripts.config import ModelConfig


def parse_args(
    args: argparse.Namespace | None = None
) -> argparse.Namespace:
    """
    Parse command line arguments for model training.

    This function sets up the argument parser with all necessary parameters for
    training an FNO model, including data configuration, training hyperparameters,
    and logging options.

    Parameters
    ----------
    args : argparse.Namespace or None, optional
        Pre-parsed arguments to use instead of sys.argv, by default None

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - data_name: Dataset to use (from Data enum)
        - num_epochs: Number of training epochs
        - batch_size: Batch size for training
        - num_train_samples: Number of training samples to use
        - seed: Random seed for reproducibility
        - enable_progress_bar: Whether to show progress bar
        - debug: Whether to enable debug mode
        - check_val_every_n_epoch: Validation frequency
        - wandb: Whether to use Weights & Biases logging
        - model_hparams: Model hyperparameters from ModelConfig
        - learning_rate: Learning rate for optimizer
        - weight_decay: Weight decay for optimizer
        - folder_path: Path to dataset directory
        - model_id: Unique identifier for the model

    Notes
    -----
    The function automatically adds several derived parameters:
    - model_hparams: Retrieved from ModelConfig based on data_name
    - learning_rate: Fixed at 1e-3
    - weight_decay: Fixed at 1e-4
    - folder_path: Constructed based on data source (ood/apebench)
    - model_id: Generated from model configuration parameters
    """
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_name",
        type=Data,
        default=Data.DIFF_LIN_1,
        help="Data name to use",
    )

    p.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs to train for",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size to use",
    )

    p.add_argument(
        "--num_train_samples",
        type=int,
        default=25,
        help="Number of training samples to use",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to use",
    )

    p.add_argument(
        "--enable_progress_bar",
        type=str_to_bool,
        default=True,
        help="Enable progress bar",
    )

    p.add_argument(
        "--debug",
        type=str_to_bool,
        default=False,
        help="Enable debug mode",
    )

    p.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="Check validation every n epoch",
    )

    p.add_argument(
        "--wandb",
        type=str_to_bool,
        default=True,
        help="Enable wandb",
    )

    args = p.parse_args(args)

    # Add constant parameters
    args.model_hparams = ModelConfig[args.data_name]
    args.learning_rate = 1e-3
    args.weight_decay = 1e-4

    # Add data path
    data_source = 'ood' if args.data_name in AdvDiffReactScenarios else 'apebench'
    args.folder_path = Path(f"./data/{data_source}") / args.data_name

    # Set model id
    args.model_id = (
        f"d={args.data_name}_"
        f"ns={args.num_train_samples}_"
        f"nl={args.model_hparams.get('num_layers')}_"
        f"w={args.model_hparams.get('width')}_"
        f"m={args.model_hparams.get('modes')}"
    )

    return args


def train(args: argparse.Namespace):
    """
    Train an FNO model with the specified configuration.

    This function handles the complete training process:
    1. Sets random seed for reproducibility
    2. Loads and prepares the dataset
    3. Initializes the model and trainer
    4. Runs the training loop
    5. Prints final metrics

    Parameters
    ----------
    args : argparse.Namespace
        Training configuration containing:
        - seed: Random seed
        - folder_path: Path to dataset
        - data_name: Dataset name
        - batch_size: Batch size
        - num_train_samples: Number of training samples
        - model_id: Model identifier
        - model_hparams: Model hyperparameters
        - learning_rate: Learning rate
        - weight_decay: Weight decay
        - enable_progress_bar: Whether to show progress bar
        - debug: Whether to enable debug mode
        - check_val_every_n_epoch: Validation frequency
        - wandb: Whether to use Weights & Biases logging
        - num_epochs: Number of training epochs

    Returns
    -------
    None
        Prints training metrics to console

    Notes
    -----
    The function uses the following components:
    - FNO: Fourier Neural Operator model
    - Trainer: Training loop and metric logging
    - DataLoaders: Dataset handling for train/val/test splits
    """
    # Fix random seed
    fix_random_seed(args.seed)

    # load data
    train_loader, val_loader, test_loader = get_data_loaders(
        folder_path=args.folder_path,
        data_name=args.data_name,
        batch_sizes=[args.batch_size] * 3,
        max_num_of_samples=[args.num_train_samples, "all", "all"],
    )

    # set trainer
    trainer = Trainer(
        model_id=args.model_id,
        model_class=FNO,
        model_hparams=args.model_hparams,
        optimizer_hparams={
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        logger_params={
            "project": "luno_train",
            "logger": "wandb" if args.wandb else "csv",
        },
        seed=args.seed,
        enable_progress_bar=args.enable_progress_bar,
        debug=args.debug,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    # train
    metrics = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
    )

    # print metrics
    print(f"Model: {args.model_id} - Num of epochs: {args.num_epochs}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
