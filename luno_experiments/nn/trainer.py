"""Trainer module for training neural networks.

This module provides a comprehensive training framework for neural networks using JAX/Flax.
It includes functionality for model training, evaluation, checkpointing, and logging.

The implementation is based on:
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html
"""

import json
import os
import pickle
import time
from collections import defaultdict
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from PIL import Image
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from tqdm.auto import tqdm

from luno_experiments.plotting.pipeline import plot_prediction_vs_target


class Trainer:
    """
    A trainer class for training and evaluating neural network models.

    This class handles the complete training pipeline including:
    - Model initialization and optimization
    - Training and evaluation loops
    - Checkpointing and model saving
    - Metrics tracking and logging
    - Progress visualization

    Parameters
    ----------
    model_id : str
        Unique identifier for the model
    model_class : nnx.Module
        The model class to instantiate
    model_hparams : dict[str, any]
        Hyperparameters for model initialization
    optimizer_hparams : dict[str, any]
        Hyperparameters for optimizer configuration
    logger_params : dict[str, any]
        Parameters for logger initialization
    seed : int
        Random seed for reproducibility
    enable_progress_bar : bool, optional
        Whether to show progress bars during training
    debug : bool, optional
        Whether to run in debug mode (disables JIT compilation)
    check_val_every_n_epoch : int, optional
        How often to run validation during training
    **kwargs : dict
        Additional keyword arguments

    Attributes
    ----------
    model : nnx.Module
        The instantiated model
    optimizer : nnx.Optimizer
        The optimizer instance
    logger : WandbLogger or CSVLogger
        The logger instance
    metrics : nnx.MultiMetric
        Metrics tracking object
    checkpointer : ocp.Checkpointer
        Checkpoint manager
    """
    def __init__(
        self,
        model_id: str,
        model_class: nnx.Module,
        model_hparams: dict[str, any],
        optimizer_hparams: dict[str, any],
        logger_params: dict[str, any],
        seed: int,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 1,
        **kwargs,
    ):
        # Set paths
        self.model_id = model_id
        seed_id = f"s={seed}"
        self.run_id = f"{model_id}_{seed_id}"
        self.save_dir = Path(os.path.join("models", model_id, seed_id))
        logger_params.update({"base_log_dir": self.save_dir})
        logger_params.update(
            {"run_id": f"{self.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
        )

        # Set model and optimizer hyperparameters
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed

        # Set config
        self.config = {
            "model_id": model_id,
            "model_class": "fno",
            "model_hparams": model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "seed": seed,
        }

        # Set flags
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.check_val_every_n_epoch = check_val_every_n_epoch

        # Set random number generator
        self.rng = nnx.Rngs(seed)

        self.metadata = self.config
        self.checkpoint_path = (self.save_dir / "checkpoint").resolve()

        # Initialize model, train/eval-step and logger
        self.init_model()
        self.init_logger(logger_params)
        self.init_checkpointer()
        self.init_metrics()
        self.create_jitted_functions()

    def set_checkpoint_epochs(self, checkpoint_epochs=None):
        """
        Set the epochs at which to save checkpoints.

        Parameters
        ----------
        checkpoint_epochs : list[int], optional
            List of epochs at which to save checkpoints.
            If None, only saves best model.
        """
        self.checkpoint_epochs = checkpoint_epochs or []

    def init_model(self):
        """Initialize the model with the specified hyperparameters."""
        self.model = self.model_class(**self.model_hparams, rngs=self.rng)

    def init_logger(self, logger_params: dict | None = None):
        """
        Initialize a logger and create a logging directory.

        Parameters
        ----------
        logger_params : dict, optional
            Dictionary containing logger configuration parameters:
            - base_log_dir: Base directory for logs
            - logger: Type of logger ('wandb' or 'csv')
            - project: Project name for wandb
            - run_id: Unique identifier for the run
            - version: Version identifier
        """
        # Determine the logging directory
        base_log_dir = logger_params.get("base_log_dir", "checkpoints/")
        self.log_dir = os.path.join(base_log_dir, "logs")
        print(logger_params.get("project"))
        # Create logger object
        logger_type = logger_params.get("logger", "wandb")
        if logger_type == "wandb":
            self.logger = WandbLogger(
                name=logger_params.get("run_id", None),
                save_dir=self.log_dir,
                project=logger_params.get("project", None),
                id=logger_params.get("run_id", None),
                version=logger_params.get("version", None),
                config=self.config,
            )

            # Log hyperparameters
            self.log_dir = self.logger.save_dir
            if not os.path.isfile(os.path.join(self.log_dir, "hparams.json")):
                os.makedirs(os.path.join(self.log_dir, "metrics/"), exist_ok=True)
                with open(os.path.join(self.log_dir, "hparams.json"), "w") as f:
                    json.dump(self.config, f)
        elif logger_type == "csv":
            self.logger = CSVLogger(
                save_dir=self.log_dir, name=logger_params.get("run_id", None)
            )
        else:
            pass

    def init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        """
        Initialize the optimizer and learning rate scheduler.

        Parameters
        ----------
        num_epochs : int
            Number of epochs the model will be trained for
        num_steps_per_epoch : int
            Number of training steps per epoch
        """
        hparams = copy(self.optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop("optimizer", "adamw")
        if optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        else:
            assert False, f'Unknown optimizer: "{opt_class}"'

        # A cosine decay scheduler is used, but others are also possible
        lr = hparams.pop("lr", 1e-2)
        _ = hparams.pop("warmup", 0)  # TODO: Make this a argument.
        total_number_of_steps = int(num_epochs * num_steps_per_epoch)
        print("Total number of steps:", total_number_of_steps)

        # Initialize learning rate scheduler
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=int(total_number_of_steps * 0.2),
            decay_steps=total_number_of_steps,
            end_value=0.0,  # Recommendation: `apebench` paper.
        )

        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip_by_global_norm(hparams.pop("gradient_clip", 1.00))]
        optimizer = optax.chain(*transf, opt_class(lr_schedule, **hparams))
        self.optimizer = nnx.Optimizer(self.model, optimizer)

    def create_jitted_functions(self):
        """
        Create jitted versions of the training and evaluation functions.

        If self.debug is True, no jitting is applied. This is useful for
        debugging purposes as it allows for easier error tracking.
        """
        train_step, eval_step = self.create_functions()
        if self.debug:  # Skip jitting
            print("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = nnx.jit(train_step)
            self.eval_step = nnx.jit(eval_step)

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Iterator | None = None,
        num_epochs: int = 100,
        num_train_steps: int = 100,
    ):
        """
        Start a training loop for the given number of epochs.

        Parameters
        ----------
        train_loader : Iterator
            Data loader for the training set
        val_loader : Iterator
            Data loader for the validation set
        test_loader : Iterator, optional
            Data loader for the test set. If provided, the best model will be
            evaluated on this set.
        num_epochs : int, optional
            Number of epochs to train for
        num_train_steps : int, optional
            Number of training steps per epoch

        Returns
        -------
        dict
            Dictionary containing the train, validation and test metrics for the
            best model on the validation set
        """
        # Create optimizer and the scheduler for the given number of epochs
        self.num_train_steps = num_train_steps
        self.init_optimizer(num_epochs, num_train_steps)

        # Prepare training loop
        best_eval_metrics = None

        for epoch_idx in self.tracker(range(1, num_epochs + 1), desc="Epochs"):
            train_metrics = self.train_epoch(train_loader)
            self.logger.log_metrics(train_metrics, step=epoch_idx)

            # Save checkpoint if epoch is in checkpoint_epochs
            if (
                hasattr(self, "checkpoint_epochs")
                and epoch_idx in self.checkpoint_epochs
            ):
                checkpoint_path = self.checkpoint_path.parent / f"e={epoch_idx}"
                self.save_model(step=epoch_idx, checkpoint_path=checkpoint_path)

            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix="val/")
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f"eval_epoch_{str(epoch_idx).zfill(3)}", eval_metrics)

                # Save best model
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(step=epoch_idx)
                    self.save_metrics("best_eval", eval_metrics)

        # Test best model if possible
        if test_loader is not None:
            self.model = self.load_model_from_checkpoint(self.checkpoint_path)
            test_metrics = self.eval_model(test_loader, log_prefix="test/")
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics("test", test_metrics)
            best_eval_metrics.update(test_metrics)
            self.plot_results(self.model, test_loader)

        # Close logger
        self.logger.finalize("success")
        return best_eval_metrics

    def train_epoch(self, train_loader: Iterator) -> dict[str, any]:
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader : Iterator
            Data loader for the training set

        Returns
        -------
        dict[str, any]
            Dictionary containing the average training metrics over all batches
        """
        # Train model for one epoch, and log avg loss and accuracy
        start_time = time.time()
        i = 0
        self.metrics.reset()
        for batch in self.tracker(train_loader, desc="Training", leave=False):
            self.train_step(self.model, self.optimizer, batch, self.metrics)
            i += 1
        metrics = {"train/" + k: v.item() for k, v in self.metrics.compute().items()}
        metrics["epoch_time"] = time.time() - start_time
        return metrics

    def eval_model(
        self, data_loader: Iterator, log_prefix: str | None = ""
    ) -> dict[str, any]:
        """
        Evaluate the model on a dataset.

        Parameters
        ----------
        data_loader : Iterator
            Data loader for the dataset to evaluate on
        log_prefix : str, optional
            Prefix to add to all metrics (e.g. 'val/' or 'test/')

        Returns
        -------
        dict[str, any]
            Dictionary containing the evaluation metrics, averaged over all
            data points in the dataset
        """
        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        num_elements = 0
        for batch in data_loader:
            self.eval_step(self.model, batch, self.metrics)
            batch_size = (
                batch[0].shape[0]
                if isinstance(batch, (list, tuple))
                else batch.shape[0]
            )
            num_elements += batch_size
        metrics = {
            (log_prefix + k): v.item() for k, v in self.metrics.compute().items()
        }
        return metrics

    def is_new_model_better(
        self, new_metrics: dict[str, any], old_metrics: dict[str, any]
    ) -> bool:
        """
        Compare two sets of evaluation metrics to decide if the new model is better.

        Parameters
        ----------
        new_metrics : dict[str, any]
            Dictionary of evaluation metrics for the new model
        old_metrics : dict[str, any]
            Dictionary of evaluation metrics for the previously best model

        Returns
        -------
        bool
            True if the new model is better than the old one, False otherwise

        Notes
        -----
        The comparison is based on the following metrics in order of priority:
        1. val_metric (lower is better)
        2. acc (higher is better)
        3. loss (lower is better)
        """
        if old_metrics is None:
            return True
        for key, is_larger in [
            ("val/val_metric", False),
            ("val/acc", True),
            ("val/loss", False),
        ]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f"No known metrics to log on: {new_metrics}"

    def plot_results(self, model: nnx.Module, iterator: Iterator) -> Iterator:
        """
        Generate and save prediction plots for the test set.

        Parameters
        ----------
        model : nnx.Module
            The trained model to use for predictions
        iterator : Iterator
            Data loader containing the test data

        Returns
        -------
        Iterator
            The input iterator
        """
        # Get and evaluate batch
        inputs, targets = next(iter(iterator))
        preds = jax.vmap(model)(inputs)

        buffer = plot_prediction_vs_target(
            preds[0, ..., 0, 0],
            targets[0, ..., 0, 0],
            dims=model.dims,
            title="Test prediction",
            plot=False,
        )
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image("Test result", images=[Image.open(buffer)])
        else:
            os.makedirs(os.path.join(self.log_dir, "metrics/"), exist_ok=True)
            save_path = os.path.join(self.log_dir, "metrics/pred_plot.png")
            with Image.open(buffer) as image:
                image.save(save_path, overwrite=True)

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """
        Wrap an iterator in a progress bar tracker if enabled.

        Parameters
        ----------
        iterator : Iterator
            Iterator to wrap in tqdm
        **kwargs : dict
            Additional arguments to pass to tqdm

        Returns
        -------
        Iterator
            Wrapped iterator if progress bar is enabled, otherwise same iterator
        """
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def init_metrics(self):
        """
        Initialize the metrics tracking object.

        Currently tracks:
        - loss: Average loss over batches
        """
        # Setup global metrics
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

    def save_metrics(self, filename: str, metrics: dict[str, any]):
        """
        Save a dictionary of metrics to a JSON file.

        Parameters
        ----------
        filename : str
            Name of the metrics file without folders and postfix
        metrics : dict[str, any]
            Dictionary of metrics to save
        """
        os.makedirs(os.path.join(self.log_dir, "metrics/"), exist_ok=True)
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def init_checkpointer(self):
        """Initialize the checkpoint manager for saving model states."""
        self.checkpointer = ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "metadata")
        )

    def save_model(self, step: int = 0, checkpoint_path=None):
        """
        Save the current training state.

        Parameters
        ----------
        step : int, optional
            Index of the step to save the model at (e.g. epoch number)
        checkpoint_path : Path, optional
            Custom path for the checkpoint. If None, uses the default path
        """
        save_path = checkpoint_path or self.checkpoint_path
        self.checkpointer.save(
            save_path.resolve(),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(nnx.state(self.model)),
                metadata=ocp.args.JsonSave(self.metadata),
            ),
            force=True,
        )
        # Save model card alongside checkpoint
        if checkpoint_path:
            model_card_path = Path(checkpoint_path).parent / "params"
            model_card_path.mkdir(parents=True, exist_ok=True)
            with open(model_card_path / "model_card.pkl", "wb") as f:
                pickle.dump(
                    {
                        "model_class": self.model_class.__name__,
                        "model_hparams": self.model_hparams,
                        "optimizer_hparams": self.optimizer_hparams,
                        "seed": self.seed,
                        "model_id": self.model_id,
                        "step": step,
                    },
                    f,
                )

    def load_model_from_checkpoint(self, checkpoint_path) -> any:
        """
        Load a model from a checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to the checkpoint directory

        Returns
        -------
        nnx.Module
            The restored model
        """
        # Get abstract reference state
        graph_def, state = nnx.split(self.model)

        # Load checkpointed parameters
        checkpointer = ocp.StandardCheckpointer()
        restored_state = checkpointer.restore(
            (checkpoint_path / "state").resolve(), target=state
        )

        # Restore model by merging
        return nnx.merge(graph_def, restored_state)

    def create_functions(self):
        """
        Create the training and evaluation step functions.

        Returns
        -------
        tuple
            - train_step: Function for performing a single training step
            - eval_step: Function for performing a single evaluation step
        """
        def mse_loss(model, batch):
            inputs, targets = batch
            preds = jax.vmap(model)(inputs)
            return jnp.mean((preds - targets.reshape(*preds.shape)) ** 2)

        def train_step(model, optimizer, batch, metrics):
            # Compute loss and gradients
            grad_fn = nnx.value_and_grad(mse_loss)
            loss, grads = grad_fn(model, batch)
            optimizer.update(grads)
            metrics.update(loss=loss)

        def eval_step(model, batch, metrics):
            loss = mse_loss(model, batch)
            metrics.update(loss=loss)

        return train_step, eval_step