"""Utility functions for model loading and manipulation.

This module provides functions for loading trained models from checkpoints,
creating model ensembles, and converting model parameters to different data types.
"""

import jax
from flax import nnx
from luno_experiments.nn.model import FNO
from pathlib import Path 
import orbax.checkpoint as ocp


def load_model(
    checkpoint_path: Path,
    model_hparams: dict,
):
    """
    Load a trained model from a checkpoint.

    This function initializes a model with the given hyperparameters and
    restores its parameters from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint directory containing the saved model state
    model_hparams : dict
        Dictionary of hyperparameters used to initialize the model

    Returns
    -------
    FNO
        The restored model with parameters loaded from the checkpoint

    Notes
    -----
    The model is initialized with a fixed random seed (0) to ensure
    consistent parameter initialization before loading the checkpoint.
    """
    # Initialize abstract model
    model = FNO(**model_hparams, rngs=nnx.Rngs(0))
    graph_def, abstract_state = nnx.split(model)

    # Restore model checkpoint
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(
        Path(checkpoint_path).resolve(),
        abstract_state,
    )

    # Merge into model
    model = nnx.merge(graph_def, state_restored)

    return model 


def load_model_ensemble(
    checkpoint_path: Path,
    model_hparams: dict,
    seeds: list[int],
    model_seed: int = 0,
):
    """
    Load an ensemble of models from checkpoints with different random seeds.

    This function loads multiple models that were trained with different random
    seeds but share the same architecture and hyperparameters. The models are
    loaded from checkpoints that follow a naming convention where the seed is
    specified in the path.

    Parameters
    ----------
    checkpoint_path : Path
        Base path to the checkpoint directory containing the saved model states
    model_hparams : dict
        Dictionary of hyperparameters used to initialize the models
    seeds : list[int]
        List of random seeds to load models for
    model_seed : int, optional
        The seed used in the checkpoint path that should be replaced with
        the seeds from the seeds list

    Returns
    -------
    list[FNO]
        List of restored models, one for each seed in the seeds list

    Raises
    ------
    ValueError
        If the model_seed is not found in the checkpoint path

    Notes
    -----
    The checkpoint path should contain the seed in the format 's={seed}'.
    For example: '/path/to/checkpoints/s=0/model.ckpt'
    """
    checkpoint_path = str(checkpoint_path)

    if f"s={model_seed}" not in checkpoint_path:
        msg = (
            f"Model seed {model_seed} not found in "
            f"checkpoint path {checkpoint_path}"
        )
        raise ValueError(msg)
    
    models = []

    for seed in seeds:
        # replace s=model_seed with s=seed
        new_checkpoint_path = checkpoint_path.replace(f"s={model_seed}", f"s={seed}")
        
        # load model
        model = load_model(new_checkpoint_path, model_hparams)
        models.append(model)

    return models


def model_to_dtype(model, dtype):
    """
    Convert all model parameters to a specified data type.

    This function creates a new model with the same architecture but with
    all parameters converted to the specified data type.

    Parameters
    ----------
    model : nnx.Module
        The input model whose parameters should be converted
    dtype : jnp.dtype
        The target data type to convert the parameters to

    Returns
    -------
    nnx.Module
        A new model with parameters converted to the specified data type

    Notes
    -----
    This function is useful for converting models between different
    precision levels (e.g., float32 to float16) or for ensuring
    consistent data types across an ensemble of models.
    """
    graph_def, params = nnx.split(model)
    params = jax.tree.map(lambda x: x.astype(dtype), params)
    return nnx.merge(graph_def, params)
