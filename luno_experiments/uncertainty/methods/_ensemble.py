"""Ensemble-based uncertainty quantification methods.

This module provides functionality for implementing ensemble-based uncertainty
quantification methods. It supports model ensembles and provides utilities for
computing ensemble predictions and uncertainties.
"""

import jax
import jax.numpy as jnp
from luno_experiments.uncertainty.methods._sampling import SamplingPredictiveMixin
from luno_experiments.uncertainty.methods._core import UQMethod
from flax import nnx
from laplax.eval.utils import finalize_fns
from functools import partial

def set_ensemble_pushforward(
    model_fn: callable,
    states: dict,
    pushforward_fns: list[callable],
    prior_arguments: dict,
    **kwargs,
):
    """Set up the ensemble pushforward computation.
    
    This function creates a probabilistic predictive function that computes
    ensemble predictions and applies pushforward functions to the results.

    Parameters
    ----------
    model_fn : callable
        The model forward function
    states : dict
        Dictionary of model states for each ensemble member
    pushforward_fns : list[callable]
        List of pushforward functions to apply to the results
    prior_arguments : dict
        Dictionary of prior arguments (unused in ensemble methods)
    **kwargs : dict
        Additional keyword arguments passed to finalize_fns

    Returns
    -------
    callable
        A function that takes input data and returns processed ensemble predictions

    Notes
    -----
    The returned function:
    1. Computes predictions for all ensemble members
    2. Takes the mean prediction as the MAP estimate
    3. Applies pushforward functions to compute additional statistics
    4. Returns both the results and auxiliary information
    """
    def prob_predictive(input):
        # Get all predictions
        pred_ensemble = jax.vmap(model_fn, in_axes=(None, 0))(input, states)
        pred = jnp.mean(pred_ensemble, axis=0)
        
        # Set results and aux
        results = {"map": pred}
        aux = {
            "model_fn": model_fn,
            "pred_ensemble": pred_ensemble,
        }
        return finalize_fns(
            fns=pushforward_fns,
            results=results,
            aux=aux,
            **kwargs,
        )
    return prob_predictive


class Ensemble(SamplingPredictiveMixin, UQMethod):
    """Ensemble-based uncertainty quantification method.
    
    This class implements uncertainty quantification using model ensembles.
    It combines predictions from multiple model instances to estimate
    prediction uncertainty.

    Notes
    -----
    The ensemble method:
    1. Requires preprocessing to set up ensemble states
    2. Does not require calibration
    3. Does not use a model wrapper
    4. Computes predictions by averaging over ensemble members
    """

    @property
    def requires_preprocessing(self) -> bool:
        """Whether preprocessing is required.
        
        Returns
        -------
        bool
            True, as ensemble methods require preprocessing to set up states
        """
        return True
    
    @property
    def requires_calibration(self) -> bool:
        """Whether calibration is required.
        
        Returns
        -------
        bool
            False, as ensemble methods do not require calibration
        """
        return False
    
    @property
    def requires_wrapper(self) -> bool:
        """Whether a model wrapper is required.
        
        Returns
        -------
        bool
            False, as ensemble methods do not use a model wrapper
        """
        return False

    @property
    def prior_args(self):
        """Get prior arguments.
        
        Returns
        -------
        None
            Ensemble methods do not use prior arguments
        """
        return None
    
    def _preprocess(self, train_loader):
        """Preprocess the ensemble models.
        
        This method extracts and stacks the states from all ensemble members.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader (unused in ensemble preprocessing)
        """
        del train_loader

        # Get model graph and state
        states = []
        for model in self.model_ensemble:
            graph_def, state = nnx.split(model)
            states.append(state)

        self.states = jax.tree.map(lambda *arrs: jnp.stack(arrs), *states)
        self.is_prepared = True

    def _init_prob_predictive(self, pushforward_fns: list[callable] | None = None, **kwargs):
        """Initialize the probabilistic predictive function.
        
        This method sets up the ensemble prediction function with the given
        pushforward functions.

        Parameters
        ----------
        pushforward_fns : list[callable] | None, optional
            List of pushforward functions to apply, by default None
        **kwargs : dict
            Additional keyword arguments passed to set_ensemble_pushforward

        Returns
        -------
        callable
            The initialized probabilistic predictive function
        """
        return partial(
            set_ensemble_pushforward,
            model_fn=self.model_fn,
            states=self.states,
            pushforward_fns=pushforward_fns or self.pushforward_fns,
            **kwargs,
        )
