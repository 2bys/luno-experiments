"""Input perturbation-based uncertainty quantification methods.

This module provides functionality for implementing uncertainty quantification
methods based on input perturbations. It supports sampling-based approaches
where uncertainty is estimated by perturbing the input data according to a
specified prior distribution.
"""

from luno_experiments.uncertainty.methods._core import UQMethod
from laplax.eval.pushforward import set_get_weight_sample
from laplax.eval.utils import finalize_fns
from loguru import logger 
from functools import partial
import jax.numpy as jnp
import jax
from luno_experiments.uncertainty.methods._sampling import SamplingPredictiveMixin
from luno_experiments.utils import default

def set_input_perturbations_pushforward(
    model_fn,   
    params,
    input_shape,
    prior_arguments,
    num_weight_samples,
    pushforward_fns,
    key,
    **kwargs,
):
    """Set up the input perturbations pushforward computation.
    
    This function creates a probabilistic predictive function that computes
    predictions for perturbed inputs and applies pushforward functions to the results.

    Parameters
    ----------
    model_fn : callable
        The model forward function
    params : dict
        Model parameters
    input_shape : tuple
        Shape of the input data
    prior_arguments : dict
        Dictionary containing prior arguments, must include 'prior_prec'
    num_weight_samples : int
        Number of input perturbation samples to generate
    pushforward_fns : list[callable]
        List of pushforward functions to apply to the results
    key : jax.random.PRNGKey
        Random key for generating perturbations
    **kwargs : dict
        Additional keyword arguments passed to finalize_fns

    Returns
    -------
    callable
        A function that takes input data and returns processed predictions with uncertainties

    Notes
    -----
    The returned function:
    1. Computes the base prediction without perturbations
    2. Generates perturbed inputs using the prior distribution
    3. Computes predictions for all perturbed inputs
    4. Applies pushforward functions to compute additional statistics
    5. Returns both the results and auxiliary information
    """
    # Set sampling in input domain
    prior_prec = prior_arguments["prior_prec"]
    get_weight_samples = set_get_weight_sample(
        key,
        jnp.zeros(input_shape),
        lambda x: prior_prec * x,
        num_weight_samples,
    )

    def prob_predictive(input):
        def compute_pred_ptw(idx):
            return model_fn(
                input=input + get_weight_samples(idx), params=params
            )

        pred = model_fn(input=input, params=params)
        pred_ensemble = jax.lax.map(compute_pred_ptw, jnp.arange(num_weight_samples))
        aux = {"pred_ensemble": pred_ensemble}

        return finalize_fns(
            fns=pushforward_fns,
            results={"map": pred},
            aux=aux,
            **kwargs
        )

    return prob_predictive


class InputPerturbations(SamplingPredictiveMixin, UQMethod):
    """Input perturbation-based uncertainty quantification method.
    
    This class implements uncertainty quantification by sampling from a prior
    distribution over input perturbations. It estimates prediction uncertainty
    by computing predictions for multiple perturbed versions of the input.

    Notes
    -----
    The input perturbations method:
    1. Requires preprocessing to determine input shape
    2. Does not use a model wrapper
    3. Computes uncertainties by sampling input perturbations
    """

    @property
    def requires_preprocessing(self) -> bool:
        """Whether preprocessing is required.
        
        Returns
        -------
        bool
            True, as the method requires preprocessing to determine input shape
        """
        return True
    
    @property
    def requires_calibration(self) -> bool:
        """Whether calibration is required.
        
        Returns
        -------
        bool
            False, as the method does not require calibration
        """
        return True

    @property
    def requires_wrapper(self) -> bool:
        """Whether a model wrapper is required.
        
        Returns
        -------
        bool
            False, as the method does not use a model wrapper
        """
        return False

    def _preprocess(self, train_loader):
        """Preprocess the input data.
        
        This method extracts the input shape from the training data loader
        and marks the method as prepared.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader used to determine input shape
        """
        self._input_shape = next(iter(train_loader))[0].shape[1:]
        self.is_prepared = True

        logger.info(f"Input shape: {self._input_shape}")
        logger.info("Method is prepared.")

    def _init_prob_predictive(self, pushforward_fns: list[callable] | None = None, **kwargs):
        """Initialize the probabilistic predictive function.
        
        This method sets up the input perturbation prediction function with the given
        pushforward functions.

        Parameters
        ----------
        pushforward_fns : list[callable] | None, optional
            List of pushforward functions to apply, by default None
        **kwargs : dict
            Additional keyword arguments passed to set_input_perturbations_pushforward

        Returns
        -------
        callable
            The initialized probabilistic predictive function
        """
        return partial(
            set_input_perturbations_pushforward,
            model_fn=self.model_fn,
            params=self.params,
            input_shape=self._input_shape,
            num_weight_samples=self.num_weight_samples,
            pushforward_fns=default(pushforward_fns, self.pushforward_fns),
            key=self.key,
            **kwargs,
        )