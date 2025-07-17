"""Isotropic Gaussian weight space uncertainty.

This module includes both sampling-based (SAMPLE_ISO) and LUNO-based (LUNO_ISO) isotropic Gaussian
weight space uncertainty methods.
"""

import jax
from laplax.curv.cov import Posterior
from luno_experiments.uncertainty.methods._core import UQMethod
from luno_experiments.uncertainty.methods._sampling import SamplingPredictiveMixin
from luno_experiments.uncertainty.methods._luno import LUNOMixin
import jax.numpy as jnp
from laplax.util.tree import get_size
from laplax.curv.utils import LowRankTerms
from luno_experiments.uncertainty.methods._luno import create_luno_posterior


class SAMPLE_ISO(SamplingPredictiveMixin, UQMethod):
    """Sampling-based isotropic uncertainty quantification method.
    
    This class implements uncertainty quantification using isotropic Gaussian weight
    space uncertainty and sampling-based pushforward.

    Notes
    -----
    The SAMPLE_ISO method:
    1. Does not require preprocessing
    2. Requires calibration for precision
    3. Requires a model wrapper
    """

    @property
    def requires_preprocessing(self) -> bool:
        """Whether preprocessing is required.
        
        Returns
        -------
        bool
            False, as the method does not require preprocessing
        """
        return False
    
    @property
    def requires_calibration(self) -> bool:
        """Whether calibration is required.
        
        Returns
        -------
        bool
            True, as the method requires calibration for prior precision
        """
        return True

    @property    
    def requires_wrapper(self) -> bool:
        """Whether a model wrapper is required.
        
        Returns
        -------
        bool
            True, as the method requires a model wrapper
        """
        return True

    @staticmethod
    def weight_space_covariance_fn(
        prior_arguments: dict, loss_scaling_factor: float, **kwargs
    ):
        """Create a weight space covariance function.
        
        This method creates functions for computing mean and covariance
        in weight space using an isotropic prior.

        Parameters
        ----------
        prior_arguments : dict
            Dictionary containing prior arguments, must include 'prior_prec'
        loss_scaling_factor : float
            Scaling factor for the loss function
        **kwargs : dict
            Additional keyword arguments (unused)

        Returns
        -------
        Posterior
            A Posterior object containing the mean and covariance functions

        Notes
        -----
        The covariance function:
        1. Uses isotropic prior precision for scaling
        2. Computes mean by scaling with prior precision
        3. Computes covariance by squaring prior precision
        """
        del kwargs

        def scale_mv_from_state(state):
            prior_prec = state["prior_prec"]
            return lambda tree: jax.tree.map(lambda x: prior_prec * x, tree)

        def cov_mv_from_state(state):
            prior_prec = state["prior_prec"]
            return lambda tree: jax.tree.map(lambda x: prior_prec**2 * x, tree)

        return Posterior(
            state=prior_arguments,
            scale_mv=scale_mv_from_state,
            cov_mv=cov_mv_from_state,
        )
    

class LUNO_ISO(LUNOMixin, UQMethod):
    """LUNO-based isotropic Gaussian weight space uncertainty.
    
    This class implements uncertainty quantification using isotropic Gaussian priors
    and the LUNO pushforward approach.

    Notes
    -----
    The LUNO_ISO method:
    1. Requires preprocessing to set up low-rank terms
    2. Requires calibration for prior precision
    3. Requires a model wrapper
    """

    @property
    def requires_preprocessing(self) -> bool:
        """Whether preprocessing is required.
        
        Returns
        -------
        bool
            True, as the method requires preprocessing to set up low-rank terms
        """
        return True
    
    @property
    def requires_calibration(self) -> bool:
        """Whether calibration is required.
        
        Returns
        -------
        bool
            True, as the method requires calibration for prior precision
        """
        return True
    
    @property
    def requires_wrapper(self) -> bool:
        """Whether a model wrapper is required.
        
        Returns
        -------
        bool
            True, as the method requires a model wrapper
        """
        return True
    
    def _preprocess(self, train_loader):
        """Preprocess the model for LUNO computation.
        
        This method initializes the low-rank terms and sets up the
        weight space covariance function.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader (unused in preprocessing)
        """
        del train_loader

        num_params = get_size(self.params)
        self.low_rank_terms = LowRankTerms(
            U=jnp.zeros((num_params, 1)),
            S=jnp.zeros((1,)),
            scalar=0,
        )

        self.weight_space_covariance_fn = create_luno_posterior(
            self.low_rank_terms, self.wrapper
        )
