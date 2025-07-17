"""Base classes and core functionality for uncertainty quantification methods.

This module provides the abstract base class and core functionality for implementing
various uncertainty quantification (UQ) methods for neural networks. It supports
different approaches including Laplace approximation, ensemble methods, and more.

Key features:
- Abstract base class for UQ methods
- Common functionality for model preparation and calibration
- Support for checkpointing and model loading
- Flexible interface for probabilistic predictions
"""

import abc
import jax
import jax.numpy as jnp
from pathlib import Path
from flax import nnx
from loguru import logger 
from luno_experiments.nn.wrapper import split_wrapper, FNOWrapper
from laplax.curv.utils import LowRankTerms


class UQMethod(abc.ABC):
    """Base class for uncertainty quantification methods.
    
    This abstract base class defines the interface and common functionality for
    implementing uncertainty quantification methods for neural networks. It handles
    model preparation, calibration, and probabilistic predictions.

    Parameters
    ----------
    model : nnx.Module
        The neural network model to analyze
    load_checkpoints : bool, optional
        Whether to load saved checkpoints, by default True
    load_calibration : bool, optional
        Whether to load calibration data, by default True
    checkpoint_base_path : str | Path | None, optional
        Base path for saving/loading checkpoints, by default None
    dtype : jnp.dtype, optional
        Data type for computations, by default jnp.float64
    seed : int, optional
        Random seed for reproducibility, by default 42
    num_weight_samples : int, optional
        Number of weight samples for sampling-based methods, by default 100
    low_rank_terms : LowRankTerms | None, optional
        Low-rank approximation terms for Laplace methods, by default None
    model_ensemble : list[nnx.Module] | None, optional
        List of models for ensemble methods, by default None
    **kwargs : dict
        Additional keyword arguments for specific methods

    Attributes
    ----------
    dtype : jnp.dtype
        Data type for computations
    seed : int
        Random seed for reproducibility
    key : jax.random.PRNGKey
        JAX random key for sampling
    num_weight_samples : int
        Number of weight samples
    low_rank_terms : LowRankTerms | None
        Low-rank approximation terms
    model_ensemble : list[nnx.Module] | None
        List of models for ensemble
    is_prepared : bool
        Whether the method has been prepared
    is_calibrated : bool
        Whether the method has been calibrated
    checkpoint_base_path : Path
        Base path for checkpoints
    wrapper : FNOWrapper | None
        Model wrapper if required
    model_fn : callable
        Model forward function
    params : dict
        Model parameters
    prior_args : dict | None
        Prior arguments for calibration
    """

    def __init__(
        self,
        model: nnx.Module,
        load_checkpoints: bool = True,
        load_calibration: bool = True,
        checkpoint_base_path: str | Path | None = None,
        dtype: jnp.dtype = jnp.float64,
        seed: int = 42,
        num_weight_samples: int = 200,
        low_rank_terms: LowRankTerms | None = None,
        model_ensemble: list[nnx.Module] | None = None,
        **kwargs,
    ):
        # Set arguments
        self.dtype = dtype

        # Sampling specific arguments
        self.seed = seed
        self.key = jax.random.key(seed)
        self.num_weight_samples = num_weight_samples

        # Laplace specific arguments
        self.low_rank_terms = low_rank_terms

        # Ensemble specific arguments
        self.model_ensemble = model_ensemble

        # Initialize method
        self.is_prepared = not self.requires_preprocessing
        self.is_calibrated = False

        # Setup checkpointing paths
        self.checkpoint_base_path = Path(checkpoint_base_path)

        # Prepare model
        self._prepare_model(model)
        self._setup_pushforward()

        # Load checkpoints is requested
        if load_checkpoints:
            self.load_all_checkpoints(load_calibration=load_calibration)

    # --------- Properties ---------
    @property
    @abc.abstractmethod
    def requires_preprocessing(self) -> bool:
        """Whether preprocessing is required.
        
        Returns
        -------
        bool
            True if the method requires preprocessing, False otherwise
        """
        pass

    @property
    @abc.abstractmethod
    def requires_calibration(self) -> bool:
        """Whether calibration is required.
        
        Returns
        -------
        bool
            True if the method requires calibration, False otherwise
        """
        pass

    @property
    @abc.abstractmethod
    def requires_wrapper(self) -> bool:
        """Whether a model wrapper is required.
        
        Returns
        -------
        bool
            True if the method requires a model wrapper, False otherwise
        """
        pass

    # --------- Setup ---------

    def _prepare_model(self, model: nnx.Module) -> nnx.Module:
        """Prepare the model for uncertainty quantification.
        
        This method handles model preparation, including wrapping if required
        and setting up the model function and parameters.

        Parameters
        ----------
        model : nnx.Module
            The neural network model to prepare

        Returns
        -------
        nnx.Module
            The prepared model
        """
        if self.requires_wrapper:
            self.wrapper = FNOWrapper(model, dtype=self.dtype)
            self.model_fn, self.params = split_wrapper(self.wrapper)
        else:
            graph_def, state = nnx.split(model)
            def model_fn(input, params):
                return nnx.call((graph_def, params))(input)[0]
            self.model_fn = model_fn
            self.params = state

    @abc.abstractmethod
    def _setup_pushforward(self, **kwargs):
        """Set up the pushforward computation.
        
        This method should be implemented by subclasses to set up any
        necessary components for the pushforward computation.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for setup
        """
        pass

    def _preprocess(self, train_loader):
        """Internal preprocessing method.
        
        This method can be overridden by subclasses to implement
        specific preprocessing steps.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        """
        pass

    def preprocess(self, train_loader, **kwargs):
        """Preprocess the data and prepare the method.
        
        This method handles the preprocessing of training data and
        marks the method as prepared.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        **kwargs : dict
            Additional keyword arguments for preprocessing
        """
        self._preprocess(train_loader)
        self.is_prepared = True
        logger.info(f"Method {self.__class__.__name__} prepared.")

    def set_prior_args(self, prior_args: dict):
        """Set prior arguments and mark the method as calibrated.
        
        Parameters
        ----------
        prior_args : dict
            Dictionary of prior arguments for calibration
        """
        self.prior_args = prior_args
        self.is_calibrated = True
        logger.info(f"Method {self.__class__.__name__} calibrated.")

    # --------- Probabilistic predictive ---------
    
    @abc.abstractmethod
    def _init_prob_predictive(self, prior_arguments: dict, **kwargs):
        """Initialize probabilistic predictive function.
        
        This method should be implemented by subclasses to set up
        the probabilistic predictive function.

        Parameters
        ----------
        prior_arguments : dict
            Dictionary of prior arguments
        **kwargs : dict
            Additional keyword arguments for initialization

        Returns
        -------
        callable
            The initialized probabilistic predictive function
        """
        pass

    def prob_predictive(self, **kwargs):
        """Get the probabilistic predictive function.
        
        This method returns the probabilistic predictive function,
        ensuring that the method is prepared and calibrated if required.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for prediction

        Returns
        -------
        callable
            The probabilistic predictive function

        Raises
        ------
        ValueError
            If the method is not prepared or calibrated when required
        """
        if self.requires_preprocessing and not self.is_prepared:
            msg = "Model not prepared. Run prepare() first."
            logger.error(msg)
            raise ValueError(msg)

        if self.requires_calibration and not self.is_calibrated:
            msg = "Method not calibrated. Consider settings prior_arguments first."
            logger.error(msg)
            raise ValueError(msg)

        return self._init_prob_predictive()(
            prior_arguments=self.prior_args, **kwargs
        )
