"""FNO Wrapper for selecting only a subset of weights of the model.

This module provides a wrapper class for FNO models that allows for selective
access to and manipulation of specific model parameters, particularly the last
Fourier block's weights. This is useful for fine-tuning or analyzing specific
parts of the model architecture.
"""

from copy import deepcopy

import jax
import jax.numpy as jnp
from flax import nnx

from luno_experiments.nn.padding import pad_signal_with_zeros, unpad_signal
from luno_experiments.nn.utils import model_to_dtype


class FNOWrapper(nnx.Module):
    """
    A wrapper class for FNO models that provides access to specific model parameters.

    This wrapper allows for selective access to the last Fourier block's parameters
    and provides methods to process inputs through different stages of the model.

    Parameters
    ----------
    model : nnx.Module
        The FNO model to wrap
    dtype : jnp.dtype, optional
        The data type for model parameters, defaults to float32

    Attributes
    ----------
    model : nnx.Module
        The wrapped FNO model
    last_fourier_block : FNOBlock
        A copy of the last Fourier block from the model
    n_data_channels : int
        Number of output channels in the model
    """
    def __init__(self, model, dtype=jnp.float32):
        # Set model and last fourier block
        self.model = model_to_dtype(model, dtype)
        self._initialize_wrapper()

    @property
    def params(self):
        """
        Get the parameters of the last Fourier block.

        Returns
        -------
        tuple
            A tuple containing:
            - R: Complex-valued convolution weights
            - W: Skip connection kernel
            - b: Skip connection bias
        """
        R = self.last_fourier_block.conv[0] + 1j * self.last_fourier_block.conv[1]
        W = self.last_fourier_block.skip["kernel"]
        b = self.last_fourier_block.skip["bias"]
        return R, W, b

    def _initialize_wrapper(self):
        """
        Initialize the wrapper by copying the last Fourier block and setting up
        necessary attributes.
        """
        self.last_fourier_block = deepcopy(self.model.fourier_blocks[-1])
        self.n_data_channels = self.model.out_channels
        # This assumes currently that time_future == 1.

    def _to_dtype(self, dtype=jnp.float32):
        """
        Convert the model parameters to a specified data type.

        Parameters
        ----------
        dtype : jnp.dtype, optional
            The target data type, defaults to float32
        """
        # Adjust model and reinitialize
        self.model = model_to_dtype(self.model, dtype)
        self._initialize_wrapper()

    def _before_last_fourier_layer(self, x: jax.Array) -> jax.Array:
        """
        Process input through all layers before the last Fourier block.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (..., time, channel)

        Returns
        -------
        jax.Array
            Processed tensor ready for the last Fourier block
        """
        # Flatten temporal and channel dimension
        x = x.reshape(
            *x.shape[:-2], -1
        )  # (..., time, channel) -> (..., time * channel)

        # Initial projection
        x = self.model.initial_projection(x)

        # Padding for non-periodic signals (if required by model type)
        x = (
            pad_signal_with_zeros(
                x,
                pad_width=[(self.model.padding, self.model.padding)] * self.model.dims,
                axes=range(-self.model.dims - 1, -1),
            )
            if self.model.padding
            else x
        )

        # Apply Fourier Blocks
        for block, activation in zip(
            self.model.fourier_blocks[:-1], self.model.activations[:-1], strict=True
        ):
            x = activation(block(x))

        return x

    def _last_fourier_layer(self, x: jax.Array):
        """
        Process input through the last Fourier block.

        Parameters
        ----------
        x : jax.Array
            Input tensor from previous layers

        Returns
        -------
        jax.Array
            Processed tensor after the last Fourier block
        """
        x = self.last_fourier_block(x)
        x = self.model.activations[-1](x)
        return x

    def _out_projection(self, x: jax.Array) -> jax.Array:
        """
        Apply the output projection layers to the processed tensor.

        Parameters
        ----------
        x : jax.Array
            Input tensor from the last Fourier block

        Returns
        -------
        jax.Array
            Processed tensor after output projection
        """
        # Output projection
        for layer in self.model.output_projection:
            x = layer(x)
        return x

    def _final_reshape(self, x: jax.Array) -> jax.Array:
        """
        Reshape the output tensor to the expected format.

        Parameters
        ----------
        x : jax.Array
            Input tensor before final reshape

        Returns
        -------
        jax.Array
            Reshaped tensor of shape (batch, *spatial, temporal, channels)
        """
        return x.reshape(
            *x.shape[:-1], -1, self.n_data_channels
        )  # Ensure shape is (batch, *spatial, temporal, channels)

    def _after_last_fourier_layer(self, x: jax.Array) -> jax.Array:
        """
        Process input through layers after the last Fourier block.

        Parameters
        ----------
        x : jax.Array
            Input tensor from the last Fourier block

        Returns
        -------
        jax.Array
            Processed tensor after all post-Fourier operations
        """
        # Unpad the signal (if needed)
        x = unpad_signal(
            x,
            pad_width=self.model.padding,
            axes=range(-self.model.dims - 1, -1),
        )

        return self._out_projection(x)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Process input through the complete model pipeline.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (batch, *spatial, temporal, channels)

        Returns
        -------
        jax.Array
            Output tensor of shape (batch, *spatial, temporal, channels)
        """
        x = self._before_last_fourier_layer(x)
        x = self._last_fourier_layer(x)
        x = self._after_last_fourier_layer(x)
        return self._final_reshape(x)


def split_wrapper(
    wrapper: FNOWrapper,
    dtype=jnp.float32,
):
    """
    Split a FNOWrapper into a model function and its parameters.

    This function separates the wrapper's parameters into relevant (last Fourier block)
    and remaining parameters, and creates a function that can be called with
    the separated parameters.

    Parameters
    ----------
    wrapper : FNOWrapper
        The wrapper instance to split
    dtype : jnp.dtype, optional
        The data type for the parameters, defaults to float32

    Returns
    -------
    tuple
        A tuple containing:
        - model_fn: A function that takes input and parameters and returns model output
        - relevant_params: The parameters of the last Fourier block
    """
    # Split parameters
    graph_def, *relevant_params, remaining_params = nnx.split(
        wrapper,
        *[
            lambda n, p: "last_fourier_block" in n and "conv" in n,
            lambda n, p: "last_fourier_block" in n and "skip" in n and "kernel" in n,
            lambda n, p: "last_fourier_block" in n and "skip" in n and "bias" in n,
        ],
        ...,
    )

    # Adjust dtype
    remaining_params = jax.tree.map(dtype, remaining_params)
    relevant_params = jax.tree.map(dtype, relevant_params)

    # Set model_fn: x, p -> model(x, p)
    def model_fn(input: jax.Array, params: tuple):
        return nnx.call(
            (graph_def, *params, remaining_params),
        )(input)[0]

    return model_fn, relevant_params
