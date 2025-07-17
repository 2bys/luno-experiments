"""FNO model implementation.

This module implements the Fourier Neural Operator (FNO) architecture using Flax.
The FNO is designed for learning mappings between function spaces, particularly
useful for solving partial differential equations.
"""

from flax import nnx
import jax
import jax.numpy as jnp

from luno.models.fno import fno_block
from luno_experiments.nn.padding import pad_signal_with_zeros, unpad_signal


class FNOBlock(nnx.Module):
    """
    A single block of the Fourier Neural Operator.

    This block implements the core FNO operation, which consists of:
    1. A Fourier layer that performs convolution in the frequency domain
    2. A skip connection with a linear transformation

    Parameters
    ----------
    modes : int
        Number of Fourier modes to use in each dimension
    width : int
        Width of the hidden layers
    dims : int
        Number of spatial dimensions
    rngs : nnx.Rngs
        Random number generator state for parameter initialization
    dtype : jnp.dtype, optional
        Data type for the parameters, defaults to float32
    """
    def __init__(
        self,
        modes: int,
        width: int,
        dims: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.width = width
        self.modes = modes
        self.dims = dims
        scale = 1 / (jnp.sqrt(2) * width * width)

        # Initialize complex-valued convolution weights
        self.conv = nnx.Param(
            scale
            * jax.random.normal(
                rngs.next(),
                ((2,) + dims * (modes,) + (width, width)),
                dtype=dtype,
            )
        )
        # Initialize skip connection parameters
        self.skip = {
            "kernel": nnx.Param(
                jax.random.uniform(
                    rngs.next(),
                    (width, width),
                    minval=-jnp.sqrt(1 / width),
                    maxval=jnp.sqrt(1 / width),
                    dtype=dtype,
                )
            ),
            "bias": nnx.Param(
                jax.random.uniform(
                    rngs.next(),
                    (width,),
                    minval=-jnp.sqrt(1 / width),
                    maxval=jnp.sqrt(1 / width),
                    dtype=dtype,
                )
            ),
        }

    def __call__(self, x):
        """
        Apply the FNO block to the input.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (..., *spatial_dims, channels)

        Returns
        -------
        jax.Array
            Output tensor of the same shape as input
        """
        # Apply Fourier convolution
        x, _ = fno_block(
            x,
            self.conv[0] + 1j * self.conv[1],
            self.skip["kernel"],
            self.skip["bias"],
            output_grid_shape=x.shape[-self.dims - 1 : -1], # Fixed output grid shape
        )
        return x


class FNO(nnx.Module):
    """
    Fourier Neural Operator (FNO) model.

    This class implements a complete FNO architecture with multiple FNO blocks,
    input/output projections, and optional padding for non-periodic signals.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes : int
        Number of Fourier modes to use in each dimension
    width : int
        Width of the hidden layers
    num_layers : int
        Number of FNO blocks
    dims : int
        Number of spatial dimensions
    rngs : nnx.Rngs
        Random number generator state for parameter initialization
    padding : int or None, optional
        Padding size for non-periodic signals. If None, no padding is applied.

    Attributes
    ----------
    initial_projection : nnx.Linear
        Linear layer for initial channel projection
    fourier_blocks : list[FNOBlock]
        List of FNO blocks
    activations : list[Callable]
        List of activation functions for each block
    output_projection : list
        List of layers for final projection
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        width: int,
        num_layers: int,
        dims: int,
        rngs: nnx.Rngs,
        padding: int | None = 2,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.dims = dims
        self.padding = padding

        # Initial projection
        self.initial_projection = nnx.Linear(in_channels, width, rngs=rngs)

        # Fourier Blocks
        self.fourier_blocks = [
            FNOBlock(modes, width, dims, rngs) for _ in range(num_layers)
        ]
        self.activations = [nnx.gelu for _ in range(num_layers - 1)]
        self.activations.append(lambda x: x)  # Last layer has no activation

        # Output projection
        self.output_projection = [
            nnx.Linear(width, 2 * width, rngs=rngs),
            nnx.gelu,
            nnx.Linear(2 * width, out_channels, rngs=rngs),
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the FNO model.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (batch, *spatial_dims, time, channels)

        Returns
        -------
        jax.Array
            Output tensor of shape (batch, *spatial_dims, time, out_channels)
        """
        # Flatten temporal and channel dimension
        inp_shape = x.shape  # (..., *spatial, time, channel)
        x = x.reshape(
            *inp_shape[:-2], -1
        )  # (..., time, channel) -> (..., time * channel)

        # Initial projection
        x = self.initial_projection(x)

        # Padding for non-periodic signals (if required by model type)
        x = (
            pad_signal_with_zeros(
                x,
                pad_width=[(self.padding, self.padding)] * self.dims,
                axes=range(-self.dims - 1, -1),
            )
            if self.padding
            else x
        )

        # Apply Fourier Blocks
        for i, (block, activation) in enumerate(
            zip(self.fourier_blocks, self.activations)
        ):
            x = activation(block(x))

        # Unpad the signal (if needed)
        x = unpad_signal(x, pad_width=self.padding, axes=range(-self.dims - 1, -1))

        # Output projection
        for layer in self.output_projection:
            x = layer(x)

        return x.reshape(
            *x.shape[:-1], -1, inp_shape[-1]
        )  # Ensure shape is (batch, *spatial, temporal, channels)
    