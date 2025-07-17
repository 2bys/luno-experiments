"""Padding utilities for the Fourier Neural Operator.

This module provides functions for padding and unpadding signals in the context of
Fourier Neural Operators. These operations are essential for handling non-periodic
signals and maintaining proper dimensions during the Fourier transformations.
"""

import jax.numpy as jnp


def pad_signal_with_zeros(signal, pad_width, axes):
    """
    Pad a signal along specific axes with zeros while preserving other dimensions.

    This function is particularly useful for handling non-periodic signals in
    Fourier Neural Operators by adding zero padding to specific dimensions
    before applying Fourier transformations.

    Parameters
    ----------
    signal : jnp.ndarray
        Input signal array with arbitrary dimensions
    pad_width : int or list[tuple[int, int]]
        Padding widths for specified axes:
        - If int: applies the same padding for all specified axes
        - If list of tuples: specify (before, after) padding for each axis
    axes : list[int]
        Axes along which to apply the zero padding

    Returns
    -------
    jnp.ndarray
        Zero-padded signal with the same number of dimensions as the input

    Raises
    ------
    ValueError
        If the length of pad_width list doesn't match the number of specified axes

    Examples
    --------
    >>> signal = jnp.ones((10, 20, 30))
    >>> padded = pad_signal_with_zeros(signal, pad_width=2, axes=[0, 1])
    >>> padded.shape
    (14, 24, 30)  # Padded by 2 on both sides of axes 0 and 1
    """
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width) for _ in axes]
    elif isinstance(pad_width, list):
        if len(pad_width) != len(axes):
            raise ValueError(
                "pad_width list length must match the number of specified axes."
            )

    # Initialize pad widths for all dimensions as no padding
    full_pad_width = [(0, 0)] * signal.ndim

    # Assign zero padding only to specified axes
    for ax, pw in zip(axes, pad_width):
        full_pad_width[ax] = pw

    # Apply zero padding
    padded_signal = jnp.pad(signal, full_pad_width, mode="constant", constant_values=0)
    return padded_signal


def unpad_signal(signal, pad_width, axes):
    """
    Remove padding from a signal along specific axes.

    This function is the inverse operation of pad_signal_with_zeros, removing
    the specified padding from the signal while preserving other dimensions.
    It's typically used after Fourier transformations to restore the original
    signal dimensions.

    Parameters
    ----------
    signal : jnp.ndarray
        Input padded signal array
    pad_width : int or list[tuple[int, int]]
        Padding widths to remove from specified axes:
        - If int: assumes the same padding for all specified axes
        - If list of tuples: specify (before, after) padding for each axis
    axes : list[int]
        Axes along which to remove the padding

    Returns
    -------
    jnp.ndarray
        Unpadded signal with padding removed from specified axes

    Raises
    ------
    ValueError
        If the length of pad_width list doesn't match the number of specified axes

    Examples
    --------
    >>> padded = jnp.ones((14, 24, 30))  # Padded signal
    >>> unpadded = unpad_signal(padded, pad_width=2, axes=[0, 1])
    >>> unpadded.shape
    (10, 20, 30)  # Removed padding of 2 from both sides of axes 0 and 1
    """
    slices = [slice(None)] * signal.ndim  # Initialize slices for all dimensions

    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width) for _ in axes]
    elif isinstance(pad_width, list):
        if len(pad_width) != len(axes):
            raise ValueError(
                "pad_width list length must match the number of specified axes."
            )

    # Create slices for each axis to remove the corresponding padding
    for ax, (before, after) in zip(axes, pad_width):
        slices[ax] = slice(before, signal.shape[ax] - after)

    # Apply slicing to remove padding
    unpadded_signal = signal[tuple(slices)]
    return unpadded_signal
