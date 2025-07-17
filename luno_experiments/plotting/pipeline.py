"""Plotting utilities for model predictions and uncertainty analysis.

This module provides functions for visualizing model predictions against targets
and analyzing prediction uncertainty through ensemble methods. It supports both
1D and 2D data visualization with customizable plotting options.
"""

import matplotlib.pyplot as plt
import io
import jax.numpy as jnp


def plot_prediction_vs_target(
    prediction, target, dims: int = 2, title=None, cmap="viridis", plot: bool = False
):
    """
    Create a side-by-side comparison plot of model predictions and targets.

    This function generates visualizations comparing model predictions with ground
    truth targets, supporting both 1D and 2D data formats. For 1D data, it creates
    a line plot, while for 2D data, it generates heatmaps with colorbars.

    Parameters
    ----------
    prediction : array-like
        Model predictions, can be 1D or 2D array
    target : array-like
        Ground truth values, must match the shape of prediction
    dims : int, optional
        Dimensionality of the data (1 or 2), by default 2
    title : str, optional
        Title for the plot, by default None
    cmap : str, optional
        Colormap for 2D visualization, by default "viridis"
    plot : bool, optional
        If True, displays the plot. If False, returns a buffer with the image,
        by default False

    Returns
    -------
    io.BytesIO or None
        If plot=False, returns a BytesIO buffer containing the image data.
        If plot=True, returns None and displays the plot.

    Raises
    ------
    ValueError
        If dims is not 1 or 2
    """
    if dims == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.plot(target, label="Target", color="black")
        axes.plot(prediction, label="Prediction", color="blue")
        axes.set_title("Target vs Prediction")
        axes.set_xlabel("Index")
        axes.set_ylabel("Value")
        axes.legend()

    elif dims == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(target, cmap=cmap, aspect="auto")
        axes[0].set_title("Target")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(prediction, cmap=cmap, aspect="auto")
        axes[1].set_title("Prediction")
        fig.colorbar(im1, ax=axes[1])

    else:
        raise ValueError("dims must be 1, or 2.")

    plt.tight_layout()

    if plot:
        plt.show()
    else:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close(fig)
        return buffer


def plot_uncertainty_vs_target(
    predictions,
    title: str = "Test",
    dims: int = 1,
    plot: bool = False,
    cmap: str = "viridis",
    std_scale: float = 1.96,
):
    """
    Create comprehensive uncertainty visualization plots.

    This function generates detailed visualizations of model predictions, including
    ensemble samples, mean predictions, standard deviations, and errors. For 1D data,
    it creates line plots with uncertainty bands, while for 2D data, it generates
    a grid of heatmaps showing different aspects of the predictions.

    Parameters
    ----------
    predictions : dict
        Dictionary containing prediction data with the following keys:
        - 'pred': Main prediction array
        - 'samples': Ensemble samples array
        - 'target': Ground truth values
        - 'pred_std': Standard deviation of predictions
        - 'pred_mean': Mean of ensemble predictions
    title : str, optional
        Title for the plot, by default "Test"
    dims : int, optional
        Dimensionality of the data (1 or 2), by default 1
    plot : bool, optional
        If True, displays the plot. If False, returns a buffer with the image,
        by default False
    cmap : str, optional
        Colormap for 2D visualization, by default "viridis"
    std_scale : float, optional
        Scaling factor for standard deviation bands (typically 1.96 for 95% confidence),
        by default 1.96

    Returns
    -------
    io.BytesIO or None
        If plot=False, returns a BytesIO buffer containing the image data.
        If plot=True, returns None and displays the plot.

    Notes
    -----
    For 1D data, the plot shows:
    - Target values
    - Main prediction
    - Ensemble mean
    - Uncertainty bands
    - Individual ensemble samples
    - Error analysis

    For 2D data, the plot shows:
    - Target heatmap
    - Prediction heatmap
    - Ensemble mean heatmap
    - Standard deviation heatmap
    - Individual ensemble samples
    - Error heatmap
    """
    # Extract elements from predictions dictionary
    pred = predictions["pred"][..., 0, 0]
    ensemble = predictions["samples"][..., 0, 0]
    target = predictions["target"][..., 0, 0]
    std = predictions["pred_std"][..., 0, 0]
    mean = predictions["pred_mean"][..., 0, 0]
    error = target - pred

    if dims == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Uncertainty ensemble
        axes[0].plot(target, label="Target", color="black", linewidth=2)
        axes[0].plot(pred, label="Pred", color="orange")
        axes[0].plot(mean, label="Mean", color="blue")
        axes[0].fill_between(
            jnp.arange(pred.shape[0]),
            mean - std_scale * std,
            mean + std_scale * std,
            alpha=0.3,
            color="blue",
        )
        axes[0].plot(ensemble[0], label="Sample 1", color="blue", alpha=0.3)
        axes[0].plot(ensemble[1], label="Sample 2", color="blue", alpha=0.3)
        axes[0].plot(ensemble[2], label="Sample 3", color="blue", alpha=0.3)
        plt.legend()
        axes[0].set_title(title)
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Value")

        # Centered uncertainty
        axes[1].plot(target - pred, label="Error", color="black", linewidth=2)
        axes[1].fill_between(
            jnp.arange(pred.shape[0]),
            -std_scale * std,
            +std_scale * std,
            alpha=0.3,
            color="blue",
        )
        axes[1].plot(target - ensemble[0], label="Sample 1", color="blue", alpha=0.3)
        axes[1].plot(target - ensemble[1], label="Sample 2", color="blue", alpha=0.3)
        axes[1].plot(target - ensemble[2], label="Sample 3", color="blue", alpha=0.3)
        plt.legend()
        axes[1].set_title(title)
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Value")

    elif dims == 2:
        # 2D Plotting
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Shared color scale for mean, pred, target, samples
        shared_min_1 = min(target.min(), pred.min(), mean.min(), ensemble.min())
        shared_max_1 = max(target.max(), pred.max(), mean.max(), ensemble.max())

        # Shared color scale for std and error
        shared_min_2 = jnp.abs(error).min()
        shared_max_2 = jnp.abs(error).max()

        # Top row
        im0 = axes[0, 0].imshow(
            target, cmap=cmap, vmin=shared_min_1, vmax=shared_max_1, aspect="auto"
        )
        axes[0, 0].set_title("Target")
        fig.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(
            pred, cmap=cmap, vmin=shared_min_1, vmax=shared_max_1, aspect="auto"
        )
        axes[0, 1].set_title("Prediction")
        fig.colorbar(im1, ax=axes[0, 1])

        im2 = axes[0, 2].imshow(
            mean, cmap=cmap, vmin=shared_min_1, vmax=shared_max_1, aspect="auto"
        )
        axes[0, 2].set_title("Prediction Mean")
        fig.colorbar(im2, ax=axes[0, 2])

        im3 = axes[0, 3].imshow(
            std, cmap=cmap, vmin=shared_min_2, vmax=shared_max_2, aspect="auto"
        )
        axes[0, 3].set_title("Prediction Std Dev")
        fig.colorbar(im3, ax=axes[0, 3])

        # Bottom row
        im4 = axes[1, 0].imshow(
            ensemble[0], cmap=cmap, vmin=shared_min_1, vmax=shared_max_1, aspect="auto"
        )
        axes[1, 0].set_title("Sample 1")
        fig.colorbar(im4, ax=axes[1, 0])

        im5 = axes[1, 1].imshow(
            ensemble[1], cmap=cmap, vmin=shared_min_1, vmax=shared_max_1, aspect="auto"
        )
        axes[1, 1].set_title("Sample 2")
        fig.colorbar(im5, ax=axes[1, 1])

        im6 = axes[1, 2].imshow(
            ensemble[2], cmap=cmap, vmin=shared_min_1, vmax=shared_max_1, aspect="auto"
        )
        axes[1, 2].set_title("Sample 3")
        fig.colorbar(im6, ax=axes[1, 2])

        im7 = axes[1, 3].imshow(
            jnp.abs(error),
            cmap=cmap,
            vmin=shared_min_2,
            vmax=shared_max_2,
            aspect="auto",
        )
        axes[1, 3].set_title("Error (Target - Pred)")
        fig.colorbar(im7, ax=axes[1, 3])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    if plot:
        plt.show()
    else:
        # Save the figure to an in-memory buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close(fig)
        return buffer
