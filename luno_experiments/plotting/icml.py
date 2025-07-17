"""This script includes plotting functions for ICML 2025 submission."""

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tueplots.constants.color.rgb import tue_blue, tue_gold, tue_red, tue_dark

# Color options
#tue_dark = np.array([0.1, 0.1, 0.1])
#dark = np.array([51.0, 51.0, 51.0]) / 255.0
#my_lblue = np.array([236, 231, 242]) / 255.0
red = tue_red
gold = tue_gold #np.array([174.0, 159.0, 109.0]) / 255.0
lgold = np.array([1, 1, 1]) - 0.5 * (np.array([1, 1, 1]) - gold)
blue = tue_blue #np.array([0.0, 0.412, 0.667])
dark = tue_dark

def plot_single_method_uncertainty_1d_w_eigenfunctions(
    data, 
    ax, 
    method_name, 
    n_samples=4, 
    first_plot=False,
    **kwargs
):
    # Get all arguments
    sample_linewidth = kwargs.get("sample_linewidth", 0.7)
    sample_color = kwargs.get("sample_color", gold)
    mean_color = kwargs.get("mean_color", blue)
    mean_linewidth = kwargs.get("mean_linewidth", 1.5)
    std_alpha = kwargs.get("std_alpha", 0.4)
    std_color = kwargs.get("std_color", blue)
    target_alpha = kwargs.get("target_alpha", 0.8)
    target_color = kwargs.get("target_color", red)
    row_1_ylim = kwargs.get("row_1_ylim", (-0.65, 0.65))
    row_1_yticks = kwargs.get("row_1_yticks", [-0.5, 0.0, 0.5])
    row_1_yticklabels = kwargs.get("row_1_yticklabels", [r"$-0.5$", r"$0.0$", r"$0.5$"])

    del kwargs
    if data is None:
        print(f"Warning: {method_name} data is None, skipping plot.")
        ax.set_title(f"{method_name} (No Data)")
        return

    # Extract & squeeze data
    ensemble = np.squeeze(data["samples"])[:n_samples, :].T  # (256, n_samples)
    mean_data = np.squeeze(data["pred_mean"])  # (256,)
    std_data = np.squeeze(data["pred_std"])  # (256,)
    target_data = np.squeeze(data["target"])  # (256,)

    # X-axis values
    x = np.arange(mean_data.size)

    # Plot ensemble samples
    # Create empty lists to store handles and labels
    handles, labels = [], []
    
    # Plot ensemble samples
    for i in range(ensemble.shape[1]):
        line = ax.plot(x, ensemble[:, i], "-", color=sample_color, linewidth=sample_linewidth)[0]
        if i == 0:
            handles.append(line)
            labels.append("Samples")

    # Plot mean prediction
    line = ax.plot(x, mean_data, color=mean_color, linewidth=mean_linewidth)[0]
    handles.append(line)
    labels.append("Mean")

    # Plot uncertainty (±1.96 std for 95% confidence interval)
    fill = ax.fill_between(
        x,
        mean_data - 1.96 * std_data,
        mean_data + 1.96 * std_data,
        color=std_color,
        alpha=std_alpha,
    )
    handles.append(fill)
    labels.append(rf"$\pm 1.96 \sigma$")

    # Plot target values
    line = ax.plot(x, target_data, color=target_color, alpha=target_alpha)[0]
    handles.append(line)
    labels.append("Target")

    # Set title & legend
    ax.set_title(method_name)

    # # Add legend
    # if first_plot:
    #     # Specify desired order: Target, Mean, Samples, Uncertainty
    #     order = [3, 1, 0, 2]  # indices corresponding to the desired order
    #     ax.legend([handles[i] for i in order], [labels[i] for i in order])
    

    ax.set_xticks([])  # Remove x-ticks for clean paper-style look
    if method_name != "Input Perturbations":
        ax.set_ylim(*row_1_ylim)
        ax.set_yticks([])
        ax.set_yticklabels([])

    if method_name == "Input Perturbations":
        ax.set_ylim(*row_1_ylim)
        ax.set_yticks(row_1_yticks)
        ax.set_yticklabels(row_1_yticklabels)


def plot_single_method_covariance(
    data,
    ax,
    method_name,
    **kwargs
):
    """Plot covariance matrix and eigenvectors for a single method."""
    # Get all arguments
    n_eigenvectors = kwargs.get("n_eigenvectors", 3)
    n_samples = kwargs.get("n_samples", 4)
    eigenvector_linewidth = kwargs.get("eigenvector_linewidth", 0.7)
    eigenvector_color = kwargs.get("eigenvector_color", tue_dark)
    sample_linewidth = kwargs.get("sample_linewidth", 0.7)
    sample_color = kwargs.get("sample_color", gold)
    sample_alpha = kwargs.get("sample_alpha", 0.7)
    std_alpha = kwargs.get("std_alpha", 0.3)
    std_color = kwargs.get("std_color", blue)
    row_2_ylim = kwargs.get("row_2_ylim", (-0.05, 0.05))
    row_2_yticks = kwargs.get("row_2_yticks", [-0.04, 0.0, 0.04])
    row_2_yticklabels = kwargs.get("row_2_yticklabels", [r"$-0.04$", r"$0.0$", r"$0.04$"])

    del kwargs

    if data is None:
        print(f"Warning: {method_name} data is None, skipping plot.")
        ax.set_title(f"{method_name} (No Data)")
        return

    # Extract data
    eigenvectors = data["vectors"]
    eigenvalues = data["values"]
    cov_matrix = data["matrix"]
    pred_data = data["data"]

    # Plot eigenvectors
    for j in range(n_eigenvectors):
        ax.plot(
            eigenvectors[:, j] * np.sqrt(eigenvalues[j]),
            linewidth=eigenvector_linewidth,
            color=eigenvector_color,
            label=f"Eigenvector {j + 1}"
        )

    # Plot standard deviation if requested
    # if plot_std:
    mean_data = np.squeeze(pred_data["pred_mean"])
    std_data = np.squeeze(pred_data["pred_std"])
    ax.fill_between(
        np.arange(len(mean_data)),
        np.zeros_like(mean_data) - 1.96 * std_data,
        np.zeros_like(mean_data) + 1.96 * std_data,
        color=std_color,
        alpha=std_alpha
    )

    white_noise = np.random.randn(len(eigenvalues), n_samples)
    samples = (eigenvectors * np.sqrt(eigenvalues)) @ white_noise
    for j in range(n_samples):
        ax.plot(
            samples[:, j],
            linewidth=sample_linewidth,
            color=sample_color,
            label="Samples" if j == 0 else None,
            alpha=sample_alpha
        )

    # # Set title and axes properties
    # ax.set_title(method_name)
    ax.set_xticks([])

    if method_name != "Input Perturbations":
        ax.set_ylim(*row_2_ylim)
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.set_ylim(*row_2_ylim)
        ax.set_yticks(row_2_yticks)
        ax.set_yticklabels(row_2_yticklabels)

    # Add inset axis for covariance matrix
    inset_ax = inset_axes(
        ax,
        width="54%",
        height="54%",
        bbox_to_anchor=(0.33, 0.25, 0.75, 0.75),
        bbox_transform=ax.transAxes,
    )
    inset_ax.imshow(cov_matrix, cmap="binary", interpolation="nearest")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])


import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
from pathlib import Path
NUM_EIGENVECTORS = 4
NUM_EIGENFUNCTIONS = 20
EIGENFUNCTION_SCALING = 5
METHOD_NAMES_PLOT_1 = ["input_perturbations", "ensemble", "laplace", "lugano"]

def compute_covariance_matrix(samples, mean):
    """
    Computes the covariance matrix for the given samples.

    Parameters:
    - samples: A 2D array of shape (num_samples, spatial_dim).
    - mean: A 1D array of shape (spatial_dim,).

    Returns:
    - cov_matrix: A covariance matrix of shape (spatial_dim, spatial_dim).
    """
    centered_samples = samples - mean
    cov_matrix = np.cov(centered_samples, rowvar=False, bias=False)
    return cov_matrix


def plot_figure_1(method_data, name, **kwargs):
    print("PLOT", name)
    with plt.style.context(bundles.icml2024(column="full", nrows=2, ncols=4)):
        fig, axes = plt.subplots(2, 4, sharex=True, sharey=False)
        axes[1,0].set_ylim(-0.04, 0.04)
        for i, method in enumerate(METHOD_NAMES_PLOT_1):
            # 1. Predictions plot
            cov_matrix = method_data[method]["cov_matrix"]
            method_name = method_data[method]["name"]  # Changed variable name to avoid shadowing
            E, U = np.linalg.eigh(cov_matrix)

            # Compute eigenfunctions 
            plot_single_method_uncertainty_1d_w_eigenfunctions(
                data=method_data[method]["data"],
                ax=axes[0, i],
                method_name=f"{method_name}",
                n_samples=4,
                first_plot=i == 0,
                **kwargs.copy()
            )

            eigendata = {
                "data": method_data[method]["data"],
                "matrix": cov_matrix,
                "values": E[-NUM_EIGENVECTORS:],
                "vectors": U[:, -NUM_EIGENVECTORS:],
            }

            plot_single_method_covariance(
                data=eigendata,
                ax=axes[1, i],
                method_name=f"{method_name}",
                num_eigenvectors=NUM_EIGENVECTORS,
                custom_palette=[gold, lgold],
                **kwargs
            )
        
        # Create the figures directory if it doesn't exist
        Path("./figures/figure1").mkdir(parents=True, exist_ok=True)
        
        # Save the figure before showing it
        fig.savefig(f"./figures/figure1/figure_1_{name}.pdf", bbox_inches="tight", dpi=300)
        plt.show()

from pathlib import Path
import pickle
from lugano_experiments.plotting import compute_covariance_matrix


method_of_interest = {
    "weight_perturbations": "Sample-Iso",
    "lugano_prior": r"\textsc{Luno}-Iso",
    # "swag": "SWAG-Sample",
    # "lugano_swag": "SWAG-Luno",
    "lugano": r"\textsc{Luno}-LA",
    "laplace": r"Sample-LA",
    "input_perturbations": "Input Perturbations",
    "ensemble": "Ensemble",
}

def load_method_data(folder_path, method_names, method_names_map):
    """Load data for all methods from a given folder path.
    
    Args:
        folder_path (Path): Path to folder containing method data
        method_names (list): List of method names to load
        method_names_map (dict): Mapping from method names to display names
    
    Returns:
        dict: Dictionary containing loaded data for each method
    """
    method_data = {}
    for tag in method_names:
        method_data[tag] = {}
        try:
            with open(folder_path / f"{tag}/samples.pkl", 'rb') as f:
                data = pickle.load(f)
                method_data[tag]["data"] = {
                    "samples": data["samples"][0, ..., 0, 0],
                    "pred_mean": data["pred_mean"][0, ..., 0, 0], 
                    "pred_std": data["pred_std"][0, ..., 0, 0],
                    "target": data["target"][0, ..., 0, 0],
                }
                method_data[tag]["name"] = method_names_map[tag]

                if tag == "ensemble":
                    method_data[tag]["cov_matrix"] = compute_covariance_matrix(
                        method_data[tag]["data"]["samples"],
                        method_data[tag]["data"]["pred_mean"]
                    )
                else:
                    method_data[tag]["cov_matrix"] = data["pred_cov_dense"][0]

        except FileNotFoundError:
            print(f"Warning: Could not find samples for {tag}")
            method_data[tag] = None
            
    return method_data

# Load data for burgers equation
burgers_folder_path = Path(f'../results/paper/d=diff_burgers_1_ns=all_nr=1_th=10_tf=1_nl=4_w=18_m=12')
ks_cons_folder_path = Path(f'../results/paper/d=diff_ks_cons_1_ns=all_nr=1_th=10_tf=1_nl=4_w=18_m=12')
diff_hyp_diff_folder_path = Path(f'../results/paper/d=diff_hyp_diff_1_ns=all_nr=1_th=10_tf=1_nl=4_w=18_m=12')

burgers_method_data = load_method_data(burgers_folder_path, METHOD_NAMES_PLOT_1, method_of_interest)
ks_cons_method_data = load_method_data(ks_cons_folder_path, METHOD_NAMES_PLOT_1, method_of_interest)
diff_hyp_diff_method_data = load_method_data(diff_hyp_diff_folder_path, METHOD_NAMES_PLOT_1, method_of_interest)

plot_figure_1(
    burgers_method_data, 
    "burgers",
    row_1_ylim=(-0.65, 0.65),
    row_1_yticks=[-0.5, 0.0, 0.5],
    row_1_yticklabels=[r"$-0.5$", r"$0.0$", r"$0.5$"],
)