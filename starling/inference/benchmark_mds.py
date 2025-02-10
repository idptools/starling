import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from IPython import embed
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from tqdm import trange

from starling import configs
from starling.frontend.ensemble_generation import generate
from starling.inference.evaluate_vae import get_errors
from starling.structure.coordinates import (
    create_ca_topology_from_coords,
    distance_matrix_to_3d_structure_gd,
    distance_matrix_to_3d_structure_mds,
    distance_matrix_to_3d_structure_torch_mds,
)
from starling.utilities import get_data


def visualize_comparison(original_dms, scipy_coords, torch_coords):
    """
    Visualization function with color scaling based on min/max errors for both methods.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))

    # Compute all differences first for consistent scaling
    all_diffs = []
    for i in range(3):
        scipy_dm = squareform(pdist(scipy_coords[i]))
        torch_dm = squareform(pdist(torch_coords[i]))

        scipy_diff = original_dms[i] - scipy_dm
        torch_diff = original_dms[i] - torch_dm

        all_diffs.extend([scipy_diff, torch_diff])

    vmin = min(diff.min() for diff in all_diffs)
    vmax = max(diff.max() for diff in all_diffs)
    abs_max = max(abs(vmin), abs(vmax))

    # Define modern colormap
    cmap = "viridis"

    for i in range(3):
        scipy_dm = squareform(pdist(scipy_coords[i]))
        torch_dm = squareform(pdist(torch_coords[i]))

        scipy_diff = original_dms[i] - scipy_dm
        torch_diff = original_dms[i] - torch_dm

        # Plot with improved aesthetics
        sns.heatmap(
            scipy_diff,
            ax=axes[0, i],
            cmap=cmap,
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            cbar_kws={"shrink": 0.8},
            square=True,
        )
        axes[0, i].set_title(
            f"Scipy MDS Conformer {i + 1}\nRMSE: {np.sqrt(np.mean(scipy_diff**2)):.2e}",
            pad=10,
            fontsize=9,
        )

        sns.heatmap(
            torch_diff,
            ax=axes[1, i],
            cmap=cmap,
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            cbar_kws={"shrink": 0.8},
            square=True,
        )
        axes[1, i].set_title(
            f"Torch SMACOF Conformer {i + 1}\nRMSE: {np.sqrt(np.mean(torch_diff**2)):.2e}",
            pad=10,
            fontsize=9,
        )

        # Remove tick labels for cleaner look
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.tight_layout()
    return fig


def plot_error_distributions(original_dms, scipy_coords, torch_coords):
    """
    Create a histogram comparing error distributions for both methods.
    """
    scipy_errors = []
    torch_errors = []

    for i in range(len(original_dms)):
        scipy_dm = squareform(pdist(scipy_coords[i]))
        torch_dm = squareform(pdist(torch_coords[i]))

        scipy_diff = (original_dms[i] - scipy_dm).flatten()
        torch_diff = (original_dms[i] - torch_dm).flatten()

        scipy_errors.extend(scipy_diff)
        torch_errors.extend(torch_diff)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Modern blue and green
    colors = ["#3498db", "#2ecc71"]

    # Plot histograms with improved aesthetics
    ax.hist(
        scipy_errors,
        bins=50,
        alpha=0.6,
        label="Scipy MDS",
        density=True,
        color=colors[0],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        torch_errors,
        bins=50,
        alpha=0.6,
        label="Torch SMACOF",
        density=True,
        color=colors[1],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Mean Error (Å)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Distribution of Errors", fontsize=12, pad=15)

    # Clean up the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # Add statistics to legend with cleaner formatting
    ax.legend(
        [
            f"Scipy MDS (μ={np.mean(scipy_errors):.2e})",
            f"Torch SMACOF (μ={np.mean(torch_errors):.2e})",
        ],
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )

    return fig


def benchmark_methods(
    dms, n_iter=100, tol=1e-4, n_repeats=3, batch_size=None, verbose=False
):
    """Benchmark methods with batch processing support"""
    scipy_times = []
    torch_times = []

    for _ in range(n_repeats):
        # Time torch SMACOF with batch processing
        start_time = time.perf_counter()
        torch_coords, _ = distance_matrix_to_3d_structure_torch_mds(
            dms,
            # batch=100,
            n_iter=n_iter,
            tol=tol,
        )
        if verbose:
            print("torch", time.perf_counter() - start_time)
        torch_times.append(time.perf_counter() - start_time)

        # Time Scipy MDS
        start_time = time.perf_counter()
        scipy_mds_coords = [distance_matrix_to_3d_structure_mds(dm) for dm in dms]
        scipy_times.append(time.perf_counter() - start_time)

        if verbose:
            print("scipy", scipy_times[-1])

    return {
        "scipy_times": np.array(scipy_times),
        "torch_times": np.array(torch_times),
        "scipy_coords": scipy_mds_coords,
        "torch_coords": torch_coords,
    }


def plot_timing_comparison(timing_results, n_conformers):
    """
    Create a violin plot comparing the timing distributions.

    Args:
        timing_results: Dictionary containing timing measurements
        n_conformers: Number of conformers processed

    Returns:
        matplotlib figure
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Modern blue and green
    colors = ["#3498db", "#2ecc71"]  

    # Create violin plot
    parts = ax1.violinplot(
        [timing_results["scipy_times"], timing_results["torch_times"]],
        showmeans=True
    )

    # Set colors for individual violins
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor(colors[i])
        body.set_alpha(0.7)  # Optional transparency for better visibility

    # Set the mean line color to red
    parts['cmeans'].set_color("#e74c3c")

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Scipy MDS", "Torch SMACOF"])
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Computation Time Distribution", pad=15)

    # Remove top and right spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add mean times as text
    mean_scipy = np.mean(timing_results["scipy_times"])
    mean_torch = np.mean(timing_results["torch_times"])

    speedup_torch = mean_scipy / mean_torch
    text = (
        f"Mean Times:\nScipy: {mean_scipy:.3f}s\ntorch: {mean_torch:.3f}s\n\n"
        f"Speedup (torch): {speedup_torch:.2f}x\n"
        f"Conformers: {n_conformers}"
    )

    ax1.text(
        0.95,
        0.95,
        text,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Bar plot with improved aesthetics
    times_per_conformer = [
        np.mean(timing_results["scipy_times"]) / n_conformers,
        np.mean(timing_results["torch_times"]) / n_conformers,
    ]

    bars = ax2.bar(
        ["Scipy MDS", "Torch SMACOF"],
        times_per_conformer,
        color=colors,
        alpha=0.8,
        width=0.6,
    )

    ax2.set_ylabel("Time per Conformer (seconds)")
    ax2.set_title("Average Computation Time", pad=15)

    # Remove top and right spines
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(False)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def run_benchmark_comparison(dms, n_iter=100, tol=1e-4, n_repeats=5):
    """Run complete benchmark with all visualizations"""
    timing_results = benchmark_methods(dms, n_iter=n_iter, tol=tol, n_repeats=n_repeats)

    # Create all visualization figures
    heatmap_fig = visualize_comparison(
        dms[:3],
        timing_results["scipy_coords"][:3],
        timing_results["torch_coords"][:3],
    )

    dist_fig = plot_error_distributions(
        dms,
        timing_results["scipy_coords"],
        timing_results["torch_coords"],
    )

    timing_fig = plot_timing_comparison(timing_results, len(dms))

    # Save figures
    heatmap_fig.savefig(
        "mds_comparison_sample_heatmaps.pdf", dpi=300, bbox_inches="tight"
    )
    dist_fig.savefig("mds_mean_error_distribution.pdf", dpi=300, bbox_inches="tight")
    timing_fig.savefig("mds_timing_comparison.pdf", dpi=300, bbox_inches="tight")

    return heatmap_fig, dist_fig, timing_fig, timing_results


if __name__ == "__main__":
    # shape (200, 384, 384)
    ens = generate("PKGS"+"PLKE"*95)
    dms = ens["sequence_1"].distance_maps()

    heatmap_fig, dist_fig, timing_fig, timing_results = run_benchmark_comparison(
        dms, n_iter=300, tol=1e-4, n_repeats=5
    )
