"""
Visualization module for clustering and fairness analysis.

Provides plotting functions for:
- Cluster visualization in 2D (via t-SNE or PCA)
- Cluster composition by demographic attributes
- Fairness metrics visualization
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from typing import Optional, Literal, Union
from .clustering import ClusteringResult
from matplotlib.colors import ListedColormap

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
             "#D55E00", "#CC79A7", "#000000", "#795548", "#607D8B"]
okabe_cmap = ListedColormap(OKABE_ITO, name="okabe_ito")
try:
    matplotlib.colormaps.register(cmap=okabe_cmap)
except Exception:
    plt.register_cmap(name="okabe_ito", cmap=okabe_cmap)

plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_ITO)
sns.set_palette(OKABE_ITO)

# NOTE: Use PCA when visualizing KMeans/BisectingKMeans results — preserves Euclidean
# distances (same metric KMeans uses). Use t-SNE for HDBSCAN/DBSCAN — preserves local
# neighborhood structure. Use MDS when clustering used Gower distance — MDS with a
# precomputed Gower matrix embeds points in 2D while respecting the mixed-type distance.
def reduce_dimensions(
    X: np.ndarray,
    method: Literal["pca", "tsne", "mds"] = "tsne",
    n_components: int = 2,
    random_state: int = 42,
    precomputed: bool = False,
) -> np.ndarray:
    """
    Reduce a feature matrix (or precomputed distance matrix) to 2D for visualization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features), or a square distance matrix
        (n_samples, n_samples) when precomputed=True.
    method : {"pca", "tsne", "mds"}, default="tsne"
        Dimensionality reduction method. Use "mds" with precomputed=True for
        Gower-distance-based projections.
    n_components : int, default=2
        Number of output dimensions.
    random_state : int, default=42
        Random seed for reproducibility.
    precomputed : bool, default=False
        If True, X is treated as a precomputed distance matrix (only valid for "mds").

    Returns
    -------
    np.ndarray
        Reduced matrix of shape (n_samples, n_components).
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=30,
        )
    elif method == "mds":
        dissimilarity = "precomputed" if precomputed else "euclidean"
        reducer = MDS(
            n_components=n_components,
            random_state=random_state,
            dissimilarity=dissimilarity,
            normalized_stress="auto",
            n_init=4,
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'pca', 'tsne', or 'mds'.")

    return reducer.fit_transform(X)


def plot_clusters(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    out_path: Optional[str] = None,
    figsize: tuple = (8, 8),
    point_size: int = 10,
    alpha: float = 0.7,
    cmap: str = "okabe_ito",
    show_legend: bool = True,
) -> plt.Figure:
    """
    Plot 2D scatter of clusters.

    Parameters
    ----------
    X_2d : np.ndarray
        2D coordinates of shape (n_samples, 2).
    labels : np.ndarray
        Cluster labels.
    title : str, default="Cluster Visualization"
        Plot title.
    out_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(8, 8)
        Figure size.
    point_size : int, default=10
        Size of scatter points.
    alpha : float, default=0.7
        Point transparency.
    cmap : str, default="okabe_ito"
        Colormap for clusters. Changed from "tab20" for colorblind accessibility.
    show_legend : bool, default=True
        Whether to show cluster legend.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    colors = plt.get_cmap(cmap, len(unique_labels))

    plot_data = []
    for i, label in enumerate(unique_labels):
        mask = labels == label
        count = mask.sum()
        color = "gray" if label == -1 else colors(i)
        label_name = f"Noise (n={count})" if label == -1 else f"Cluster {label} (n={count})"
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=point_size,
            alpha=alpha if label != -1 else 0.3,
            c=[color],
            label=label_name,
        )
        # Collect data for CSV
        for j in range(X_2d[mask].shape[0]):
            plot_data.append([X_2d[mask, 0][j], X_2d[mask, 1][j], label, label_name])


    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    if show_legend and len(unique_labels) <= 15:
        ax.legend(markerscale=2, loc="best")

    plt.tight_layout()

    if out_path:
        # Save plot data to CSV
        df = pd.DataFrame(plot_data, columns=['x', 'y', 'cluster_label', 'cluster_name'])
        csv_path = out_path.replace('.svg', '_scatter_data.csv').replace('.pdf', '_scatter_data.csv')
        df.to_csv(csv_path, index=False)
        fig.savefig(out_path)

    return fig


def plot_clusters_by_attribute(
    X_2d: np.ndarray,
    labels: np.ndarray,
    attribute: np.ndarray,
    attribute_name: str,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
    figsize: tuple = (10, 5),
    point_size: int = 10,
    alpha: float = 0.7,
) -> plt.Figure:
    """
    Plot clusters side-by-side: one colored by cluster, one by attribute.

    Parameters
    ----------
    X_2d : np.ndarray
        2D coordinates of shape (n_samples, 2).
    labels : np.ndarray
        Cluster labels.
    attribute : np.ndarray
        Attribute values for coloring (e.g., gender, age).
    attribute_name : str
        Name of the attribute for labeling.
    title : str, optional
        Overall plot title.
    out_path : str, optional
        Path to save the figure.
    figsize : tuple, default=(10, 5)
        Figure size.
    point_size : int, default=10
        Size of scatter points.
    alpha : float, default=0.7
        Point transparency.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: clusters
    scatter1 = axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap="okabe_ito", s=point_size, alpha=alpha
    )
    axes[0].set_title("By Cluster")
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")

    # Right: attribute
    scatter2 = axes[1].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=attribute, cmap="okabe_ito", s=point_size, alpha=alpha
    )
    axes[1].set_title(f"By {attribute_name}")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    plt.colorbar(scatter2, ax=axes[1])

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if out_path:
        # Save plot data to CSV
        df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'cluster_label': labels,
            attribute_name: attribute
        })
        csv_path = out_path.replace('.svg', '_comparison_data.csv').replace('.pdf', '_comparison_data.csv')
        df.to_csv(csv_path, index=False)
        fig.savefig(out_path)

    return fig


def plot_cluster_composition(
    labels: np.ndarray,
    attribute: np.ndarray,
    attribute_name: str,
    attribute_labels: Optional[dict] = None,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot stacked bar chart showing demographic composition of each cluster.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels.
    attribute : np.ndarray
        Categorical attribute values (e.g., gender encoded as 0/1).
    attribute_name : str
        Name of the attribute.
    attribute_labels : dict, optional
        Mapping from attribute values to display names.
        Example: {0: "Male", 1: "Female"}
    title : str, optional
        Plot title.
    out_path : str, optional
        Path to save the figure.
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_prop_cycle(plt.cycler('color', [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(set(attribute)))]))

    unique_clusters = sorted(set(labels) - {-1})
    unique_attrs = sorted(set(attribute))

    if attribute_labels is None:
        attribute_labels = {v: str(v) for v in unique_attrs}

    # Compute proportions
    proportions = {attr: [] for attr in unique_attrs}
    plot_data = []
    for cluster in unique_clusters:
        cluster_mask = labels == cluster
        cluster_size = cluster_mask.sum()
        row = {'cluster': cluster, 'cluster_size': cluster_size}
        for attr in unique_attrs:
            count = ((labels == cluster) & (attribute == attr)).sum()
            proportion = count / cluster_size if cluster_size > 0 else 0
            proportions[attr].append(proportion)
            row[f'prop_{attribute_labels.get(attr, attr)}'] = proportion
            row[f'count_{attribute_labels.get(attr, attr)}'] = count
        plot_data.append(row)


    # Plot stacked bars
    x = np.arange(len(unique_clusters))
    bottom = np.zeros(len(unique_clusters))

    for attr in unique_attrs:
        ax.bar(
            x, proportions[attr], bottom=bottom,
            label=attribute_labels.get(attr, str(attr))
        )
        bottom += np.array(proportions[attr])

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}\n(n={(labels == c).sum()})" for c in unique_clusters])
    ax.legend(title=attribute_name)
    ax.set_title(title or f"Cluster Composition by {attribute_name}")

    plt.tight_layout()

    if out_path:
        # Save plot data to CSV
        df = pd.DataFrame(plot_data)
        csv_path = out_path.replace('.svg', '_composition_data.csv').replace('.pdf', '_composition_data.csv')
        df.to_csv(csv_path, index=False)
        fig.savefig(out_path)

    return fig


def visualize_clustering_result(
    result: ClusteringResult,
    attribute: Optional[np.ndarray] = None,
    attribute_name: str = "Attribute",
    reduction_method: Literal["pca", "tsne"] = "tsne",
    output_dir: Optional[str] = None,
    prefix: str = "cluster",
) -> dict:
    """
    Generate standard visualizations for a ClusteringResult.

    Parameters
    ----------
    result : ClusteringResult
        Output from the cluster() function.
    attribute : np.ndarray, optional
        Demographic attribute for composition analysis.
    attribute_name : str, default="Attribute"
        Name of the attribute.
    reduction_method : {"umap", "pca"}, default="umap"
        Method for dimensionality reduction.
    output_dir : str, optional
        Directory to save figures. If None, figures are not saved.
    prefix : str, default="cluster"
        Prefix for output filenames.

    Returns
    -------
    dict
        Dictionary of figure objects with keys:
        - "scatter": cluster scatter plot
        - "composition": composition bar chart (if attribute provided)
        - "comparison": side-by-side comparison (if attribute provided)
    """
    figures = {}

    # Reduce dimensions
    X = result.feature_matrix
    if result.mask is not None:
        if attribute is not None:
            attribute = attribute[result.mask]

    X_2d = reduce_dimensions(X, method=reduction_method)

    # Cluster scatter plot
    out_path = f"{output_dir}/{prefix}_scatter.pdf" if output_dir else None
    figures["scatter"] = plot_clusters(
        X_2d, result.labels,
        title=f"Clusters (n={result.n_clusters}, noise={result.n_noise})",
        out_path=out_path,
    )

    if attribute is not None:
        # Composition chart
        out_path = f"{output_dir}/{prefix}_composition.pdf" if output_dir else None
        figures["composition"] = plot_cluster_composition(
            result.labels, attribute, attribute_name,
            out_path=out_path,
        )

        # Side-by-side comparison
        out_path = f"{output_dir}/{prefix}_comparison.pdf" if output_dir else None
        figures["comparison"].plt.tight_layout()
        figures["comparison"] = plot_clusters_by_attribute(
            X_2d, result.labels, attribute, attribute_name,
            out_path=out_path,
        )

    return figures


def plot_silhouette_heatmap(
    silhouette_values: np.ndarray,
    row_labels: list,
    title: str = "Silhouette Scores",
    out_path: Optional[str] = None,
    figsize: tuple = (4, 6),
) -> plt.Figure:
    """
    Plot silhouette scores as a heatmap with inverted colors.

    Parameters
    ----------
    silhouette_values : np.ndarray
        Array of silhouette scores.
    row_labels : list
        Labels for each row (e.g., experimental conditions).
    title : str, default="Silhouette Scores"
        Plot title.
    out_path : str, optional
        Path to save the figure.
    figsize : tuple, default=(4, 6)
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use reversed colormap so higher=blue (better)
    # vlag_r is the reversed version of vlag
    data = np.array(silhouette_values).reshape(-1, 1)

    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        center=0,
        cbar=True,
        cmap="Blues",  # Blue colormap: higher values = darker blue = better
        ax=ax,
        yticklabels=row_labels,
        xticklabels=["silhouette"],
    )

    ax.set_title(title)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()

    if out_path:
        # Save plot data to CSV
        df = pd.DataFrame({'condition': row_labels, 'silhouette': silhouette_values})
        csv_path = out_path.replace('.svg', '_silhouette_data.csv').replace('.pdf', '_silhouette_data.csv')
        df.to_csv(csv_path, index=False)
        fig.savefig(out_path, bbox_inches='tight')

    return fig


def plot_quality_metrics_heatmap(
    metrics_df,
    title: str = "Quality Metrics",
    out_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    silhouette_col: str = "silhouette",
) -> plt.Figure:
    """
    Plot quality metrics heatmap with special handling for silhouette.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with quality metrics. Index should be condition names.
    title : str, default="Quality Metrics"
        Plot title.
    out_path : str, optional
        Path to save the figure.
    figsize : tuple, default=(8, 6)
        Figure size.
    silhouette_col : str, default="silhouette"
        Name of the silhouette column.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [4, 1]})

    # Left: p-value columns (lower=better, use vlag centered at 0.05)
    pval_cols = [c for c in metrics_df.columns if c != silhouette_col]
    if pval_cols:
        sns.heatmap(
            metrics_df[pval_cols],
            annot=True,
            fmt=".4f",
            center=0.05,
            cbar=False,
            cmap=sns.color_palette("vlag", as_cmap=True),
            ax=axes[0],
            robust=True,
        )
        axes[0].set_title("P-values (lower=better)")
        axes[0].xaxis.tick_top()
        axes[0].tick_params(axis='x', which='major', length=0)
        axes[0].tick_params(axis='y', which='major', length=0, pad=5)

    # Right: silhouette column (higher=better, use Blues)
    if silhouette_col in metrics_df.columns:
        sns.heatmap(
            metrics_df[[silhouette_col]],
            annot=True,
            fmt=".3f",
            cbar=False,
            cmap="Blues",  # Higher=darker blue=better
            ax=axes[1],
            yticklabels=False,
        )
        axes[1].set_title("Silhouette\n(higher=better)")
        axes[1].xaxis.tick_top()
        axes[1].tick_params(axis='x', which='major', length=0)
        axes[1].tick_params(axis='y', which='major', length=0)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if out_path:
        # Save plot data to CSV
        csv_path = out_path.replace('.svg', '_quality_metrics_data.csv').replace('.pdf', '_quality_metrics_data.csv')
        metrics_df.to_csv(csv_path)
        fig.savefig(out_path, bbox_inches='tight')

    return fig
