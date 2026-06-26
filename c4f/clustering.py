"""
Clustering module for fairness analysis.

This module provides a flexible clustering function that supports:
- Filtering by prediction outcome (TP, TN, FP, FN)
- Multiple distance metrics (Euclidean, Manhattan, Gower)
- Feature weighting
- Multiple clustering algorithms (HDBSCAN, with placeholders for others)
- Cluster quality evaluation
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Literal, Union
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from kmodes.kprototypes import KPrototypes, euclidean_dissim, matching_dissim
from sklearn_extra.cluster import KMedoids
from .scoring import ScoringFn, silhouette_scorer
@dataclass
class ClusteringResult:
    """Container for clustering results and evaluation metrics."""

    labels: np.ndarray
    n_clusters: int
    n_noise: int
    silhouette: Optional[float]
    calinski_harabasz: Optional[float]
    cluster_sizes: dict
    feature_matrix: np.ndarray
    mask: Optional[np.ndarray]
    distance_matrix: Optional[np.ndarray] = None  # precomputed Gower matrix (non-noise rows), if available

def gower_distance(X: np.ndarray,
                   categorical_features: Optional[list[int]] = None,
                   weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Gower distance matrix for mixed-type data.

    Vectorized implementation that processes data in chunks to conserve memory.
    It loops over features and chunks of samples, avoiding large intermediate matrices.
    Handles NaN by skipping missing features per pair and re-normalizing.

    Parameters
    ----------
    X : np.ndarray
        Numeric feature matrix of shape (n_samples, n_features).
    categorical_features : list[int], optional
        Indices of categorical features (match/mismatch distance).
    weights : np.ndarray, optional
        Feature weights of shape (n_features,).

    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape (n_samples, n_samples) as float32.
    """
    n_samples, n_features = X.shape

    if categorical_features is None:
        categorical_features = []

    if weights is None:
        weights = np.ones(n_features)

    weights = weights / weights.sum()

    X = X.astype(float)

    categorical_set = set(categorical_features)
    numeric_features = [i for i in range(n_features) if i not in categorical_set]

    ranges = np.ones(n_features)
    for i in numeric_features:
        col_valid = X[:, i][~np.isnan(X[:, i])]
        if len(col_valid) > 0:
            r = col_valid.max() - col_valid.min()
            ranges[i] = r if r > 0 else 1.0

    nan_mask = np.isnan(X)
    X_safe = np.where(nan_mask, 0.0, X)
    has_nan = nan_mask.any()

    D = np.zeros((n_samples, n_samples), dtype=np.float32)

    # Process in chunks to limit memory usage
    chunk_size = 1024  # Adjust based on available memory
    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        
        for j in range(i, n_samples, chunk_size):
            end_j = min(j + chunk_size, n_samples)
            
            D_chunk = np.zeros((end_i - i, end_j - j), dtype=np.float32)
            if has_nan:
                W_chunk = np.zeros((end_i - i, end_j - j), dtype=np.float32)

            for k in range(n_features):
                xi_chunk = X_safe[i:end_i, k]
                xj_chunk = X_safe[j:end_j, k]

                if k in categorical_set:
                    contrib = (xi_chunk[:, None] != xj_chunk[None, :]).astype(np.float32)
                else:
                    contrib = np.abs(xi_chunk[:, None] - xj_chunk[None, :]) / ranges[k]
                    contrib = contrib.astype(np.float32)

                if has_nan:
                    nan_i = nan_mask[i:end_i, k]
                    nan_j = nan_mask[j:end_j, k]
                    pair_nan_k = nan_i[:, None] | nan_j[None, :]
                    contrib[pair_nan_k] = 0.0
                    W_chunk += weights[k] * (~pair_nan_k)
                
                D_chunk += weights[k] * contrib

            if has_nan:
                with np.errstate(invalid='ignore', divide='ignore'):
                    D_chunk = np.where(W_chunk > 0, D_chunk / W_chunk, 0.0)
            
            D[i:end_i, j:end_j] = D_chunk
            if i != j:
                D[j:end_j, i:end_i] = D_chunk.T

    return D


def _find_best_k(
    X_score: np.ndarray,
    X_fit: np.ndarray,
    n_min: int,
    n_max: int,
    algorithm: str,
    scoring_fn: ScoringFn,
    random_state: int = 42,
    max_iter: int = 300,
    categorical_features: Optional[list] = None,
    weights: Optional[np.ndarray] = None,
    kmedoids_metric: Optional[str] = None,
) -> tuple:
    """
    Search for the best k in [n_min, n_max] using the given scoring function.

    Parameters
    ----------
    X_score : np.ndarray
        Raw feature matrix used for scoring (always Euclidean-compatible).
    X_fit : np.ndarray
        Input passed to the clusterer — may be a precomputed distance matrix
        (e.g. Gower) for kmedoids. Equals X_score for all other algorithms.
    n_min, n_max : int
        Range of k values to try.
    algorithm : str
        One of 'kmeans', 'bisectingkmeans', 'kmedoids', 'kprototypes'.
    scoring_fn : ScoringFn
        Callable(X, labels) -> float. Higher = better.
    random_state : int
        Random seed.
    max_iter : int
        Max iterations for the clusterer.
    categorical_features : list, optional
        Required for kprototypes.

    Returns
    -------
    tuple of (best_k, best_labels, best_score)
    """
    best_score, best_k, best_labels = -np.inf, n_min, None
    use_precomputed = kmedoids_metric == "precomputed"

    print(f"  k range: [{n_min}, {n_max}]")
    for k in range(n_min, n_max + 1):
        print(f"  Running k={k}...", end="\r", flush=True)
        if algorithm == "kmeans":
            clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=max_iter)
            labels = clusterer.fit_predict(X_fit)
        elif algorithm == "bisectingkmeans":
            clusterer = BisectingKMeans(n_clusters=k, random_state=random_state, max_iter=max_iter)
            labels = clusterer.fit_predict(X_fit)
        elif algorithm == "kmedoids":
            clusterer = KMedoids(n_clusters=k, metric=kmedoids_metric, random_state=random_state, max_iter=max_iter, init="k-medoids++")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn_extra")
                warnings.filterwarnings("ignore", category=FutureWarning)
                labels = clusterer.fit_predict(X_fit)
        elif algorithm == "kprototypes":
            kp_kwargs = dict(n_clusters=k, random_state=random_state, n_init=10, max_iter=max_iter)
            if weights is not None:
                kp_kwargs["gamma"] = 1
            clusterer = KPrototypes(**kp_kwargs)
            labels = clusterer.fit_predict(X_fit, categorical=categorical_features)
        else:
            raise ValueError(f"_find_best_k does not support algorithm: {algorithm}")

        # For precomputed distance matrices pass X_fit so silhouette_scorer uses the right metric
        X_for_scoring = X_fit if use_precomputed else X_score
        score = scoring_fn(X_for_scoring, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    print(f"  Best k={best_k} (score={best_score:.4f})")
    return best_k, best_labels, best_score


def cluster(
    features: Union[np.ndarray, pd.DataFrame],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    subset: Optional[Literal["TP", "TN", "FP", "FN", "TP_TN", "FP_FN"]] = None,
    algorithm: Literal["dbscan", "hdbscan", "kmeans", "bisectingkmeans", "kmedoids", "kprototypes"] = "hdbscan",
    distance: Literal["euclidean", "manhattan", "gower"] = "euclidean",
    categorical_features: Optional[list[int]] = None,
    feature_weights: Optional[Union[np.ndarray, dict]] = None,
    eps: float = 0.5,
    min_cluster_size: int = 15,
    min_samples: int = 10,
    n_clusters: Optional[int] = None,
    n_min: Optional[int] = None,
    n_max: Optional[int] = None,
    max_iter: int = 300,
    random_state: int = 42,
    standardize: bool = True,
    min_datapoints: Optional[int] = None,
    scoring_fn: Optional[ScoringFn] = None,
    ohe_features: Optional[list[int]] = None,
) -> ClusteringResult:
    """
    Perform clustering on features with flexible configuration.

    Parameters
    ----------
    features : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
        Can include embeddings, demographic attributes, etc.
    y_true : np.ndarray, optional
        Ground truth labels for computing confusion matrix subsets.
    y_pred : np.ndarray, optional
        Predicted labels for computing confusion matrix subsets.
    subset : {"TP", "TN", "FP", "FN", "TP_TN", "FP_FN"}, optional
        Filter to specific confusion matrix category before clustering.
        TP_TN = correct predictions, FP_FN = errors.
        Requires y_true and y_pred to be provided.
    algorithm : {"hdbscan", "kmeans", "bisecting", "agglomerative"}, default="hdbscan"
        Clustering algorithm to use. "bisecting" uses Bisecting K-Means.
    distance : {"euclidean", "manhattan", "gower"}, default="euclidean"
        Distance metric for clustering.
        Use "gower" for mixed numeric/categorical features.
    categorical_features : list[int], optional
        Indices of categorical features (required for Gower distance).
    feature_weights : np.ndarray or dict, optional
        Weights for each feature. Can be array of shape (n_features,)
        or dict mapping feature names to weights (if features is DataFrame).
    eps : float, default=0.5                                                                                                                                                            
          Maximum distance between samples for neighborhood (DBSCAN only).
    min_cluster_size : int, default=50
        Minimum cluster size (for HDBSCAN).
    min_samples : int, default=10
        Minimum samples in neighborhood (for HDBSCAN).
    n_clusters : int, optional
        Number of clusters (for KMeans, Agglomerative).
    n_min : int, optional
        Minimum number of clusters for range-based search.
    n_max : int, optional
        Maximum number of clusters for range-based search.
    max_iter : int, default=300
        Maximum number of iterations for KMeans/BisectingKMeans.
    random_state : int, default=42
        Random seed for reproducibility.
    standardize : bool, default=True
        Whether to standardize numeric features before clustering.

    Returns
    -------
    ClusteringResult
        Dataclass containing cluster labels, metrics, and metadata.
    """
    # Convert DataFrame to numpy if needed
    # Always copy to avoid in-place standardization mutating the caller's DataFrame
    if isinstance(features, pd.DataFrame):
        feature_names = features.columns.tolist()
        X = features.values.copy()
    else:
        feature_names = None
        X = features.copy()

    # Unify min_datapoints with HDBSCAN's native min_cluster_size:
    # When min_datapoints is set for HDBSCAN, pass it directly as min_cluster_size so the
    # algorithm enforces the minimum internally (better than a post-hoc relabeling).
    # For all other algorithms, min_datapoints is applied as a post-hoc filter below.
    if algorithm == "hdbscan" and min_datapoints is not None:
        min_cluster_size = min_datapoints
        min_datapoints = None  # already handled natively above
        
    # Compute mask for confusion matrix subset
    mask = None
    if subset is not None:
        if y_true is None or y_pred is None:
            raise ValueError(f"subset='{subset}' requires y_true and y_pred")
        if subset == "TP":
            mask = (y_true == 1) & (y_pred == 1)
        elif subset == "TN":
            mask = (y_true == 0) & (y_pred == 0)
        elif subset == "FP":
            mask = (y_true == 0) & (y_pred == 1)
        elif subset == "FN":
            mask = (y_true == 1) & (y_pred == 0)
        elif subset == "TP_TN":
            mask = y_true == y_pred
        elif subset == "FP_FN":
            mask = y_true != y_pred
        else:
            raise ValueError(f"Invalid subset: {subset}")

        X = X[mask]

    # Handle feature weights
    weights = None
    if feature_weights is not None:
        if isinstance(feature_weights, dict) and feature_names is not None:
            weights = np.array([feature_weights.get(name, 1.0) for name in feature_names])
        else:
            weights = np.asarray(feature_weights)

    if standardize and distance != "gower" and algorithm != "kprototypes":
        exclude = set(categorical_features or []) | set(ohe_features or [])
        if exclude:
            X = X.astype(float)
            numeric_mask = [i for i in range(X.shape[1]) if i not in exclude]
            scaler = StandardScaler()
            X[:, numeric_mask] = scaler.fit_transform(X[:, numeric_mask])
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    # Apply feature weights (for non-Gower distances)
    if weights is not None and distance != "gower":
        X = X * np.sqrt(weights)

    if algorithm == "dbscan":
        if distance == "gower":
            # DBSCAN does not have built-in support for Gower distance.
            # Compute the distance matrix manually and pass with metric="precomputed".
            dist_matrix = gower_distance(X, categorical_features, weights)
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="precomputed",
            )
            labels = clusterer.fit_predict(dist_matrix)
        elif distance in ("euclidean", "manhattan"):
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=distance,
            )
            labels = clusterer.fit_predict(X)
        else:
            raise ValueError(f"DBSCAN does not support distance='{distance}'. Use 'euclidean', 'manhattan', or 'gower'.")
    elif algorithm == "hdbscan":
        if distance == "gower":
            # HDBSCAN does not have built-in support for Gower distance.
            # Compute the distance matrix manually and pass with metric="precomputed".
            dist_matrix = gower_distance(X, categorical_features, weights)
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="precomputed",
            )
            labels = clusterer.fit_predict(dist_matrix)
        elif distance in ("euclidean", "manhattan"):
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=distance,
            )
            labels = clusterer.fit_predict(X)
        else:
            raise ValueError(f"HDBSCAN does not support distance='{distance}'. Use 'euclidean', 'manhattan', or 'gower'.")


    elif algorithm in ("kmeans", "bisectingkmeans", "kmedoids", "kprototypes"):
        # Default scoring function if none provided
        _scoring = scoring_fn if scoring_fn is not None else silhouette_scorer
        # Validate kprototypes requirements
        if algorithm == "kprototypes":
            # NOTE: KPrototypes uses its own internal distance metric (Huang's cost function):
            #   - Numeric features: squared Euclidean distance
            #   - Categorical features: simple matching dissimilarity (0 if match, 1 otherwise)
            # Gower distance is not compatible with KPrototypes.
            # For Gower-based mixed-type clustering, use DBSCAN or HDBSCAN with --distance gower.
            if categorical_features is None or len(categorical_features) == 0:
                raise ValueError("kprototypes requires categorical_features to be specified")

        if algorithm in ("kmeans", "bisectingkmeans") and distance == "gower":
            raise ValueError(
                f"{algorithm} requires a Euclidean feature space to compute centroids and "
                f"cannot use Gower distance. Use --algorithm kmedoids --distance gower instead."
            )

        if algorithm == "kmedoids":
            if distance == "gower":
                dist_matrix = gower_distance(X, categorical_features, weights)

                # Tie-breaking for discrete data: add tiny random noise to the distance matrix.
                # This helps k-medoids++ find distinct initial medoids and prevents assignment ties
                # where all points collapse into the first cluster.
                # The noise is symmetric and small enough to not distort real clusters.
                if dist_matrix.shape[0] > 1:
                    rng = np.random.RandomState(random_state)
                    n_samples = dist_matrix.shape[0]
                    chunk_size = 1024  # Process in chunks to conserve memory
                    for i in range(0, n_samples, chunk_size):
                        end_i = min(i + chunk_size, n_samples)
                        for j in range(i, n_samples, chunk_size):
                            end_j = min(j + chunk_size, n_samples)

                            noise_chunk = rng.rand(end_i - i, end_j - j).astype(np.float32)

                            if i == j:
                                # Diagonal block: must be symmetric
                                noise_chunk = (noise_chunk + noise_chunk.T) / 2
                                dist_matrix[i:end_i, j:end_j] += noise_chunk * 1e-6
                            else:
                                # Off-diagonal block: add transpose to symmetric position
                                dist_matrix[i:end_i, j:end_j] += noise_chunk * 1e-6
                                dist_matrix[j:end_j, i:end_i] += noise_chunk.T * 1e-6

                    np.fill_diagonal(dist_matrix, 0)
                
                X_fit = dist_matrix
                kmedoids_metric = "precomputed"
            else:
                X_fit = X
                kmedoids_metric = "euclidean"
        else:
            X_fit = X
            kmedoids_metric = None

        if n_clusters is not None:
            if algorithm == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=max_iter)
                labels = clusterer.fit_predict(X_fit)
            elif algorithm == "bisectingkmeans":
                # NOTE: Open research question — Mitzal & Radecka's papers may suggest BisectingKMeans
                # is not well-suited for fairness clustering. Evaluate whether it adds value over standard KMeans.
                clusterer = BisectingKMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
                labels = clusterer.fit_predict(X_fit)
            elif algorithm == "kmedoids":
                clusterer = KMedoids(n_clusters=n_clusters, metric=kmedoids_metric, random_state=random_state, max_iter=max_iter, init="k-medoids++")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn_extra")
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    labels = clusterer.fit_predict(X_fit)
            elif algorithm == "kprototypes":
                kp_kwargs = dict(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=max_iter)
                if weights is not None:
                    kp_kwargs["gamma"] = 1
                clusterer = KPrototypes(**kp_kwargs)
                labels = clusterer.fit_predict(X_fit, categorical=categorical_features)

        # Range-based k search using scoring function
        elif n_min is not None and n_max is not None:
            best_k, labels, best_score = _find_best_k(
                X, X_fit, n_min, n_max, algorithm, _scoring,
                random_state=random_state, max_iter=max_iter,
                categorical_features=categorical_features,
                weights=weights,
                kmedoids_metric=kmedoids_metric,
            )
            print(f"  Best k={best_k} (score={best_score:.3f})")

        else:
            raise ValueError(f"n_clusters or n_min/n_max required for {algorithm}")

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Filter small clusters to noise if min_datapoints is set
    if min_datapoints is not None:
        for label in set(labels):
            if label == -1:
                continue
            cluster_count = (labels == label).sum()
            if cluster_count < min_datapoints:
                labels = np.where(labels == label, -1, labels)

    # Relabel clusters to contiguous 0..k-1 (kmedoids k-search can produce non-sequential labels)
    unique_non_noise = sorted(set(labels) - {-1})
    if unique_non_noise != list(range(len(unique_non_noise))):
        label_map = {old: new for new, old in enumerate(unique_non_noise)}
        labels = np.array([label_map.get(l, -1) for l in labels])

    # Compute evaluation metrics
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = (labels == -1).sum()

    # Silhouette and Calinski-Harabasz require at least 2 clusters and numeric data
    silhouette = None
    calinski = None
    gower_mat = None
    if n_clusters >= 2:
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > n_clusters:
            if algorithm == "kprototypes":
                X_sub = X[non_noise_mask].astype(object)
                cat_set = set(categorical_features or [])
                num_idx = np.array([i for i in range(X_sub.shape[1]) if i not in cat_set])
                cat_idx = np.array(sorted(cat_set), dtype=int) if cat_set else np.array([], dtype=int)
                gamma = float(clusterer.gamma)
                n_sub = X_sub.shape[0]
                num_dm = np.zeros((n_sub, n_sub))
                cat_dm = np.zeros((n_sub, n_sub))
                if len(num_idx):
                    X_num = X_sub[:, num_idx].astype(float)
                    diff = X_num[:, np.newaxis, :] - X_num[np.newaxis, :, :]
                    num_dm = np.sqrt(np.sum(diff ** 2, axis=2))
                if len(cat_idx):
                    X_cat = X_sub[:, cat_idx]
                    cat_dm = np.sum(X_cat[:, np.newaxis, :] != X_cat[np.newaxis, :, :], axis=2).astype(float)
                gower_mat = num_dm + gamma * cat_dm
                silhouette = silhouette_score(gower_mat, labels[non_noise_mask], metric="precomputed")
            elif distance == "gower":
                # Recompute Gower matrix on non-noise rows and use metric="precomputed"
                gower_mat = gower_distance(X[non_noise_mask], categorical_features, weights)
                silhouette = silhouette_score(gower_mat, labels[non_noise_mask], metric="precomputed")
                # Calinski-Harabasz not supported with precomputed metric
            else:
                silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                calinski = calinski_harabasz_score(X[non_noise_mask], labels[non_noise_mask])

    # Compute cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        cluster_sizes[label] = (labels == label).sum()

    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=silhouette,
        calinski_harabasz=calinski,
        cluster_sizes=cluster_sizes,
        feature_matrix=X,
        mask=mask,
        distance_matrix=gower_mat,
    )