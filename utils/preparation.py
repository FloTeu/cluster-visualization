import json
import random
from typing import Optional

import numpy as np
import streamlit as st
from sklearn import cluster
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import _VALID_METRICS as sklearn_metrics

from constants import DEFAULT_DATASET_N_SAMPLES, DEFAULT_CLUSTER_ALGO_PARAMS
from data_classes import DatasetName, ClusterAlgo


def read_cluster_algo_default_params() -> dict:
    """Reads json file with dynamic default values of sklearn cluster algorithms """
    with open('cluster_algo_default_params.json') as json_file:
        cluster_algo_default_params = json.load(json_file)
    return cluster_algo_default_params

def get_sklearn_metrics(default):
    """Returns a list of sklearn metrics and includes default as first element of array"""
    metrics = sklearn_metrics.copy()
    metrics.remove(default)
    return [default] + metrics

def get_default_dataset_points(dataset_name: DatasetName, is_3d: bool,
                               n_samples: int = DEFAULT_DATASET_N_SAMPLES) -> np.ndarray:
    """Returns a 2d or 3d numpy array for a provided DatasetName object"""
    n_features = 3 if is_3d else 2
    if dataset_name == DatasetName.BLOBS:
        dataset_points, default_cluster_labels = make_blobs(n_samples=n_samples, n_features=n_features,
                                                              random_state=0)
    elif dataset_name == DatasetName.CIRCLES:
        dataset_points, default_cluster_labels = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=0)
        if is_3d:
            dataset_points = np.insert(dataset_points, 2, 1, axis=1)
    elif dataset_name == DatasetName.Moons:
        dataset_points, default_cluster_labels = make_moons(n_samples=n_samples, noise=0.05)
        if is_3d:
            dataset_points = np.insert(dataset_points, 2, 1, axis=1)
    elif dataset_name == DatasetName.NO_STRUCTURE:
        dataset_points, default_cluster_labels = np.random.rand(n_samples, n_features), None
    elif dataset_name == DatasetName.VARIED_VARIANCES:
        dataset_points, default_cluster_labels = make_blobs(
            n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], n_features=n_features, random_state=170
        )
    elif dataset_name == DatasetName.ANISOTROPICLY_DISTRIBUTED:
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        if is_3d:
            transformation = [[0.6, -0.6, 0], [0.6, -0.6, 0.6], [-0.4, 0.8, -0.4]]
        X_aniso = np.dot(X, transformation)
        dataset_points, default_cluster_labels = (X_aniso, y)
    return dataset_points


def add_user_data_input_listener(dataset_points: np.ndarray) -> np.ndarray:
    """
    Creates the streamlit text area input to allow the user to provide clustering features on their own.
    Returns a 2d or 3d numpy array with clustering features.
    """
    is_3d = dataset_points.shape[1] == 3
    # Numpy array to streamlit text area strings
    x_str = st.sidebar.text_area(
        "x (comma separated numbers)",
        value=(",").join(map(str, dataset_points[:, 0])),
    )
    x = list(map(float,x_str.split(",")))
    y_str = st.sidebar.text_area(
        "y (comma separated numbers",
        value=(",").join(map(str, dataset_points[:, 1])),
    )
    y = list(map(float,y_str.split(",")))
    if is_3d:
        z_str = st.sidebar.text_area(
            "z (comma separated numbers",
            value=(",").join(map(str, dataset_points[:, 2])),
        )
        z = list(map(float,z_str.split(",")))
        cluster_features = np.column_stack((x, y, z))
    else:
        cluster_features = np.column_stack((x, y))

    return cluster_features


def get_cluster_algo_parameters(cluster_algo: ClusterAlgo, cluster_features: np.ndarray, dataset_name: DatasetName) -> dict:
    """ Includes ui elements for parameter configuration.
        Based on the selected dataset_name the default cluster algo parameter may change.

    Args:
        cluster_algo: sklearn cluster algorithm string or ClusterAlgo object
        cluster_features: 2 or 3 dimensional cluster features
        dataset_name: Dataset name which implies data points

    Returns:
        Default sklearn cluster algorithm parameters as dict

    """
    params = DEFAULT_CLUSTER_ALGO_PARAMS.copy()
    cluster_algo_default_params = read_cluster_algo_default_params()
    params.update(cluster_algo_default_params[dataset_name])

    if cluster_algo == ClusterAlgo.KMEANS:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"], key=f"{cluster_algo} NoC")
        init = st.sidebar.selectbox("Method for Initialization", ["k-means++", "random"], key=f"{cluster_algo} MoI")
        cluster_algo_kwargs = {"n_clusters": n_clusters, "n_init": "auto", "init": init, "random_state": 0}
    elif cluster_algo == ClusterAlgo.AFFINITY_PROPAGATION:
        damping = st.sidebar.number_input("Damping Factor", value=params["damping"], min_value=0.5, max_value=0.99999999, key=f"{cluster_algo} DF")
        preference = st.sidebar.number_input("Preference", value=params["preference"], key=f"{cluster_algo} P")
        cluster_algo_kwargs = {"damping": damping, "preference": preference, "random_state": 0}
    elif cluster_algo == ClusterAlgo.MEAN_SHIFT:
        bandwidth = cluster.estimate_bandwidth(cluster_features, quantile=params["quantile"])
        bandwidth_ui = st.sidebar.number_input("Bandwidth", value=bandwidth, key=f"{cluster_algo} B")
        bin_seeding = st.sidebar.checkbox("Bin Seeding", value=True, key=f"{cluster_algo} BS")
        cluster_algo_kwargs = {"bandwidth": bandwidth_ui, "bin_seeding": bin_seeding}
    elif cluster_algo == ClusterAlgo.SPECTRAL_CLUSTERING:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"],
                                             key="SPECTRAL_CLUSTERING NoC")
        eigen_solver = st.sidebar.selectbox("Eigenvalue Decomposition Strategy", ["arpack", "lobpcg", "amg"], key=f"{cluster_algo} EDS")
        affinity = st.sidebar.selectbox("Construction of Affinity Matrix", ["nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"], key=f"{cluster_algo} CAM")
        n_neighbors = st.sidebar.number_input("Number of Neighbors", 10, key="SPECTRAL_CLUSTERING NoN")
        cluster_algo_kwargs = {"n_clusters": n_clusters, "eigen_solver": eigen_solver, "affinity": affinity, "n_neighbors": n_neighbors,"random_state": 0}
    elif cluster_algo in [ClusterAlgo.WARD, ClusterAlgo.AGGLOMERATIVE_CLUSTERING]:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"],
                                             key=f"{cluster_algo} NoC")
        n_neighbors = st.sidebar.number_input("Number of Neighbors for Connectivity", value=params["n_neighbors"], key=f"{cluster_algo} NoN")
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            cluster_features, n_neighbors=n_neighbors, include_self=False
        )
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        if cluster_algo == ClusterAlgo.AGGLOMERATIVE_CLUSTERING:
            linkage = st.sidebar.selectbox("Linkage Criterion", ["average", "ward", "complete", "single"], key=f"{cluster_algo} CAM")
            metric = st.sidebar.selectbox("Metric to Compute Linkage", get_sklearn_metrics(default="cityblock"), key=f"{cluster_algo} MCL")
        else:
            linkage = "ward"
            metric = None
        cluster_algo_kwargs = {"n_clusters": n_clusters, "connectivity": connectivity,
                               "linkage": linkage, "metric": metric}
    elif cluster_algo == ClusterAlgo.DBSCAN:
        eps = st.sidebar.number_input("Epsilon", value=params["eps"])
        metric = st.sidebar.selectbox("Metric",
                                      get_sklearn_metrics(default="euclidean"),
                                      key=f"{cluster_algo} MDC")
        min_samples = st.sidebar.number_input("Minimum Samples", value=params["min_samples"], key=f"{cluster_algo} MS")
        algorithm = st.sidebar.selectbox("Algorithm for Nearest Neighbors",
                                      ["auto", "ball_tree", "kd_tree", "brute"],
                                      key=f"{cluster_algo} ANN")
        cluster_algo_kwargs = {"eps": eps, "min_samples": min_samples, "algorithm": algorithm, "metric": metric}
    elif cluster_algo == ClusterAlgo.OPTICS:
        min_samples = st.sidebar.number_input("Minimum Samples", value=params["min_samples"], key=f"{cluster_algo} MS")
        metric = st.sidebar.selectbox("Metric for Distance Computation",
                                      get_sklearn_metrics(default="minkowski"),
                                      key=f"{cluster_algo} MDC")
        p = st.sidebar.number_input("Parameter for the Minkowski metric", min_value=1, value=2, key=f"{cluster_algo} P")
        xi = st.sidebar.number_input("XI", value=params["xi"], min_value=0.0, max_value=1.0, key=f"{cluster_algo} XI")
        min_cluster_size = st.sidebar.number_input("Minimum Number of Samples in Cluster", value=params["min_cluster_size"], key=f"{cluster_algo} min_cluster_size")
        algorithm = st.sidebar.selectbox("Algorithm for Nearest Neighbors",
                                      ["auto", "ball_tree", "kd_tree", "brute"],
                                      key=f"{cluster_algo} ANN")
        cluster_algo_kwargs = {"min_samples": min_samples, "p": p, "xi": xi, "metric": metric,
                               "min_cluster_size": min_cluster_size, "algorithm": algorithm}
    elif cluster_algo == ClusterAlgo.BIRCH:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"], key=f"{cluster_algo} NoC")
        threshold = st.sidebar.number_input("Threshold", value=0.5, key=f"{cluster_algo} T")
        branching_factor = st.sidebar.number_input("Branching Factor", value=50, key=f"{cluster_algo} BF")
        cluster_algo_kwargs = {"n_clusters": n_clusters, "threshold": threshold, "branching_factor": branching_factor}
    elif cluster_algo == ClusterAlgo.GAUSSIAN_MIXTURE:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"], key=f"{cluster_algo} NoC")
        covariance_type = st.sidebar.selectbox(
            "Covariance type", ["full", "tied", "diag", "spherical"])
        init_params = st.sidebar.selectbox(
            "Init Method", ["kmeans", "k-means++", "random", "random_from_data"])
        cluster_algo_kwargs = {"n_components": n_clusters, "covariance_type": covariance_type, "init_params": init_params, "random_state": 0}
    else:
        cluster_algo_kwargs = {}
    return cluster_algo_kwargs

def split_list(list_obj, split_size):
    return [list_obj[i:i+split_size] for i in range(0, len(list_obj), split_size)]
