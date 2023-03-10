import json

import numpy as np
import streamlit as st
from sklearn import cluster
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import kneighbors_graph

from constants import DEFAULT_DATASET_N_SAMPLES, DEFAULT_PARAMS, DEFAULT_N_CLUSTERS
from data_classes import DatasetName, ClusterAlgo


@st.cache_data
def read_cluster_algo_default_params():
    with open('cluster_algo_default_params.json') as json_file:
        cluster_algo_default_params = json.load(json_file)
    return cluster_algo_default_params

def get_dataset_points(dataset_name: DatasetName, is_3d: bool,
                       n_samples: int = DEFAULT_DATASET_N_SAMPLES) -> np.ndarray:
    """ Returns a 2d or 3d numpy array for a provided DatasetName object
    TODO: Add some variance to z dimension
    """
    n_features = 3 if is_3d else 2
    if dataset_name == DatasetName.BLOBS:
        dataset_points, default_cluster_labels = make_blobs(n_samples=n_samples, n_features=n_features,
                                                              random_state=8)
    elif dataset_name == DatasetName.CIRCLES:
        dataset_points, default_cluster_labels = make_circles(n_samples=n_samples, factor=0.5, noise=0.05,
                                                                random_state=6)
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


def get_cluster_features(dataset_points: np.ndarray) -> np.ndarray:
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


def get_cluster_algo_parameters(cluster_algo: ClusterAlgo, dataset_name: DatasetName):
    params = DEFAULT_PARAMS.copy()
    cluster_algo_default_params = read_cluster_algo_default_params()
    params.update(cluster_algo_default_params[dataset_name])
    # Cluster Algo Parameters
    if cluster_algo == ClusterAlgo.KMEANS:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"], key="KMEANS NoC")
        cluster_algo_kwargs = {"n_clusters": n_clusters, "n_init": "auto", "random_state": 1}
    elif cluster_algo == ClusterAlgo.DBSCAN:
        eps = st.sidebar.number_input("Epsilon", value=0.5)
        min_samples = st.sidebar.number_input("Minimum Samples", value=5)
        cluster_algo_kwargs = {"eps": eps, "min_samples": min_samples}
    elif cluster_algo == ClusterAlgo.GAUSSIAN_MIXTURE:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=DEFAULT_N_CLUSTERS, key="GAUSSIAN_MIXTURE NoC")
        covariance_type = st.sidebar.selectbox(
            "Covariance type", ["full", "tied", "diag", "spherical"])
        cluster_algo_kwargs = {"n_components": n_clusters, "covariance_type": covariance_type}
    elif cluster_algo == ClusterAlgo.MEAN_SHIFT:
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])
        cluster_algo_kwargs = {"bandwidth": bandwidth, "bin_seeding": True}
    elif cluster_algo in [ClusterAlgo.WARD, ClusterAlgo.AGGLOMERATIVE_CLUSTERING]:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"],
                                             key=f"{cluster_algo} NoC")
        n_neighbors = st.sidebar.number_input("Number of Neighbors for Connectivity", value=params["n_neighbors"])
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=n_neighbors, include_self=False
        )
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # TODO: Include input for linkage + cityblock + connectivity
        cluster_algo_kwargs = {"n_clusters": n_clusters, "connectivity": connectivity,
                               "linkage": "ward" if cluster_algo == ClusterAlgo.WARD else "average"}
        if cluster_algo == ClusterAlgo.AGGLOMERATIVE_CLUSTERING:
            cluster_algo_kwargs["metric"] = "cityblock"
    elif cluster_algo == ClusterAlgo.SPECTRAL_CLUSTERING:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"],
                                             key="SPECTRAL_CLUSTERING NoC")
        # TODO: Include input for eigen_solver + affinity
        cluster_algo_kwargs = {"n_clusters": n_clusters, "eigen_solver": "arpack", "affinity": "nearest_neighbors"}
    elif cluster_algo == ClusterAlgo.OPTICS:
        # TODO: Include input for min_samples + xi + min_cluster_size
        cluster_algo_kwargs = {"min_samples": params["min_samples"], "xi": params["xi"],
                               "min_cluster_size": params["min_cluster_size"]}
    elif cluster_algo == ClusterAlgo.AFFINITY_PROPAGATION:
        # TODO: Include input for damping + preference + min_cluster_size
        cluster_algo_kwargs = {"damping": params["damping"], "preference": params["preference"], "random_state": 0}
    elif cluster_algo == ClusterAlgo.BIRCH:
        # TODO: Include input for damping + preference + min_cluster_size
        n_clusters = st.sidebar.number_input("Number of Clusters", value=params["n_clusters"], key="BIRCH NoC")
        cluster_algo_kwargs = {"n_clusters": n_clusters}
    else:
        cluster_algo_kwargs = {}
    return cluster_algo_kwargs
