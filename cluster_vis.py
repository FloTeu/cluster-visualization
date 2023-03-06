from enum import Enum
from typing import List, Dict

import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, mixture


class ClusterAlgo(str, Enum):
    KMEANS = "KMeans"
    DBSCAN = "DBSCAN"
    MEAN_SHIFT = "mean_shift"
    WARD = "ward"
    AGGLOMERATIVE_CLUSTERING = "agglomerative_clustering"
    SPECTRAL_CLUSTERING = "spectral_clustering"
    OPTICS = "optics"
    AFFINITY_PROPAGATION = "affinity_propagation"
    BIRCH = "birch"
    GAUSSIAN_MIXTURE = "Gaussian Mixture"


# TODO: Customize icon
st.set_page_config(
    page_title="ikneed",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

CLUSTER_ALGORITHMS = [ClusterAlgo.KMEANS, ClusterAlgo.DBSCAN, ClusterAlgo.MEAN_SHIFT, ClusterAlgo.WARD,
                      ClusterAlgo.AGGLOMERATIVE_CLUSTERING, ClusterAlgo.SPECTRAL_CLUSTERING, ClusterAlgo.OPTICS,
                      ClusterAlgo.AFFINITY_PROPAGATION, ClusterAlgo.BIRCH, ClusterAlgo.GAUSSIAN_MIXTURE]
DEFAULT_N_CLUSTERS = 4


def parse_feature_input(x_str, y_str, z_str=None):
    # TODO clean up. Maybe allow more than 2 features?

    # parse x and y
    x = [float(_) for _ in x_str.split(",")]
    y = [float(_) for _ in y_str.split(",")]
    if z_str:
        z = [float(_) for _ in z_str.split(",")]
    else:
        z = None

    return x, y, z


def plot_figure(x, y, z=None, cluster_labels=None):
    if z:
        return px.scatter_3d(x=x, y=y, z=z, color=cluster_labels)
    else:
        return px.scatter(x=x, y=y, color=cluster_labels)


def get_cluster_labels(X, cluster_algo: ClusterAlgo, **kwargs):
    if cluster_algo == ClusterAlgo.KMEANS:
        return cluster.KMeans(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.DBSCAN:
        return cluster.DBSCAN(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.MEAN_SHIFT:
        return cluster.MeanShift(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.WARD:
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=kwargs["n_neighbors"], include_self=False
        )
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        return cluster.AgglomerativeClustering(**kwargs, linkage="ward", connectivity=connectivity).fit(X).labels_
    elif cluster_algo == ClusterAlgo.AGGLOMERATIVE_CLUSTERING:
        return cluster.AgglomerativeClustering(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.SPECTRAL_CLUSTERING:
        return cluster.SpectralClustering(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.OPTICS:
        return cluster.OPTICS(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.AFFINITY_PROPAGATION:
        return cluster.AffinityPropagation(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.BIRCH:
        return cluster.Birch(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.GAUSSIAN_MIXTURE:
        return mixture.GaussianMixture(**kwargs).fit(X).predict(X)


def main():
    """
    The main function
    """
    is_3d = st.sidebar.checkbox("Use 3D Features", value=False)

    # Default Data
    n_samples = 500
    n_features = 3 if is_3d else 2
    data_method = st.sidebar.selectbox(
        "Default Data", ["Blobs", "Circles", "Moons", "No Structure", "Varied Variances", "Anisotropicly distributed"])
    if data_method == "Blobs":
        default_cluster_features, default_cluster_labels = make_blobs(n_samples=n_samples, centers=DEFAULT_N_CLUSTERS,
                                                                      n_features=n_features, random_state=6)
    elif data_method == "Circles":
        default_cluster_features, default_cluster_labels = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=6)
        if is_3d:
            default_cluster_features = np.insert(default_cluster_features, 2, 1, axis=1)
    elif data_method == "Moons":
        default_cluster_features, default_cluster_labels = make_moons(n_samples=n_samples, noise=0.05)
        if is_3d:
            default_cluster_features = np.insert(default_cluster_features, 2, 1, axis=1)
    elif data_method == "No Structure":
        default_cluster_features, default_cluster_labels = np.random.rand(n_samples, n_features), None
    elif data_method == "Varied Variances":
        default_cluster_features, default_cluster_labels = make_blobs(
            n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], n_features=n_features, random_state=170
        )
    elif data_method == "Anisotropicly distributed":
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        if is_3d:
            transformation = [[0.6, -0.6, 0], [0.6, -0.6, 0.6], [-0.4, 0.8, -0.4]]
        X_aniso = np.dot(X, transformation)
        default_cluster_features, default_cluster_labels = (X_aniso, y)

    x_str = st.sidebar.text_area(
        "x (comma separated numbers)",
        value=(",").join(map(str, default_cluster_features[:, 0])),
    )
    y_str = st.sidebar.text_area(
        "y (comma separated numbers",
        value=(",").join(map(str, default_cluster_features[:, 1])),
    )
    if is_3d:
        z_str = st.sidebar.text_area(
            "z (comma separated numbers",
            value=(",").join(map(str, default_cluster_features[:, 2])),
        )
    else:
        z_str = None

    x, y, z = parse_feature_input(x_str, y_str, z_str)
    if z:
        X = np.column_stack((x, y, z))
    else:
        X = np.column_stack((x, y))

    # Cluster Algo
    cluster_algos: List[str] = st.sidebar.multiselect(
        "Cluster Algorithms", [e.value for e in CLUSTER_ALGORITHMS], [CLUSTER_ALGORITHMS[0]])
    display_cols = st.columns(len(cluster_algos))

    for i, cluster_algo_str in enumerate(cluster_algos):
        # Cluster Algo Parameters
        if cluster_algo_str == ClusterAlgo.KMEANS:
            st.sidebar.title(ClusterAlgo.KMEANS.value)
            n_clusters = st.sidebar.number_input("Number of Clusters", value=DEFAULT_N_CLUSTERS)
            cluster_algo_kwargs = {"n_clusters": n_clusters, "random_state": 1}
        elif cluster_algo_str == ClusterAlgo.DBSCAN:
            st.sidebar.title(ClusterAlgo.DBSCAN.value)
            eps = st.sidebar.number_input("Epsilon", value=0.5)
            min_samples = st.sidebar.number_input("Minimum Samples", value=5)
            cluster_algo_kwargs = {"eps": eps, "min_samples": min_samples}
        elif cluster_algo_str == ClusterAlgo.GAUSSIAN_MIXTURE:
            st.sidebar.title(ClusterAlgo.GAUSSIAN_MIXTURE.value)
            n_clusters = st.sidebar.number_input("Number of Clusters", value=DEFAULT_N_CLUSTERS)
            covariance_type = st.sidebar.selectbox(
                "Covariance type", ["full", "tied", "diag", "spherical"])
            cluster_algo_kwargs = {"n_components": n_clusters, "covariance_type": covariance_type}
        else:
            cluster_algo_kwargs = {}

        cluster_labels = get_cluster_labels(X, cluster_algo_str, **cluster_algo_kwargs)

        # plot the figure
        display_cols[i].subheader(cluster_algo_str)
        fig = plot_figure(x, y, z, cluster_labels)
        display_cols[i].plotly_chart(fig, use_container_width=True)




if __name__ == "__main__":
    main()
