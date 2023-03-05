from enum import Enum

import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, mixture


class ClusterAlgo(str, Enum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    MEAN_SHIFT = "mean_shift"
    WARD = "ward"
    AGGLOMERATIVE_CLUSTERING = "agglomerative_clustering"
    SPECTRAL_CLUSTERING = "spectral_clustering"
    OPTICS = "optics"
    AFFINITY_PROPAGATION = "affinity_propagation"
    BIRCH = "birch"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


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
    data_method = st.sidebar.selectbox(
        "Default Data", ["Blobs", "Circles", "Moons", "No Structure", "Varied Variances"])
    if data_method == "Blobs":
        default_cluster_features, default_cluster_labels = make_blobs(n_samples=n_samples, centers=DEFAULT_N_CLUSTERS,
                                                                      n_features=3 if is_3d else 2, random_state=6)
    elif data_method == "Circles":
        default_cluster_features, default_cluster_labels = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=6)
    elif data_method == "Moons":
        default_cluster_features, default_cluster_labels = make_moons(n_samples=n_samples, noise=0.05)
    elif data_method == "No Structure":
        default_cluster_features, default_cluster_labels = np.random.rand(n_samples, 3 if is_3d else 2), None
    elif data_method == "Varied Variances":
        default_cluster_features, default_cluster_labels = make_blobs(
            n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], n_features=3 if is_3d else 2, random_state=170
        )

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

    # Cluster Algo
    cluster_algo = st.sidebar.selectbox(
        "Cluster Algorithm", [e.value for e in CLUSTER_ALGORITHMS])

    if cluster_algo == ClusterAlgo.KMEANS:
        # Cluster Algo Parameters
        n_clusters = st.sidebar.number_input("Number of Clusters", value=DEFAULT_N_CLUSTERS)
        cluster_algo_kwargs = {"n_clusters": n_clusters}
    elif cluster_algo == ClusterAlgo.DBSCAN:
        eps = st.sidebar.number_input("Epsilon", value=0.5)
        min_samples = st.sidebar.number_input("Minimum Samples", value=5)
        cluster_algo_kwargs = {"eps": eps, "min_samples": min_samples}
    elif cluster_algo == ClusterAlgo.GAUSSIAN_MIXTURE:
        n_clusters = st.sidebar.number_input("Number of Clusters", value=DEFAULT_N_CLUSTERS)
        covariance_type = st.sidebar.selectbox(
            "Covariance type", ["full", "tied", "diag", "spherical"])
        cluster_algo_kwargs = {"n_components": n_clusters, "covariance_type": covariance_type}
    else:
        cluster_algo_kwargs = {}

    x, y, z = parse_feature_input(x_str, y_str, z_str)
    if z:
        X = np.column_stack((x, y, z))
    else:
        X = np.column_stack((x, y))

    cluster_labels = get_cluster_labels(X, cluster_algo, **cluster_algo_kwargs)

    # plot the figure
    st.write(plot_figure(x, y, z, cluster_labels))


if __name__ == "__main__":
    main()
