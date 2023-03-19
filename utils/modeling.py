import numpy as np
from sklearn import cluster, mixture

from data_classes import ClusterAlgo


def get_cluster_labels(cluster_features, cluster_algo: ClusterAlgo, **kwargs) -> np.ndarray:
    """ Calculates and returns the cluster label for each data point in cluster_features

    Args:
        cluster_features: 2 or 3 dimensional cluster features
        cluster_algo: Sklearn cluster algorithm name
        **kwargs: Dict with all cluster algorithm parameters

    Returns:
        Array of labels assigned to the input data

    """
    if cluster_algo == ClusterAlgo.KMEANS:
        return cluster.KMeans(**kwargs).fit(cluster_features).labels_
    elif cluster_algo == ClusterAlgo.DBSCAN:
        return cluster.DBSCAN(**kwargs).fit(cluster_features).labels_
    elif cluster_algo == ClusterAlgo.MEAN_SHIFT:
        return cluster.MeanShift(**kwargs).fit(cluster_features).labels_
    elif cluster_algo == ClusterAlgo.WARD or cluster_algo == ClusterAlgo.AGGLOMERATIVE_CLUSTERING:
        return cluster.AgglomerativeClustering(**kwargs).fit(cluster_features).labels_
    elif cluster_algo == ClusterAlgo.SPECTRAL_CLUSTERING:
        return cluster.SpectralClustering(**kwargs).fit(cluster_features).labels_
    elif cluster_algo == ClusterAlgo.OPTICS:
        return cluster.OPTICS(**kwargs).fit(cluster_features).labels_
    elif cluster_algo == ClusterAlgo.AFFINITY_PROPAGATION:
        return cluster.AffinityPropagation(**kwargs).fit(cluster_features).labels_
    elif cluster_algo == ClusterAlgo.BIRCH:
        return cluster.Birch(**kwargs).fit(cluster_features).labels_
    elif cluster_algo == ClusterAlgo.GAUSSIAN_MIXTURE:
        return mixture.GaussianMixture(**kwargs).fit_predict(cluster_features)
