from sklearn import cluster, mixture

from data_classes import ClusterAlgo


def get_cluster_labels(X, cluster_algo: ClusterAlgo, **kwargs):
    if cluster_algo == ClusterAlgo.KMEANS:
        return cluster.KMeans(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.DBSCAN:
        return cluster.DBSCAN(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.MEAN_SHIFT:
        return cluster.MeanShift(**kwargs).fit(X).labels_
    elif cluster_algo == ClusterAlgo.WARD or cluster_algo == ClusterAlgo.AGGLOMERATIVE_CLUSTERING:
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
