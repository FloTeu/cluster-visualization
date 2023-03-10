from data_classes import ClusterAlgo, DatasetName

DEFAULT_DATASET_N_SAMPLES = 500
CLUSTER_ALGORITHMS = [ClusterAlgo.KMEANS, ClusterAlgo.DBSCAN, ClusterAlgo.MEAN_SHIFT, ClusterAlgo.WARD,
                      ClusterAlgo.AGGLOMERATIVE_CLUSTERING, ClusterAlgo.SPECTRAL_CLUSTERING, ClusterAlgo.OPTICS,
                      ClusterAlgo.AFFINITY_PROPAGATION, ClusterAlgo.BIRCH, ClusterAlgo.GAUSSIAN_MIXTURE]
DATASET_NAMES = [DatasetName.BLOBS, DatasetName.CIRCLES, DatasetName.Moons, DatasetName.VARIED_VARIANCES, DatasetName.ANISOTROPICLY_DISTRIBUTED, DatasetName.NO_STRUCTURE]
DEFAULT_N_CLUSTERS = 4

DEFAULT_PARAMS = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}