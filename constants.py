from data_classes import ClusterAlgo, DatasetName

DEFAULT_DATASET_N_SAMPLES = 500
MAX_PLOTS_PER_ROW = 3
CLUSTER_ALGORITHMS = [ClusterAlgo.KMEANS, ClusterAlgo.AFFINITY_PROPAGATION, ClusterAlgo.MEAN_SHIFT,
                      ClusterAlgo.SPECTRAL_CLUSTERING, ClusterAlgo.WARD, ClusterAlgo.AGGLOMERATIVE_CLUSTERING,
                      ClusterAlgo.DBSCAN, ClusterAlgo.OPTICS, ClusterAlgo.BIRCH, ClusterAlgo.GAUSSIAN_MIXTURE]
DATASET_NAMES = [DatasetName.CIRCLES, DatasetName.Moons, DatasetName.VARIED_VARIANCES, DatasetName.ANISOTROPICLY_DISTRIBUTED, DatasetName.BLOBS, DatasetName.NO_STRUCTURE]

DEFAULT_CLUSTER_ALGO_PARAMS = {
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