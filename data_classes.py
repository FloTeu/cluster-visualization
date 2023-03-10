from enum import Enum

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


class DatasetName(str, Enum):
    BLOBS = "Blobs"
    CIRCLES = "Circles"
    Moons = "Moons"
    NO_STRUCTURE = "No Structure"
    VARIED_VARIANCES = "Varied Variances"
    ANISOTROPICLY_DISTRIBUTED = "Anisotropicly Distributed"