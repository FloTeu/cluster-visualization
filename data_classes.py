from enum import Enum

class ClusterAlgo(str, Enum):
    KMEANS = "KMeans"
    DBSCAN = "DBSCAN"
    MEAN_SHIFT = "MeanShift"
    WARD = "Ward"
    AGGLOMERATIVE_CLUSTERING = "Agglomerative Clustering"
    SPECTRAL_CLUSTERING = "Spectral Clustering"
    OPTICS = "OPTICS"
    AFFINITY_PROPAGATION = "Affinity Propagation"
    BIRCH = "BIRCHS"
    GAUSSIAN_MIXTURE = "Gaussian Mixture"


class DatasetName(str, Enum):
    BLOBS = "Blobs"
    CIRCLES = "Circles"
    Moons = "Moons"
    NO_STRUCTURE = "No Structure"
    VARIED_VARIANCES = "Varied Variances"
    ANISOTROPICLY_DISTRIBUTED = "Anisotropicly Distributed"
    CUSTOM = "Custom"

class DimensionReductionAlgo(str, Enum):
    PCA="Principal Component Analysis"
    T_SNE="t-SNE"
    UMAP="UMAP"