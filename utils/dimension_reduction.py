import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data_classes import DimensionReductionAlgo

def apply_standardization(pdf: pd.DataFrame) -> np.ndarray:
    """
    Applies standardization, so we get training data with mean = 0 and variance = 1
    """
    standardScaler = StandardScaler()
    return standardScaler.fit_transform(pdf)

def apply_pca(X: np.ndarray, principal_components=2) -> np.ndarray:
    """ Dimension reduction with PCA suitable for linear data
    """
    pca = PCA(n_components=principal_components)
    return pca.fit_transform(X)

def apply_umap(X: np.ndarray, out_dimension=2) -> np.ndarray:
    """ Dimension reduction with UMAP suitable for non-linear data
    """
    dim_reducer = UMAP(n_components=out_dimension, random_state=0)
    return dim_reducer.fit_transform(X)

def apply_tsne(X: np.ndarray, out_dimension=2, prp = 40) -> np.ndarray:
    """ Dimension reduction with t-SNE suitable for non-linear data
    """
    # creae the model
    tsne = TSNE(n_components=out_dimension,
                perplexity=prp,
                random_state=42,
                n_iter=5000,
                n_jobs=-1)
    # apply it to the data
    return tsne.fit_transform(X)

def dimensionality_reduction(X: np.ndarray, dim_red_algo:DimensionReductionAlgo, out_dimension=2) -> np.ndarray:
    """ Performs a dimension reduction
    """
    if dim_red_algo == DimensionReductionAlgo.PCA:
        return apply_pca(X, principal_components=out_dimension)
    elif dim_red_algo == DimensionReductionAlgo.UMAP:
        return apply_umap(X, out_dimension=out_dimension)
    elif dim_red_algo == DimensionReductionAlgo.T_SNE:
        return apply_tsne(X, out_dimension=out_dimension)
    else:
        raise NotImplementedError
