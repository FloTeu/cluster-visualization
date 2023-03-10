from typing import Optional, List
from plotly import express as px
import numpy as np


def plot_figure(cluster_features: np.ndarray, cluster_labels=None):
    if cluster_features.shape[1] == 3:
        return px.scatter_3d(x=cluster_features[:,0], y=cluster_features[:,1], z=cluster_features[:,2],
                             color=cluster_labels)
    else:
        return px.scatter(x=cluster_features[:,0], y=cluster_features[:,1], color=cluster_labels)
