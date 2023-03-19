from typing import Optional, List
from plotly import express as px
import plotly.graph_objs as go
import numpy as np


def plot_figure(cluster_features: np.ndarray, cluster_labels=None) -> go.Figure:
    """

    Args:
        cluster_features: 2 or 3 dimensional numerical features
        cluster_labels: The predicted cluster label for each data point of cluster_features

    Returns:
        Plotly Figure

    """
    if cluster_features.shape[1] == 3:
        return px.scatter_3d(x=cluster_features[:,0], y=cluster_features[:,1], z=cluster_features[:,2],
                             color=cluster_labels, color_continuous_scale=px.colors.diverging.Portland)
    else:
        return px.scatter(x=cluster_features[:,0], y=cluster_features[:,1], color=cluster_labels, template="plotly_dark",
                 color_continuous_scale=px.colors.diverging.Portland)
