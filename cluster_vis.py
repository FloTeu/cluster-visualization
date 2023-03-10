from typing import List

import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

import utils.preparation
from constants import CLUSTER_ALGORITHMS, DATASET_NAMES
import utils
from utils.preparation import get_dataset_points, get_cluster_algo_parameters, get_cluster_features
from utils.modeling import get_cluster_labels
from utils.visualization import plot_figure

# TODO: Customize icon
st.set_page_config(
    page_title="ikneed",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)



def main():
    """
    The main function
    """
    is_3d = st.sidebar.checkbox("Use 3D Features", value=False)

    dataset_name = st.sidebar.selectbox(
        "Default Data", [dn.value for dn in DATASET_NAMES])

    dataset_points = get_dataset_points(dataset_name, is_3d)
    cluster_features = get_cluster_features(dataset_points)

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(cluster_features)

    # Cluster Algo
    cluster_algos: List[str] = st.sidebar.multiselect(
        "Cluster Algorithms", [ca.value for ca in CLUSTER_ALGORITHMS], [CLUSTER_ALGORITHMS[0]])
    display_cols = st.columns(len(cluster_algos))

    for i, cluster_algo_str in enumerate(cluster_algos):
        st.sidebar.title(cluster_algo_str)
        cluster_algo_kwargs = get_cluster_algo_parameters(cluster_algo_str, dataset_name)
        cluster_labels = get_cluster_labels(X, cluster_algo_str, **cluster_algo_kwargs)

        # plot the figure
        display_cols[i].subheader(cluster_algo_str)

        fig = plot_figure(cluster_features,cluster_labels)
        # prevent that on visualization does not take the whole width
        use_container_width = len(cluster_algos) > 1
        display_cols[i].plotly_chart(fig, use_container_width=use_container_width)





if __name__ == "__main__":
    main()
