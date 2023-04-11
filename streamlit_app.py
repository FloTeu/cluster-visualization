from typing import List

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

from constants import CLUSTER_ALGORITHMS, DATASET_NAMES, MAX_PLOTS_PER_ROW
from utils.preparation import get_default_dataset_points, get_cluster_algo_parameters, add_user_data_input_listener, split_list
from utils.modeling import get_cluster_labels
from utils.visualization import plot_figure
from utils.dimension_reduction import apply_standardization, dimensionality_reduction
from data_classes import DatasetName, DimensionReductionAlgo

st.set_page_config(
    page_title="Cluster Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """
    The main function
    """

    st.sidebar.write("Either upload a CSV file, or use one of the example datasets")
    is_3d = st.sidebar.checkbox("Use 3D Features", value=False)
    st.sidebar.subheader("1. Upload a CSV File")
    uploaded_file = st.sidebar.file_uploader("File Uploader", type="csv")
    if uploaded_file is not None:
        dataset_name = DatasetName.CUSTOM.value
        df = pd.read_csv(uploaded_file)
        default_dataset_points = df.to_numpy()
        if default_dataset_points.shape[1] not in [2,3]:
            dim_red_algo: DimensionReductionAlgo = st.sidebar.selectbox("Choose an Algo for Dimension Reduction:", [DimensionReductionAlgo.PCA.value, DimensionReductionAlgo.UMAP.value, DimensionReductionAlgo.T_SNE.value])
            do_standardization = st.sidebar.checkbox("Standardize Data", value=True)
            if do_standardization:
                default_dataset_points = apply_standardization(default_dataset_points)
            default_dataset_points = dimensionality_reduction(default_dataset_points, dim_red_algo, out_dimension=3 if is_3d else 2)
    else:
        st.sidebar.subheader("2. Example Dataset")
        dataset_name = st.sidebar.selectbox(
            "Default Data", [dn.value for dn in DATASET_NAMES])
        default_dataset_points = get_default_dataset_points(dataset_name, is_3d)

    cluster_features = add_user_data_input_listener(default_dataset_points)
    # normalize dataset for easier parameter selection
    cluster_features_scaled = StandardScaler().fit_transform(cluster_features)

    st.title("Cluster Visualization")
    st.write("Streamlit application of the [sklearn cluster comparison](https://scikit-learn.org/stable/modules/clustering.html) page. \n"
             "Feel free to upload your own custom data or to play around with one of the example datasets. The parameters of the cluster algorithm can be updated interactively and three-dimensional visualizations are also possible.")
    st.write("[Source code](https://github.com/FloTeu/cluster-visualization)")

    # Cluster Algo
    cluster_algos: List[str] = st.sidebar.multiselect(
        "Cluster Algorithms", [ca.value for ca in CLUSTER_ALGORITHMS], [CLUSTER_ALGORITHMS[0]])
    for cluster_algo_splitted_list in split_list(cluster_algos, MAX_PLOTS_PER_ROW):
        display_cols = st.columns(MAX_PLOTS_PER_ROW)
        for i, cluster_algo_str in enumerate(cluster_algo_splitted_list):
            st.sidebar.title(cluster_algo_str)
            cluster_algo_kwargs = get_cluster_algo_parameters(cluster_algo_str, cluster_features_scaled, dataset_name)
            cluster_labels = get_cluster_labels(cluster_features_scaled, cluster_algo_str, **cluster_algo_kwargs)

            # Visualize clustering results
            display_cols[i].subheader(cluster_algo_str)
            fig = plot_figure(cluster_features,cluster_labels)

            # prevent that a single visualization does not take the whole width
            use_container_width = len(cluster_algos) > 1
            display_cols[i].plotly_chart(fig, use_container_width=use_container_width)


if __name__ == "__main__":
    main()
