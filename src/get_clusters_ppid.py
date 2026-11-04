# ==========================================
# Author: Ulises Daniel Serratos Hernandez
# Contact: ulises.serratos@ucl.ac.uk
# ==========================================

import umap.umap_
import hdbscan
from sklearn.decomposition import PCA
from utils import *
from dask_ml.preprocessing import StandardScaler
from typing import Union, Dict
import os
from get_clusters import plot_clusters

print_welcome_message()

# === GENERAL CLUSTERING PARAMETERS === #
PCA_THRESHOLD = 0.95  # 0.95
DATA_COLLECTION_RATE = 100

# === UMAP ===#
NEIGHBORS = 2  # 300
MIN_DIST = 0.1  # 0.001
DENSMAP = False  # True

# === HDBSCAN === #
CLUSTER_SIZE = 2  # best 300; then: 500, 700; old: 500
SAMPLES = None  # best None; then: None, None; old:30
EPSILON = 1.5  # best 0.0, then: 0.0, 0.0 ; old: 1.0
SINGLE_CLUSTER = True  # True


@tictoc
def get_umap_embedding(features_matrix, n_components):

    n_samples = features_matrix.shape[0]
    n_components = min(n_components, n_samples - 2)  # Ensure k < N

    embedding = np.array(
        umap.UMAP(
            n_components=n_components,
            n_neighbors=NEIGHBORS,
            min_dist=MIN_DIST,
            random_state=42,
            # densmap=DENSMAP,
            # low_memory=True,
            # n_jobs=-1,
        ).fit_transform(features_matrix)
    )

    return embedding

@tictoc
def get_hdbscan_clusters(clusterer, embedding):
    labels = np.array(clusterer.fit_predict(embedding))
    return labels


@tictoc
def process_data(features_raw, pca_threshold, plot_correlations=False):
    # Standardize the data
    features_scaled = StandardScaler().fit_transform(features_raw)

    # Compute correlation matrix
    features_scaled = features_scaled.sort_index()

    # Perform Principal Component Analysis
    pca_model = PCA(n_components=pca_threshold)
    pca_transformed = pca_model.fit_transform(features_scaled)


    print_color(
        f"\n {pca_model.n_components_} PCA components > "
        + f"{round(pca_threshold*100)} % variance \n",
        bold_color.GREEN,
    )

    return features_scaled, pca_transformed, pca_model.n_components_

@tictoc
def perform_clustering(
    embedding: Union[np.ndarray, pd.DataFrame],
    cluster_size: int,
    samples: int,
    single_cluster: bool,
    approx_min_span_tree: bool,
    epsilon: float = 0.0,
) -> np.ndarray:
    hdbscan_params = {
        "min_cluster_size": cluster_size,
        "min_samples": samples,
        "allow_single_cluster": single_cluster,
        "cluster_selection_epsilon": epsilon,
        "approx_min_span_tree": approx_min_span_tree,
    }


    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    labels = get_hdbscan_clusters(clusterer, embedding)

    labels_clustered = labels[labels >= 0]

    percentage_labeled = round(
        len(labels_clustered) / len(embedding) * 100, 2
    )
  
    print_color(f"\nlabels: {set(labels)}\n", bold_color.DARKCYAN)
    print_color(
        f"\nlabeled data: {percentage_labeled}% | "
        + f"non-labeled data: {round(100-percentage_labeled,2)}%\n",
        bold_color.GREEN,
    )

    return labels, clusterer, percentage_labeled, hdbscan_params


@tictoc
def main(
    path: str,
    pca_threshold: float = PCA_THRESHOLD,
    cluster_size: int = CLUSTER_SIZE,
    samples: int = SAMPLES,
    epsilon: float = EPSILON,
    single_cluster: bool = SINGLE_CLUSTER,
) -> None:
    
    try:
        basename = "_".join(os.path.basename(path).split("_")[:-2])
        data = pd.read_csv(path)
        features = data.drop(columns=["condition", "observations"])
        print(f'features: {features}')



        features_da, _, n_components = process_data(
            features, pca_threshold, plot_correlations=False
        )

        # === DIMENSIONALITY REDUCTION UMAP EMBEDING === #
        embedding = get_umap_embedding(
            features_da, n_components
        )

        embedding = pd.DataFrame(embedding)
            

        scores = None
        if n_components:
            n_components = str(n_components) + " components"

        # === CLUSTERING === #

        approx_min_span_tree = False

        (
            labels,
            _,
            _,
            _,
        ) = perform_clustering(
            embedding.iloc[:, 2:],
            cluster_size,
            samples,
            single_cluster,
            approx_min_span_tree,
            epsilon=epsilon,
        )

        print_color(f"\nScores: {scores}\n", bold_color.DARKCYAN)

        figure1 = plot_clusters(embedding=embedding, labels=labels, title=f"Clusters {basename}", show=True, marker_size=40)

        data['labels'] = labels
        data_ppid_label = data[['condition', 'labels']]
        data.to_csv(path.replace(".csv", f"_clusters_all_data_{basename}.csv"), index=False)
        data_ppid_label.to_csv(path.replace(".csv", f"_clusters_only_{basename}.csv"), index=False)
        figure1.savefig(path.replace(".csv", f"_clusters_{basename}.png"), dpi=300)
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    os.system("cls||clear")

    os_system, _ = welcome()

    path = input("\nPlease provide input directory/file name: \n -> ")
    if path == "":
        path = r"D:\repos\tmp\feature_matrix_curated\syllable_counts_output\ppid_vr\original\ppid_vr_stats_medians.csv"

    main(
        path=path
    )
