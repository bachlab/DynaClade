# ==========================================
# Author: Ulises Daniel Serratos Hernandez
# Contact: ulises.serratos@ucl.ac.uk
# ==========================================

import umap.umap_
import hdbscan
from sklearn.decomposition import PCA
from utils import *
from predict_where_unknown import prepare_and_load_data
from dask_ml.preprocessing import StandardScaler
from typing import Union, Dict
import os
import argparse
from get_clusters import plot_clusters
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, LocalCluster
from joblib import dump, load
from contextlib import contextmanager
import pandas as pd

print_welcome_message()

# === GENERAL CLUSTERING PARAMETERS === #
PCA_THRESHOLD = 0.95  # 0.95
DATA_COLLECTION_RATE = 100

# === UMAP ===#
NEIGHBORS = 15  # 300
MIN_DIST = 0.1  # 0.001
DENSMAP = False  # True

# === HDBSCAN === #
CLUSTER_SIZE = 100  # best 300; then: 500, 700; old: 500
SAMPLES = None  # best None; then: None, None; old:30
EPSILON = 1  # best 0.0, then: 0.0, 0.0 ; old: 1.0
SINGLE_CLUSTER = True  # True

@contextmanager
def dask_client_pool():
    """Context manager for Dask client pool."""
    cluster = LocalCluster(
        n_workers=None,  # Use all available cores
        threads_per_worker=None,  # Use all available threads per worker
        processes=True,  # Use processes instead of threads
        memory_limit='auto',  # Automatically manage memory
        local_directory=None,  # Use system temp directory
        resources=None,
        silence_logs=False,
        dashboard_address=':8787'
    )
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()

@tictoc
def get_umap_embedding(features_matrix, n_components):
    """Get UMAP embedding with optimized memory usage and error handling."""
    try:
        n_samples = features_matrix.shape[0]
        n_components = min(n_components, n_samples - 2)  # Ensure k < N

        # Convert to float32 to reduce memory usage
        if isinstance(features_matrix, (dd.DataFrame, pd.DataFrame)):
            features_matrix = features_matrix.astype(np.float32)
        elif isinstance(features_matrix, da.Array):
            features_matrix = features_matrix.astype(np.float32)

        # Handle different input types
        if isinstance(features_matrix, dd.DataFrame):
            features_matrix = features_matrix.compute().values
        elif isinstance(features_matrix, da.Array):
            features_matrix = features_matrix.compute()

        # Use delayed computation for UMAP
        embedding = umap.UMAP(
            n_components=n_components,
            n_neighbors=NEIGHBORS,
            min_dist=MIN_DIST,
            random_state=42,
            low_memory=True,
            n_jobs=-1,
            verbose=True
        ).fit_transform(features_matrix)

        return da.from_array(embedding, chunks=(10000, n_components))  # Specify chunk size

    except Exception as e:
        print_color(f"\nError in UMAP embedding: {str(e)}\n", bold_color.RED)
        raise

@tictoc
def get_hdbscan_clusters(clusterer, embedding):
    """Get HDBSCAN clusters with optimized memory usage."""
    try:
        if isinstance(embedding, da.Array):
            embedding = embedding.compute()
        
        labels = clusterer.fit_predict(embedding)
        return da.from_array(labels)
    except Exception as e:
        print_color(f"\nError in HDBSCAN clustering: {str(e)}\n", bold_color.RED)
        raise

@tictoc
def process_data(features_raw, pca_threshold, plot_correlations=False):
    """Process data with optimized memory usage and Dask integration."""
    try:
        # Convert to float32 to reduce memory usage
        if isinstance(features_raw, (dd.DataFrame, pd.DataFrame)):
            features_raw = features_raw.astype(np.float32)
        elif isinstance(features_raw, da.Array):
            features_raw = features_raw.astype(np.float32)

        # Standardize the data using Dask
        scaler = StandardScaler()
        if isinstance(features_raw, dd.DataFrame):
            features_scaled = features_raw.map_partitions(
                lambda df: pd.DataFrame(
                    scaler.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )
            )
        else:
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features_raw),
                columns=features_raw.columns,
                index=features_raw.index
            )

        # Sort index for consistency
        features_scaled = features_scaled.sort_index()

        # Perform PCA with optimized settings
        if features_scaled.shape[0] > 10000: 
            print_color("\nLarge dataset detected, using randomized SVD solver\n", bold_color.YELLOW)
            # First fit with a subset to estimate n_components
            subset_size = min(10000, features_scaled.shape[0])
            subset_indices = np.random.choice(features_scaled.shape[0], subset_size, replace=False)
            subset_data = features_scaled.iloc[subset_indices]
            
            temp_pca = PCA(n_components=None, svd_solver='full')
            temp_pca.fit(subset_data)
            
            # Calculate number of components needed
            cumulative_variance_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.searchsorted(cumulative_variance_ratio, pca_threshold) + 1
            
            # Now fit final PCA with estimated n_components
            pca_model = PCA(n_components=int(n_components), svd_solver='randomized')
        else:
            pca_model = PCA(n_components=pca_threshold)

        pca_transformed = pca_model.fit_transform(features_scaled)

        print_color(
            f"\n {pca_model.n_components_} PCA components > "
            + f"{round(pca_threshold*100)} % variance \n",
            bold_color.GREEN,
        )

        return features_scaled, pca_transformed, pca_model.n_components_

    except Exception as e:
        print_color(f"\nError in data processing: {str(e)}\n", bold_color.RED)
        raise

@tictoc
def perform_clustering(
    embedding: Union[np.ndarray, pd.DataFrame, da.Array],
    cluster_size: int,
    samples: int,
    single_cluster: bool,
    approx_min_span_tree: bool,
    epsilon: float = 0.0,
) -> tuple:
    """Perform clustering with optimized settings and error handling."""
    try:
        hdbscan_params = {
            "min_cluster_size": cluster_size,
            "min_samples": samples,
            "allow_single_cluster": single_cluster,
            "cluster_selection_epsilon": epsilon,
            "approx_min_span_tree": approx_min_span_tree,
        }

        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        labels = get_hdbscan_clusters(clusterer, embedding)

        if isinstance(labels, da.Array):
            labels = labels.compute()
        
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

    except Exception as e:
        print_color(f"\nError in clustering: {str(e)}\n", bold_color.RED)
        raise


def get_subcluster_data(features_da, args, current_cluster_label):

    # # Get UMAP embedding
    # embedding = get_umap_embedding(
    #     features_da, args.n_components
    # )

    # embedding = dd.from_dask_array(embedding)
    embedding = features_da.copy()
    scores = None

    if args.n_components:
        args.n_components = str(args.n_components) + " components"

    # Perform clustering
    approx_min_span_tree = False
    labels, _, _, _ = perform_clustering(
        # embedding,
        embedding,
        args.cluster_size,
        args.samples,
        args.single_cluster,
        approx_min_span_tree,
        epsilon=args.epsilon,
    )

    print_color(f"\nScores: {scores}\n", bold_color.DARKCYAN)

    # Plot and save results
    figure1 = plot_clusters(embedding=embedding.compute(), labels=labels, title=f"Clusters {args.basename}", show=True, marker_size=40)

    features_da["subcluster_label"] = features_da[args.labels_column].astype(str) + "_" + labels.astype(str)
    
    output_dir = os.path.join(os.path.normpath(args.output_dir), "subclusters")
    os.makedirs(output_dir, exist_ok=True)
    # Save results efficiently
    features_da.compute().to_csv(output_dir + args.path.replace(".csv", f"_clusters_all_data_{args.basename}_{current_cluster_label}.csv"), index=False)
    figure1.savefig(output_dir + args.path.replace(".csv", f"_clusters_{args.basename}_{current_cluster_label}.png"), dpi=300)

    return features_da


@tictoc
def main(args) -> None:
    """Main function with optimized data handling and error handling."""
    try:
        basename = "_".join(os.path.basename(args.path).split("_")[:-2])
        args.basename = basename

        meta_columns = ["ppid", "session", "trial", "time"]
        
        with dask_client_pool() as client:
            # Read data efficiently with Dask using optimized settings
            data = dd.read_csv(
                args.path, 
                blocksize='25MB',
                assume_missing=True,
                dtype_backend='numpy_nullable',
                dtype={'ppid': 'int64', 'session': 'int64', 'trial': 'int64', args.labels_column: 'int64'}
            )
                
            
            try:

                # Get list of cluster labels to subcluster - MOVED HERE after features_da is created
                if args.group_size_threshold_to_subcluster:
                    # Compute value counts once and store
                    value_counts = data[args.labels_column].value_counts().compute()
                    max_frequency = value_counts.max()
                    normalized_counts = value_counts / max_frequency
                    normalized_threshold = args.group_size_threshold_to_subcluster
                    args.group_label_list_to_subcluster = list(normalized_counts[normalized_counts > normalized_threshold].index)
                
                print_color(f"\nGroup label list to subcluster: {args.group_label_list_to_subcluster}\n", bold_color.DARKCYAN)

                features_embedding = data.iloc[:, ~data.columns.isin(meta_columns + [args.labels_column])].copy()
                labels = data[args.labels_column].copy().astype(int)

                import pdb; pdb.set_trace()

                grouped_data = data.groupby(args.labels_column)
                for current_cluster_label, group in grouped_data:
                    import pdb; pdb.set_trace()
                    if current_cluster_label in args.group_label_list_to_subcluster:
                        features_embedding_with_labels = get_subcluster_data(group, args, current_cluster_label)

                print_color(f"\nSubclusters saved in {args.output_dir}\n", bold_color.BLUE)

            except Exception as e:
                print_color(f"\nError during merge operation: {str(e)}", bold_color.RED)
                raise
                
            features_embedding_with_labels.to_csv(args.output_dir + args.path.replace(".csv", f"_clusters_all_data_{args.basename}_{current_cluster_label}.csv"), index=False)

    except Exception as e:
        print_color(f"\nError in main execution: {str(e)}\n", bold_color.RED)
        raise

def parse_arguments():
    """Parse command line arguments with improved validation."""
    parser = argparse.ArgumentParser(description='Get subclusters')
    parser.add_argument('--path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--group-size-threshold-to-subcluster', type=float, required=True, help='Group size threshold to subcluster')
    parser.add_argument('--group-label-list-to-subcluster', type=str, default=None, help='Group label list to subcluster passed as a csv file')
    parser.add_argument('--pca-threshold', type=float, default=PCA_THRESHOLD, help='PCA threshold')
    parser.add_argument('--cluster-size', type=int, default=CLUSTER_SIZE, help='Cluster size')
    parser.add_argument('--samples', type=int, default=SAMPLES, help='Samples')
    parser.add_argument('--epsilon', type=float, default=EPSILON, help='Epsilon')
    parser.add_argument('--single-cluster', type=bool, default=SINGLE_CLUSTER, help='Single cluster')
    parser.add_argument('--labels-column', type=str, default='labels', help='Label column')
    parser.add_argument('--output-dir', type=str, default='./', help='Output directory')
    parser.add_argument('--feature-matrix-file', type=str, default=None, help='Feature matrix file')

    args = parser.parse_args()

    if args.group_label_list_to_subcluster and args.group_size_threshold_to_subcluster:
        raise ValueError("only one of group_label_list_to_subcluster or group_size_threshold_to_subcluster can be provided")

    return args

if __name__ == "__main__":
    os.system("cls||clear")
    args = parse_arguments()
    main(args)
