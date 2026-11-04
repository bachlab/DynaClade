# ==========================================
# Author: Ulises Daniel Serratos Hernandez
# Contact: ulises.serratos@ucl.ac.uk
# ==========================================

import os 
import re
import time
import pdb
# import umap.umap_
from joblib import dump, load, Parallel

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.decomposition import PCA
from dask_ml.preprocessing import StandardScaler
import seaborn as sns
from dask import dataframe as dd
import dask.array as da
import dask.array as da
from sklearn.preprocessing import StandardScaler
import json
from dask.distributed import Client, LocalCluster

from utils import print_welcome_message, print_color, bold_color, tictoc, list_directory_items, create_directory, logger, logging, Union, create_pastel_rainbow_cmap, Dict, ask_question_options, welcome

# cluster = LocalCluster(n_workers=None, threads_per_worker=None, processes=True, memory_limit='auto', local_directory=None, resources=None, silence_logs=False, dashboard_address=':8787')
# client = Client(cluster)

print_welcome_message()

# === GENERAL CLUSTERING PARAMETERS === #
PCA_THRESHOLD = 0.95  # 0.95
DATA_COLLECTION_RATE = 200

# === UMAP ===#
NEIGHBORS = 300  # 300
MIN_DIST = 0.0  # 0.001
DENSMAP = False  # True

# === HDBSCAN === #
CLUSTER_SIZE = 300  # best 300; then: 500, 700; old: 500
SAMPLES = None  # best None; then: None, None; old:30
EPSILON = 0.0  # best 0.0, then: 0.0, 0.0 ; old: 1.0
SINGLE_CLUSTER = True  # True


def plot_histogram(
    x,
    y,
    fig_num: int = None,
    xlabel: str = "Princial Components",
    ylabel: str = "Variance %",
    title: str = "Principal Component Variance",
    show: bool = False,
):
    plt.figure(fig_num) if fig_num else plt.figure()
    plt.bar(x, y, color="gold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.title(title)
    if show:
        plt.show()


def plot_histograms(features_raw, bins=100, figsize=(20, 15), show=False):
    axes = features_raw.compute().hist(bins=bins, figsize=figsize)
    for ax in axes.flatten():
        ax.set_title(ax.get_title(), fontsize=5, weight="bold")
        ax.set_xticklabels([])  # remove x-axis tick labels
        ax.set_yticklabels([])  # remove y-axis tick labels
        ax.set(xlabel="", ylabel="")  # remove axis labels
    if show:
        plt.show()


def plot_heatmap(corr, title, cmap="plasma", annot=False, show=False):
    plt.figure()
    sns.heatmap(corr, cmap=cmap, annot=annot)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title + " correlation matrix")
    if show:
        plt.show()


def plot_clusters(
    embedding,
    labels=None,
    title=None,
    marker_size=2,
    marker_size_fixed=True,
    show=False,
) -> None:
    import numpy as np
    import pandas as pd

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    embedding = (
        pd.DataFrame(embedding)
        if isinstance(embedding, np.ndarray)
        else embedding
    )
    labels = (
        pd.DataFrame(labels)
        if labels is not None and isinstance(labels, np.ndarray)
        else labels
    )

    if labels is None:
        s = marker_size if marker_size_fixed else embedding.iloc[:, 3] * 2
        ax.scatter(
            embedding.iloc[:, 0],
            embedding.iloc[:, 1],
            embedding.iloc[:, 2],
            s=s,
        )
    else:
        labels = pd.DataFrame(labels)
        unique_labels = labels.squeeze().unique()
        num_labels = len(unique_labels)
        plt.gca().set_prop_cycle(
            plt.cycler("color", plt.cm.jet(np.linspace(0, 1, num_labels))))

        for i in unique_labels:
            index = labels == i
            s = (
                marker_size
                if marker_size_fixed
                else embedding.loc[index.values, 3] * 2
            )
            color, alpha = ("gray", 0.03) if i == -1 else (None, 0.5)
            ax.scatter(
                embedding.loc[index.values, 0],
                embedding.loc[index.values, 1],
                embedding.loc[index.values, 2],
                s=s,
                label=i,
                c=color,
                alpha=alpha,
            )

    ax.set(xlabel="X axis", ylabel="Y axis", zlabel="Z axis", title=title)
    # plt.legend()

    if show:
        plt.show()

    return fig


@tictoc
def plot_sns_jointplot(x, y, labels, show=True) -> None:
    sns.jointplot(x=x, y=y, hue=labels, kind="kde")
    plt.legend([], [], frameon=False)
    plt.show() if show else None


def create_dask_df_from_path(path: str, subsample_data_rate: int = None) -> dd:
    # Split the path into components and extract numbers from the last three components
    ppid, session, trial = map(
        int,
        re.findall(
            r"\d+", "/".join(os.path.normpath(path).split(os.sep)[-3:])
        ),
    )

    # Load the dataset lazily using dask with proper dtype_backend
    df = dd.read_csv(path, dtype_backend='numpy_nullable')
    
    if subsample_data_rate:
        # Sample using modulo division on the index
        df = df.map_partitions(lambda pdf: pdf.iloc[::subsample_data_rate]) 
        # Compute length after sampling
        df = df.persist()  # Cache the sampled dataframe
    
    length = len(df)

    # Calculate divisions based on the length
    chunk_size = max(1, length // df.npartitions)

    # Create arrays for each column
    ppid_arr = da.full(length, ppid, chunks=chunk_size)
    session_arr = da.full(length, session, chunks=chunk_size)
    trial_arr = da.full(length, trial, chunks=chunk_size)
    time_arr = da.arange(length, chunks=chunk_size) / DATA_COLLECTION_RATE

    # Stack arrays vertically
    stacked_arrays = da.vstack([ppid_arr, session_arr, trial_arr, time_arr]).T

    # Create a pandas DataFrame as metadata
    meta = pd.DataFrame({
        'ppid': pd.Series(dtype='int64'),
        'session': pd.Series(dtype='int64'),
        'trial': pd.Series(dtype='int64'),
        'time': pd.Series(dtype='float64')
    })

    # Create the dask DataFrame using from_dask_array
    ddf_info = dd.from_dask_array(
        stacked_arrays,
        columns=['ppid', 'session', 'trial', 'time'],
        meta=meta
    )

    # Create proper index
    ddf_info = ddf_info.reset_index(drop=True)
    
    # Repartition to match the original number of partitions
    ddf_info = ddf_info.repartition(npartitions=df.npartitions)

    if subsample_data_rate:
        # Apply the same sampling to ddf_info
        ddf_info = ddf_info.map_partitions(lambda pdf: pdf.iloc[::subsample_data_rate]) 
        ddf_info = ddf_info.persist()  # Cache the sampled dataframe

    return ddf_info, {
        "ppid": ppid,
        "session": session,
        "trial": trial,
        "len": length,
    }


@tictoc
def get_all_data(path: str, search_pattern: str | list, subsample_data_rate: int = None, chunk_size: str = '25MB', curated_list: list = None) -> dd:
    files_list = list_directory_items(
        file_or_directory=path, recursive=True, search_pattern=search_pattern
    )
  
    all_dfs = []
    all_trial_infos = []

    if curated_list:
        curated_list = pd.read_csv(curated_list)
        curated_list = list(curated_list.groupby(["ppid", "session", "trial"]).groups.keys())
    
    for file_path in files_list["items"]:
        # Extract metadata from file path
        ppid, session, trial = map(
            int,
            re.findall(
                r"\d+", "/".join(os.path.normpath(file_path).split(os.sep)[-3:])
            ),
        )
        
        if curated_list:
            if (ppid, session, trial) not in curated_list:
                print_color(f"|{__file__}| Skipping {file_path} because it is not in the curated list", bold_color.YELLOW)
                continue
        
        # Read each file as a pandas DataFrame first to get exact length
        df = pd.read_csv(file_path)
        original_length = len(df)
        
        if subsample_data_rate:    
            # Calculate sampled indices
            sampled_indices = list(range(0, original_length, subsample_data_rate))
            sampled_length = len(sampled_indices)
            
            # Sample the DataFrame
            df = df.iloc[sampled_indices]
            
            # Create trial info with matching length
            trial_info = pd.DataFrame({
                'ppid': [ppid] * sampled_length,
                'session': [session] * sampled_length,
                'trial': [trial] * sampled_length,
                'time': np.array(sampled_indices) / DATA_COLLECTION_RATE
            })
        else:
            # Create trial info with original length
            trial_info = pd.DataFrame({
                'ppid': [ppid] * original_length,
                'session': [session] * original_length,
                'trial': [trial] * original_length,
                'time': np.arange(original_length) / DATA_COLLECTION_RATE
            })
        
        # Convert both to dask DataFrames with matching partitions
        npartitions = max(1, len(df) // 1000)  # Adjust partition size as needed
        df = dd.from_pandas(df, npartitions=npartitions)
        trial_info = dd.from_pandas(trial_info, npartitions=npartitions)
        
        all_dfs.append(df)
        all_trial_infos.append(trial_info)
    
    # Concatenate all DataFrames
    final_df = dd.concat(all_dfs, axis=0)
    final_trial_info = dd.concat(all_trial_infos, axis=0)
    
    # Reset indices to ensure alignment
    final_df = final_df.reset_index(drop=True)
    final_trial_info = final_trial_info.reset_index(drop=True)
    
    return final_df, final_trial_info


def get_mode(path: str, embedding: bool = False) -> str:
    """ """
    if embedding:
        return "embedding"
    elif os.path.isdir(path):
        return "directory"
    elif os.path.isfile(path):
        return "file"
    else:
        logger.error(
            f'\n===> ERROR: directory/file not found. Input: "{path}"\n'
        )
        exit()


@tictoc
def prepare_data(path: str, search_pattern: str | list, mode: str, rapids_cuml: bool = False, subsample_data_rate: int = None, curated_list: list = None) -> dd:
    """ """
    import pandas as pd
    data = features = parent_path = data_with_info = None

    if mode == "directory":
        parent_path = path
        # Set a reasonable chunk size for Dask operations
        chunk_size = '25MB'  # Adjust this value based on your available memory
        
        data, trial_info = get_all_data(
            path=path, 
            search_pattern=search_pattern, 
            subsample_data_rate=subsample_data_rate,
            chunk_size=chunk_size,
            curated_list=curated_list
        )
        
        if rapids_cuml:
            import cudf
            import numpy as np
            # Convert data to cuDF DataFrame
            data = data.map_partitions(lambda df: df.astype(np.float32))
            data = data.compute()  # Compute the Dask DataFrame first
            
            # Convert trial_info to cuDF DataFrame
            trial_info = trial_info.compute()  # Compute the Dask DataFrame first
            
            # Ensure indices match before conversion to cuDF
            if not data.index.equals(trial_info.index):
                data = data.reset_index(drop=True)
                trial_info = trial_info.reset_index(drop=True)
            
            # Fill null values with mean of each column in data
            for column in data.columns:
                data[column] = data[column].fillna(data[column].mean())

            # Convert to cuDF with preserved indices
            data = cudf.DataFrame(data)
            trial_info = cudf.DataFrame(trial_info)  # trial_info shouldn't have nulls
            
            if 'time' in data.columns:
                trial_info = trial_info.drop(columns=['time'])
            
            # Verify indices still match
            if not data.index.equals(trial_info.index):
                data = data.reset_index(drop=True)
                trial_info = trial_info.reset_index(drop=True)
            
            data_with_info = cudf.concat([trial_info, data], axis=1)
        else:
            # Keep data lazy with dask
            if 'time' in data.columns:
                trial_info = trial_info.drop(columns=['time'])
            data_with_info = dd.concat([trial_info, data], axis=1)
        features = data.iloc[:, 1:]
    elif mode in ["file", "embedding"]:
        parent_path = os.path.abspath(os.path.join(path, os.pardir))
        # Read file lazily with optimized settings
        chunk_size = '25MB'  # Adjust based on your memory constraints
        if subsample_data_rate:
            data = dd.read_csv(
                path,
                blocksize=chunk_size,
                assume_missing=True,
                dtype_backend='numpy_nullable'
            )
            # Sample efficiently using modulo on the index
            data = data.map_partitions(lambda df: df.iloc[::subsample_data_rate])
        else:
            data = dd.read_csv(
                path,
                blocksize=chunk_size,
                assume_missing=True,
                dtype_backend='numpy_nullable'
            )
        features = data.iloc[:, 4:-1]
        data_with_info = data.iloc[:, :-1]
        
        if rapids_cuml:
            import cudf
            # Convert to cudf DataFrame for RAPIDS
            data_with_info = data_with_info.compute()  # Compute first
            features = features.compute()  # Compute first
            
            # Fill null values with mean of each column
            for column in data_with_info.columns:
                data_with_info[column] = data_with_info[column].fillna(data_with_info[column].mean())
            for column in features.columns:
                features[column] = features[column].fillna(features[column].mean())
            
            # Reset indices before conversion
            data_with_info = data_with_info.reset_index(drop=True)
            features = features.reset_index(drop=True)
            
            # Convert to cuDF
            data_with_info = cudf.DataFrame(data_with_info)
            features = cudf.DataFrame(features)
            
        if mode == "embedding":
            logger.info("|main| embedding was passed as input")
    else:
        logger.error(
            f'\n===> ERROR: directory/file not found. Input: "{path}"\n'
        )
        exit()

    # Ensure consistent column names
    if data_with_info is not None:
        data_with_info.columns = ["ppid", "session", "trial", "time"] + list(
            data_with_info.columns[4:]
        )
   
    # Instead of checking for duplicates, just ensure index is reset
    if isinstance(features, dd.DataFrame):
        features = features.reset_index(drop=True)
    elif isinstance(features, pd.DataFrame):
        features = features.reset_index(drop=True)
    elif isinstance(features, cudf.DataFrame):
        features = features.reset_index(drop=True)
    
    features = features.dropna()

    if features.isnull().values.any():
        print("NaNs are present in the data")

    return data, features, parent_path, data_with_info

@tictoc
def get_umap_embedding(features_matrix, n_components, n_neighbours, min_dist, densmap, umap_save_path, use_cuml=False):
    if use_cuml:
        print_color("\nUsing RAPIDS cuML UMAP\n", bold_color.GREEN)
        import cupy as cp
        import numpy as np
        from cuml.manifold.umap import UMAP as cuMLUMAP
        from cuml.preprocessing import StandardScaler as cuStandardScaler
        from cuml import PCA as cuPCA
        import cudf

        chunksize = 10000
        print("\nUsing RAPIDS cuML UMAP\n")
        
        # Handle different input types first to get to CuPy array
        if isinstance(features_matrix, da.Array):
            features_matrix = features_matrix.compute()
        
        # Convert to appropriate format based on input type
        if isinstance(features_matrix, cudf.DataFrame):
            print("Input is cuDF DataFrame")
            # Keep as cuDF DataFrame
            pass
        elif isinstance(features_matrix, pd.DataFrame):
            print("Input is pandas DataFrame")
            features_matrix = cudf.DataFrame(features_matrix)
        elif isinstance(features_matrix, np.ndarray):
            print("Input is NumPy array")
            features_matrix = cudf.DataFrame(features_matrix)
        elif isinstance(features_matrix, cp.ndarray):
            print("Input is CuPy array")
            features_matrix = cudf.DataFrame(features_matrix)
        else:
            print(f"Input is {type(features_matrix)}")
            features_matrix = cudf.DataFrame(np.asarray(features_matrix))
            
        # Check for NaN values and handle them
        if features_matrix.isna().any().any():
            print("Warning: NaN values found in data. Filling with column means...")
            features_matrix = features_matrix.fillna(features_matrix.mean())
            
        # Calculate safe n_neighbors value - make it more conservative
        n_samples = len(features_matrix)
        n_features = features_matrix.shape[1]
        
        # More conservative n_neighbors based on data dimensionality
        safe_n_neighbors = min(
            n_neighbours,
            int(np.sqrt(n_samples) / 2),  # More conservative than before
            max(10, min(50, n_features)),  # Scale with dimensionality
            50  # Hard cap
        )
        
        if safe_n_neighbors != n_neighbours:
            print_color(f"\nWarning: Adjusted n_neighbors from {n_neighbours} to {safe_n_neighbors} to match dataset size and dimensionality\n", bold_color.YELLOW)

        print(f"Original data shape: {features_matrix.shape}")
        print(f"Using n_neighbors: {safe_n_neighbors}")
        print(f"Data type: {features_matrix.dtypes}")
        
        try:
            # Standardize the data
            print("Standardizing data...")
            scaler = cuStandardScaler(with_mean=True, with_std=True)
            features_matrix = scaler.fit_transform(features_matrix)
            
            # If dimensionality is high, reduce it first with PCA
            if n_features > 50:
                print_color("\nReducing dimensionality with PCA first...\n", bold_color.YELLOW)
                n_components_pca = min(50, n_features)
                pca = cuPCA(n_components=n_components_pca)
                features_matrix = pca.fit_transform(features_matrix)
                print(f"Reduced data shape: {features_matrix.shape}")
            
            # Convert to float32 for better GPU performance
            if isinstance(features_matrix, cudf.DataFrame):
                features_matrix = features_matrix.astype(np.float32)
            else:
                features_matrix = cp.asarray(features_matrix, dtype=cp.float32)
            
            # Initialize UMAP with direct kNN computation
            print("Running UMAP...")
            umap = cuMLUMAP(
                n_components=int(n_components),
                n_neighbors=safe_n_neighbors,
                min_dist=min_dist,
                spread=1.0,
                local_connectivity=1,  # Reduce this to help with sparse data
                repulsion_strength=2.0,  # Increase this to help with clustering
                negative_sample_rate=5,
                transform_queue_size=4.0,
                init="spectral",
                random_state=42,
                metric='euclidean',  # Use euclidean distance
                verbose=True  # Enable verbose output
            )
            
            # Fit and transform
            print("Running UMAP fit_transform...")
            embedding = umap.fit_transform(features_matrix)
            
            # Convert back to Dask array
            if isinstance(embedding, cp.ndarray):
                embedding = cp.asnumpy(embedding)
            elif isinstance(embedding, cudf.DataFrame):
                # Convert cuDF DataFrame to numpy array first
                embedding = embedding.values.get()
            embedding = da.from_array(embedding, chunks=(chunksize, n_components))
            
        except RuntimeError as e:
            print_color(f"\nError with RAPIDS cuML UMAP: {str(e)}\nFalling back to CPU UMAP...\n", bold_color.YELLOW)
            # Convert data back to CPU
            if isinstance(features_matrix, cudf.DataFrame):
                features_matrix = features_matrix.to_numpy()
            elif isinstance(features_matrix, cp.ndarray):
                features_matrix = cp.asnumpy(features_matrix)
            
            from umap import UMAP as skUMAP
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            print("Standardizing data...")
            scaler = StandardScaler(with_mean=True, with_std=True)
            features_matrix = scaler.fit_transform(features_matrix)
            
            # If dimensionality is high, reduce it first with PCA
            if n_features > 50:
                print_color("\nReducing dimensionality with PCA first...\n", bold_color.YELLOW)
                from sklearn.decomposition import PCA
                n_components_pca = min(50, n_features)
                pca = PCA(n_components=n_components_pca)
                features_matrix = pca.fit_transform(features_matrix)
                print(f"Reduced data shape: {features_matrix.shape}")
            
            print("Running UMAP...")
            umap = skUMAP(
                n_components=int(n_components),
                n_neighbors=safe_n_neighbors,
                min_dist=min_dist,
                spread=1.0,
                local_connectivity=1,
                repulsion_strength=2.0,
                negative_sample_rate=5,
                init="spectral",
                random_state=42,
                metric='euclidean',
                verbose=True,
                densmap=densmap,
                low_memory=True
            )
            embedding = umap.fit_transform(features_matrix)
            embedding = da.array(embedding)

    else:
        print_color("\nUsing sklearn UMAP\n", bold_color.GREEN)
        from umap import UMAP as skUMAP
        import numpy as np

        # Handle different input types
        if isinstance(features_matrix, pd.DataFrame):
            features_matrix = features_matrix.values
        elif isinstance(features_matrix, dd.DataFrame):
            features_matrix = features_matrix.compute().values
        elif isinstance(features_matrix, da.Array):
            features_matrix = features_matrix.compute()
        elif not isinstance(features_matrix, np.ndarray):
            raise TypeError(f"features_matrix must be a NumPy array, Pandas DataFrame, or Dask array/dataframe. Got type: {type(features_matrix)}")

        # Calculate safe n_neighbors value
        n_samples = features_matrix.shape[0]
        safe_n_neighbors = min(n_neighbours, n_samples - 1)
        
        if safe_n_neighbors != n_neighbours:
            print_color(f"\nWarning: Adjusted n_neighbors from {n_neighbours} to {safe_n_neighbors} to match dataset size\n", bold_color.YELLOW)

        umap = skUMAP(
            n_components=int(n_components),
            n_neighbors=safe_n_neighbors,
            min_dist=min_dist,
            densmap=densmap,
            low_memory=True,
            random_state=42,
        )

        embedding = umap.fit_transform(features_matrix)
        embedding = da.array(embedding)
    
    try:
        dump(umap, umap_save_path)
        print_color(f"UMAP object saved to: {umap_save_path}", bold_color.GREEN)
    except Exception as e:
        print_color(f"Error saving UMAP object: {e}", bold_color.RED)
        raise
    
    return embedding, umap


@tictoc
def get_hdbscan_clusters(clusterer, embedding):
    labels = da.array(clusterer.fit_predict(embedding))
    return labels


@tictoc
def get_hdbscan_cuml_clusters(clusterer, embedding):
    labels = da.array(clusterer.fit_predict(embedding))
    return labels


@tictoc
def hdbscan_condensed_tree(clusterer, plot=True, show=True):
    condensed_tree = clusterer.condensed_tree_.to_pandas()
    if plot:
        plt.figure()
        clusterer.condensed_tree_.plot(
            select_clusters=True,
        )
        if show:
            plt.show()

    return condensed_tree


@tictoc
def hdbscan_single_linkage_tree(
    clusterer, plot=True, show=False
):
    linkage_tree = clusterer.single_linkage_tree_.to_pandas()
    if plot:
        plt.figure()
        clusterer.single_linkage_tree_.plot()
        if show:
            plt.show()

    return linkage_tree


def get_hdbscan_single_linkage_clusters(
    clusterer, epsilon_value, min_cluster_size
):
    """ """

    labels = clusterer.single_linkage_tree_.get_clusters(
        epsilon_value, min_cluster_size=min_cluster_size
    )

    return labels


@tictoc
def validation_score(func, X, labels, func_name) -> float:
    try:
        logger.info(f"| validation_score | {func_name} computing...")
        score = func(X, labels)
        try:
            score = round(score, 2)
        except:
            pass

    except Exception as e:
        logger.error(f"| validation_score | {func_name} error: {e}")
        score = "not_computed"

    return score


@tictoc
def validate_clusters(
    embedding,
    labels,
    subset: bool = True,
    subset_size: float = 0.1,
    score_type: list[str] = ["dbcv-hdbscan", "davis-bouldin", "silhouette"],
    mask_noise: bool = True,
) -> dict[str, float]:
    """
    Calculates cluster validation scores for a given embedding and labels.

    Args:
        embedding: A DataFrame containing the feature representation of data.
        labels:  A DataFrame containing the cluster labels for each data point.
        subset: If True, resamples the data using stratified shuffling.
        score_type: A list of score types to calculate
                    (e.g., 'dbcv-hdbscan', 'davis-bouldin', 'silhouette').
        test_size: The proportion of data to use for the test set during resampling.
        mask_noise: If True, removes data points with a noise label (-1).

    Returns:
         A dictionary containing the calculated validation scores.
    """

    # Input validation (add more checks as needed)
    if not isinstance(embedding, pd.DataFrame) or not isinstance(
        labels, pd.DataFrame
    ):
        raise TypeError("embedding and labels must be Pandas DataFrames")

    # Convert to NumPy arrays for efficiency
    embedding = embedding.astype(float)
    labels = labels.astype(int)

    # Precompute noise mask
    if mask_noise:
        mask = (
            labels.squeeze() != -1
        )  # Convert labels to a Series before creating the mask
        if not np.any(mask):
            raise ValueError("labels does not contain any -1 values")

    validation_funcs = {
        "dbcv-hdbscan": validity.validity_index,
        "davis-bouldin": davies_bouldin_score,
        "silhouette": silhouette_score,
    }

    scores = {}  # Initialize the scores dictionary

    try:
        if mask_noise:
            embedding = embedding.loc[mask, :].astype(float)
            labels = labels.loc[mask].astype(int)

        if subset:
            tratsplit = StratifiedShuffleSplit(
                n_splits=1, test_size=subset_size, random_state=0
            )
            for _, test_index in tratsplit.split(embedding, labels):
                test_index = test_index.flatten()  # Flatten test_index
                embedding_sample, labels_sample = (
                    embedding.iloc[test_index, :],
                    labels.iloc[test_index],
                )
                embedding_sample = (
                    embedding_sample.to_numpy()
                )  # Convert embedding_sample to a numpy array
                labels_sample = (
                    labels_sample.squeeze().to_numpy().astype(int)
                )  # Ensure labels_sample is a Series, convert it to a numpy array, and ensure it's of integer type
                for name, func in validation_funcs.items():
                    if name in score_type:
                        scores[name] = func(embedding_sample, labels_sample)

        else:  # No resampling
            for name, func in validation_funcs.items():
                if name in score_type:
                    embedding = (
                        embedding.to_numpy()
                    )  # Convert embedding_sample to a numpy array
                    labels = labels.squeeze().to_numpy().astype(int)
                    scores[name] = func(embedding, labels)

    except ValueError as e:  # Log the error
        logging.error(f"|main| validation error: {e}")
        scores = {name: "not_computed" for name in score_type}

    return scores


Parallel(
    n_jobs=-1,
    backend="loky",
    temp_folder=r"C:\Users\userratos\AppData\Local\Temp",
    verbose=10,
)  # max_nbytes=None


@tictoc
def process_data(features_raw, pca, pca_threshold=0.95, 
                 rapids_cuml=False, n_components: int = None, 
                 subgroup_label_to_cluster: str = None, 
                 subgroup_clustering: bool = False) -> tuple:
    # Convert to float32 to reduce memory usage
    if rapids_cuml:
        import cudf
        import cupy as cp
        if isinstance(features_raw, (dd.DataFrame, pd.DataFrame)):
            # First compute if it's a Dask DataFrame
            if isinstance(features_raw, dd.DataFrame):
                features_raw = features_raw.compute()
            
            # Fill null values with mean of each column
            for column in features_raw.columns:
                features_raw[column] = features_raw[column].fillna(features_raw[column].mean())
            
            # Then convert to cuDF DataFrame
            features_raw = cudf.DataFrame(features_raw)
        features_raw = features_raw.astype(np.float32)
        from cuml.preprocessing import StandardScaler
    else:
        if isinstance(features_raw, dd.DataFrame):
            features_raw = features_raw.compute()
        features_raw = features_raw.astype(np.float32)
        from sklearn.preprocessing import StandardScaler

    # Process data in chunks using Dask
    scaler = StandardScaler()
    if rapids_cuml:
        features_scaled = scaler.fit_transform(features_raw)
        features_scaled.columns = features_raw.columns
        features_scaled.index = features_raw.index
    else:
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

    features_da = features_scaled.copy()

    if isinstance(features_da, dd.DataFrame):
        features_da = features_da.compute()
    
    if rapids_cuml:
        if isinstance(features_da, cudf.DataFrame):
            print("features_da is a cudf.DataFrame")
            
            # Ensure no nulls before converting to array
            for column in features_da.columns:
                features_da[column] = features_da[column].fillna(features_da[column].mean())
            # Convert to float32 array
            features_da = cp.asarray(features_da.values, dtype=cp.float32)
        else:
            # Convert to cuDF, handle nulls, then get values
            features_da = cudf.DataFrame(features_da)
            for column in features_da.columns:
                features_da[column] = features_da[column].fillna(features_da[column].mean())
            # Convert to float32 array
            features_da = cp.asarray(features_da.values, dtype=cp.float32)
    else:
        features_da = np.asarray(features_da, dtype=np.float32)

    if n_components:
        n_components = n_components
    else:
        # For RAPIDS cuML, we need to handle PCA differently
        if rapids_cuml:
            try:
                # First fit with all components to get variance ratio
                n_features = min(features_da.shape[0], features_da.shape[1])
                pca = cuMLPCA(n_components=int(n_features), svd_solver='full')
                
                # Check if data is too large for GPU memory
                data_size = features_da.nbytes / (1024 ** 3)  # Size in GB
                if data_size > 2:  # If data is larger than 2GB
                    print_color("\nData too large for GPU memory, using CPU PCA\n", bold_color.YELLOW)
                    raise MemoryError("Data too large for GPU memory")
                
                if cp.isnan(features_da).any():
                    print("NaNs are present in the data")
                
                # Fit PCA with all components
                pca.fit(features_da)
                
                # Get explained variance ratio from cuML PCA
                explained_variance_ratio = pca.explained_variance_ratio_
                if isinstance(explained_variance_ratio, np.ndarray):
                    explained_variance_ratio = cp.asarray(explained_variance_ratio)
                
                # Use cp.cumsum for CuPy arrays
                cumulative_variance_ratio = cp.cumsum(explained_variance_ratio)
                
                # Convert threshold to CuPy array for searchsorted
                threshold_array = cp.array([pca_threshold])
                n_components = int(cp.searchsorted(cumulative_variance_ratio, threshold_array)[0]) + 1
                
            except Exception as e:
                print_color(f"\nFalling back to CPU PCA: {str(e)}\n", bold_color.YELLOW)
                # Convert data back to CPU if needed
                if isinstance(features_da, cp.ndarray):
                    features_da = cp.asnumpy(features_da)
                
                # Use sklearn PCA
                from sklearn.decomposition import PCA as skPCA
                pca = skPCA(n_components=None, svd_solver='full')
                pca.fit(features_da)
                
                cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.searchsorted(cumulative_variance_ratio, pca_threshold) + 1
                
        else:
            # For large datasets, use randomized SVD solver
            if features_da.shape[0] > 10000:
                print_color("\nLarge dataset detected, using randomized SVD solver\n", bold_color.YELLOW)
                from sklearn.decomposition import PCA as skPCA
                
                # First fit with a subset to estimate n_components
                subset_size = min(10000, features_da.shape[0])
                subset_indices = np.random.choice(features_da.shape[0], subset_size, replace=False)
                subset_data = features_da[subset_indices]
                
                temp_pca = skPCA(n_components=None, svd_solver='full')
                temp_pca.fit(subset_data)
                
                # Calculate number of components needed
                cumulative_variance_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
                n_components = np.searchsorted(cumulative_variance_ratio, pca_threshold) + 1
                
                # Now fit final PCA with estimated n_components
                pca = skPCA(n_components=int(n_components), svd_solver='randomized')
                pca.fit(features_da)
            else:
                pca.fit(features_da)
                
                # Calculate cumulative explained variance ratio
                cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
                
                # Determine the number of components that explain the desired amount of variance
                pca_threshold_array = np.array([pca_threshold])
                n_components = np.searchsorted(cumulative_variance_ratio, pca_threshold_array)[0] + 1
        
        if n_components < 1:
            raise ValueError("Number of components cannot be less than one.")

    print_color(
        f"\n {n_components} PCA components > "
        + f"{round(pca_threshold*100)} % variance \n",
        bold_color.GREEN,
    )

    return features_scaled, n_components


def get_full_file_names(parent_path):
    file_names = {
        "embedding": "embedding.csv",
        # 'embedding_with_labels_hdbscan': 'embedding_with_labels_hdbscan.csv',
        # 'features_with_labels_hdbscan': 'features_with_labels_hdbscan.csv',
        "metadata": "metadata.json",
        # 'original_full': 'original_full.csv',
        "embedding_full": "embedding_with_labels.csv",
        "embedding_full_light": "embedding_with_labels_light.csv",
        "clusters_figure": "clusters_figure.png",
        "embedding_figure": "embedding_figure.png",
        "clusterer": "clusterer.joblib",
        "colour_map": "colour_map.pkl",
        "label_encoder": "label_encoder.pkl",
        "umap_save_path": "umap.joblib",
        "features_full": "features_full.csv",
    }

    full_file_names = {
        k: os.path.join(parent_path, v) for k, v in file_names.items()
    }

    return full_file_names


@tictoc
def perform_clustering(
    embedding,
    cluster_size: int,
    samples: int,
    single_cluster: bool,
    #    rapids_cuml: bool,
    approx_min_span_tree: bool,
    epsilon: float = 0.0,
    rapids_cuml: bool = False,
):
    hdbscan_params = {
        "min_cluster_size": cluster_size,
        "min_samples": samples,
        "allow_single_cluster": single_cluster,
        "cluster_selection_epsilon": epsilon,
        "approx_min_span_tree": approx_min_span_tree,
    }

    if rapids_cuml:
        print_color("\nUsing RAPIDS cuML HDBSCAN\n", bold_color.GREEN)
        clusterer = cuMLHDBSCAN(**hdbscan_params)
        labels = get_hdbscan_cuml_clusters(clusterer, embedding)
    else:
        # clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        clusterer = regularHDBSCAN(**hdbscan_params)
        labels = get_hdbscan_clusters(clusterer, embedding)

    labels_clustered = labels.compute()[labels.compute() >= 0]

    
    if isinstance(embedding, da.Array):
        embedding = embedding.compute_chunk_sizes()
        percentage_labeled = round(
            len(labels_clustered) / len(embedding) * 100, 2)
    else:
        percentage_labeled = round(
            len(labels_clustered) / len(embedding) * 100, 2)

    if rapids_cuml:
        labels_set = set(labels.compute().to_pandas())
    else:
        labels_set = set(labels.compute())
    print_color(f"\nlabels: {labels_set}\n", bold_color.DARKCYAN)
    print_color(
        f"\nlabeled data: {percentage_labeled}% | "
        + f"non-labeled data: {round(100-percentage_labeled,2)}%\n",
        bold_color.GREEN,
    )

    return labels, clusterer, percentage_labeled, hdbscan_params


@tictoc
def save_data(
    embedding, cluster_labels, filenames: dict, features = None
) -> None:
    print_color("=== ENTERING save_data FUNCTION ===", bold_color.CYAN)
    print_color(f"save_data called with embedding type: {type(embedding)}, cluster_labels type: {type(cluster_labels)}", bold_color.YELLOW)
    
    # Ensure cluster_labels is a list or dict-like object
    if isinstance(cluster_labels, pd.Series):
        cluster_labels = cluster_labels.to_pandas()
        if features is not None:
            features = features.to_pandas()
    elif isinstance(cluster_labels, dd.Series):
        cluster_labels = cluster_labels.compute().to_pandas()
        if features is not None:
            features = features.compute()
    elif isinstance(cluster_labels, pd.DataFrame):
        cluster_labels = cluster_labels.to_pandas()
        if features is not None:
            features = features.to_pandas()
    # elif isinstance(cluster_labels, da.dataframe.DataFrame):
    #     cluster_labels = cluster_labels.compute().to_pandas()

    print_color(f"After type conversion - cluster_labels type: {type(cluster_labels)}", bold_color.YELLOW)

    # Convert Dask arrays to Pandas dataframes
    try:
        cluster_labels_df = pd.DataFrame(cluster_labels.compute())
        cluster_labels_df.columns = ['label']
        embeddings_df = pd.DataFrame(embedding)
        cluster_labels_df.index = embeddings_df.index
        print_color(f"Successfully created DataFrames. cluster_labels_df shape: {cluster_labels_df.shape}, embeddings_df shape: {embeddings_df.shape}", bold_color.GREEN)
    except Exception as e:
        print_color(f"Error creating DataFrames: {e}", bold_color.RED)
        raise

    embedding_with_labels = pd.concat(
        [embeddings_df, cluster_labels_df], axis=1
    )

    if features is not None:
        features_df = pd.DataFrame(features)
        features_with_labels = pd.concat(
            [embeddings_df.iloc[:, :4], features_df, cluster_labels_df], axis=1
        )

    embedding_with_labels.columns = (
        ["ppid", "session", "trial", "time"]
        + list(embedding_with_labels.columns[4:-1])
        + ["label"]
    )

    if features is not None:
        features_with_labels.columns = (["ppid", "session", "trial", "time"] + list(features.columns) + ["label"])

    embedding_with_labels_light = embedding_with_labels[["ppid", "session", "trial", "time", "label"]]

    # Save raw embedding (without labels)
    try:
        embeddings_df.to_csv(
            filenames["embedding"], header=True, index=False
        )
        print_color(f"Saved embedding to: {filenames['embedding']}", bold_color.GREEN)
    except Exception as e:
        logger.error(f"Error saving embedding: {e}")
        raise

    try:
        embedding_with_labels_light.to_csv(
            filenames["embedding_full_light"], header=True, index=False
        )
        print_color(f"Saved embedding_with_labels_light to: {filenames['embedding_full_light']}", bold_color.GREEN)
    except Exception as e:
        logger.error(f"Error saving embedding_with_labels_light: {e}")
        raise

    try:
        embedding_with_labels.to_csv(
            filenames["embedding_full"], header=True, index=False
        )
        print_color(f"Saved embedding_with_labels to: {filenames['embedding_full']}", bold_color.GREEN)
    except Exception as e:
        logger.error(f"Error saving embedding_with_labels: {e}")
        raise

    if features is not None:
        try:
            features_with_labels.to_csv(
                filenames["features_full"], header=True, index=False
            )
            print_color(f"Saved features_with_labels to: {filenames['features_full']}", bold_color.GREEN)
        except Exception as e:
            logger.error(f"Error saving features_with_labels: {e}")
            raise

    print_color("=== EXITING save_data FUNCTION ===", bold_color.CYAN)
    return embedding_with_labels


def verify_files_saved(filenames: dict) -> None:
    """Verify that all expected files were saved successfully."""
    missing_files = []
    for key, filepath in filenames.items():
        if key in ["embedding", "embedding_full", "embedding_full_light", "features_full", "metadata", "clusterer", "colour_map", "label_encoder", "umap_save_path"]:
            if not os.path.exists(filepath):
                missing_files.append(f"{key}: {filepath}")
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        raise FileNotFoundError(f"Failed to save {len(missing_files)} files: {missing_files}")
    else:
        print_color(f"All files saved successfully to: {os.path.dirname(filenames['metadata'])}", bold_color.GREEN)


from tqdm import tqdm


def plot_label_sequence(x, labels, colors, color_map, legend_args, rapids_cuml=False):
    # Create a sequence of colors corresponding to the labels
    import numpy as np

    if rapids_cuml:
        import cupy as cp

    plt.figure(figsize=(8, 1))

    # Create a bar plot
    plt.bar(x, [1] * len(labels), color=colors)

    # Set the limits of the x-axis and y-axis
    plt.xlim([min(x), max(x)])
    plt.ylim([0, 1])

    # Set the x-ticks to values in seconds
    ticks = np.arange(min(x), max(x), step=5)

    if rapids_cuml: 
        if isinstance(ticks, cp.ndarray):
            ticks = ticks.get()
    elif isinstance(ticks, np.ndarray):
        ticks = ticks

    plt.xticks(
        ticks, labels=[f"{int(tick)}s" for tick in ticks], fontsize=10
    )

    # Remove the box surrounding the plot
    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(False)

    # Change the color of the x-axis ticks and line to grey
    plt.tick_params(axis="x", colors="grey")

    patches = [
        plt.Line2D([0], [0], color=color, lw=5, label=label)
        for label, color in color_map.items()
    ]

    legend_args.update(
        {
            "handles": patches,
            "prop": {"size": 4},
            "ncol": 2,
            "labelspacing": 0.05,
            "borderpad": 0.4,
            "fontsize": 4,
            "bbox_to_anchor": (1, 1.4),
        }
    )

    plt.legend(**legend_args)

    # Remove the y-axis
    plt.yticks([])


@tictoc
def plot_data(
    data,
    filenames: dict,
    show: bool = False,
    block: bool = True,
    save_figures: bool = True,
    keep_figures_open: bool = False,
    labels_scatter: bool = False,
    labels_column: int = None,
    plot_all_data_figures: bool = True,
    rapids_cuml: bool = False,
) -> None:
    CMAP_COLORMAP = create_pastel_rainbow_cmap()
    SCATTER_MARKER_SIZE = 9
    DPI = 300
    
    # Create paths
    parent_dir = os.path.dirname(filenames["metadata"])
  
    parent_dir_bar = create_directory(
        os.path.join(parent_dir, "label_sequence_plots")
    )

    legend_args = {
        "handles": "",
        "bbox_to_anchor": (1, 1),
        "loc": "upper left",
        "handlelength": 0.1,
        "fontsize": 5,  # Reduced font size
        "frameon": False,
        "borderpad": 0.1,  # Reduced padding inside the legend border
        "labelspacing": 0.03,  # Reduced spacing between legend entries
        "prop": {"size": 8},  # Adjust the size of the legend and its elements
    }

    embedding = data.iloc[:, 4:]
    if labels_column:
        labels = data.loc[:, labels_column]
    else:
        labels = data.iloc[:, -1]
        labels_column = data.columns[-1]
        
    import numpy as np
    import pandas as pd
    
    if rapids_cuml:
        import cupy as cp
        import cudf as cudf
        embedding = embedding.to_pandas()
        # labels = labels.to_pandas().to_numpy()
        labels = labels.to_pandas()
        data = data.to_pandas()
    else:
        labels = labels

    x_seconds = pd.DataFrame(embedding).index / DATA_COLLECTION_RATE

    if rapids_cuml:
        if isinstance(labels, cp.ndarray):
            labels = labels.get()
        elif isinstance(labels, cudf.Series):
            labels = labels.to_pandas().to_numpy()
    else:
        labels = labels

    unique_labels = np.unique(labels)
    unique_labels = [tuple(label) if isinstance(label, np.ndarray) else int(label) for label in unique_labels]

    cmap = plt.get_cmap(CMAP_COLORMAP)
    pastel_grey = "grey"
    color_map = {
        label: cmap(i / len(unique_labels)) if (label != -1) and (label != '-1') else pastel_grey
        for i, label in enumerate(unique_labels)
    }

    from sklearn.preprocessing import LabelEncoder
    import pickle

    # Save the group_color_map dictionary to a file
    with open(filenames["colour_map"], "wb") as f:
        pickle.dump(color_map, f)

    # Create and fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(list(color_map.keys()))

    # Save the label encoder to a file
    with open(filenames["label_encoder"], "wb") as f:
        pickle.dump(label_encoder, f)

    colors = [color_map[label] for label in labels]

    if plot_all_data_figures:
        fig_scatter_clusters = plt.figure()
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.scatter(
            x_seconds,
            labels,
            marker="s",
            s=SCATTER_MARKER_SIZE,
            c=colors,
            edgecolors="none",
        )

        # Set the x-ticks to values in seconds
        ticks = np.arange(min(x_seconds), max(x_seconds), step=1000)
        plt.xticks(ticks, labels=[f"{int(tick)}s" for tick in ticks], fontsize=8)

        # plt.xticks(sub_x_seconds, sub_x_ticklabels, rotation=45)
        plt.title("Clusters | All", fontweight="bold")

        # Add x and y axis labels
        plt.xlabel("time", fontweight="bold")
        plt.ylabel("label", fontweight="bold")

        patches = [
            plt.Line2D([0], [0], color=color, lw=4, label=label)
            for label, color in color_map.items()
        ]
        legend_args["handles"] = patches
        legend_args["ncol"] = 2

        if rapids_cuml:
            labels = labels
        else:
            labels = labels.values

        fig_embedding = plot_clusters(
            embedding=embedding.values,
            labels=labels.values if rapids_cuml else labels,
            title="UMAP componets | UMAP-HDBSCAN",
        )

        if save_figures:
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())

            fig_scatter_clusters.savefig(filenames["clusters_figure"], dpi=DPI)
            fig_embedding.savefig(filenames["embedding_figure"], dpi=DPI)

            if not keep_figures_open:
                plt.close(fig_scatter_clusters)
                plt.close(fig_embedding)

            print_color(
                f"\nMain cluster figures saved successfully.\n", bold_color.GREEN
            )

        if show:
            plt.show(block=block)

    grouped_df = data.groupby(["ppid", "session", "trial"])

    for name, group in tqdm(
        grouped_df, desc="Processing groups", unit="group"
    ):
        ppid, session, trial = name
        ppid = int(ppid)
        session_str = str(int(session)).zfill(3)
        trial_str = str(int(trial)).zfill(3)
        # figure_file_name_scatter = (
        #     f"label_scatter_{ppid}_S{session_str}_T{trial_str}.png"
        # )
    
        figure_file_name_bar = (
            f"label_sequence_{ppid}_S{session_str}_T{trial_str}.png"
        )
        figure_full_filename_bar = os.path.join(
            parent_dir_bar, figure_file_name_bar
        )

        # Use the color map created outside of the loop
        if rapids_cuml:
            # labels = group[labels_column].to_pandas()
            # group_labels = np.unique(group[labels_column].to_pandas().to_numpy())
            labels = group[labels_column]
            group_labels = np.unique(group[labels_column])
        else:
            labels = group[labels_column]
            group_labels = np.unique(group[labels_column])
        colors = [color_map[label] for label in labels]

        group_color_map = {
            label: color_map[label]
            for label in group_labels
            if label in color_map
        }

        # Create a custom legend
        patches = [
            plt.Line2D([0], [0], color=color, lw=4, label=label)
            for label, color in group_color_map.items()
        ]

        # Create a custom legend with adjusted font size
        legend = plt.legend(handles=patches, title="Groups", fontsize=8)

        legend_args["handles"] = patches
        legend_args["ncol"] = 1
        legend_args["labelspacing"] = 0.2
        legend_args["borderpad"] = 0.4
        legend_args["bbox_to_anchor"] = (1, 1)
        legend_args["prop"] = {"size": 8}
        legend_args["fontsize"] = 5


        # Pass the group-specific color map to the plot_label_sequence function
        if rapids_cuml:
            # time = group["time"].to_pandas().to_numpy()
            # group_labels_current = group[labels_column].to_pandas().to_numpy()
            time = group["time"]
            group_labels_current = group[labels_column]
        else:
            time = group["time"]
            group_labels_current = group[labels_column]

        plot_label_sequence(
            time,
            group_labels_current,
            colors,
            color_map=group_color_map,
            legend_args=legend_args,
            rapids_cuml=rapids_cuml,
        )
        # plot_label_sequence(group['time'], group[label_column], colors, color_map=color_map, legend_args=legend_args)
        plt.title(
            f"Label sequence | Participant: {ppid}, Session: {session_str}, Trial: {trial_str}",
            fontweight="bold",
            fontsize=12,
        )

        plt.tight_layout()

        if show:
            plt.show(block=block)
        plt.savefig(figure_full_filename_bar, dpi=DPI)
        plt.close("all")

    print_color(
        f"\nSub cluster figures saved successfully.\n", bold_color.GREEN
    )


@tictoc
def hyperparameter_search(
    data,
    approx_min_span_tree: bool = False,
    min_cluster_sizes=[300, 500, 700],
    min_samples_values=[None, 30, 50],
    cluster_selection_epsilons=[0.0, 1.0, 2.0],
    metrics=["euclidean"],
    subset_hyper: bool = False,
    subset_size_hyper: float = 0.1,
    subset_validation: bool = True,
    subset_size_validation: float = 0.1,
    rapids_cuml: bool = False,
) -> Dict:
    # Initialize the best score
    best_score = -np.inf
    best_hyperparameters = {}

    if subset_hyper:
        from sklearn.model_selection import train_test_split

        # Assuming 'data' is your dataset
        _, data = train_test_split(
            data, test_size=subset_size_hyper, random_state=0, shuffle=False
        )

    results = []
    # Loop over the hyperparameters
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            for cluster_selection_epsilon in cluster_selection_epsilons:
                for metric in metrics:
                    print_color(
                        f"Started computing with => Hyperparameters: {min_cluster_size}, {min_samples}, {cluster_selection_epsilon}, {metric} [{time.asctime()}]",
                        bold_color.GREEN,
                    )
                    # Initialize the HDBSCAN estimator
                    try:
                        if rapids_cuml:
                            clusterer = cuMLHDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_epsilon=cluster_selection_epsilon,
                                metric=metric,
                                allow_single_cluster=approx_min_span_tree,
                            )
                        else:
                            clusterer = regularHDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_epsilon=cluster_selection_epsilon,
                                metric=metric,
                                allow_single_cluster=approx_min_span_tree,
                            )
                        
                        # Fit the estimator to the data
                        clusterer.fit(data)

                        # Compute score
                        score = validate_clusters(
                            data,
                            clusterer.labels_,
                            subset=subset_validation,
                            subset_size=subset_size_validation,
                            score_type=["dbcv-hdbscan"],
                        )
                        # score = hdbscan.validity.silhouette_score(data, clusterer.labels_)
                        print_color(
                            f"Finished computing => Hyperparameters: {min_cluster_size}, {min_samples}, {cluster_selection_epsilon}, {metric} | score:{score} [{time.asctime()}]",
                            bold_color.GREEN,
                        )
                        # Update the best score and hyperparameters if the current score is better

                        results.append(
                            {
                                "score": score,
                                "hyperparameters": {
                                    "min_cluster_size": min_cluster_size,
                                    "min_samples": min_samples,
                                    "cluster_selection_epsilon": cluster_selection_epsilon,
                                    "metric": metric,
                                },
                            }
                        )

                        if score[
                            "dbcv-hdbscan"
                        ] != "not_computed" and isinstance(
                            score["dbcv-hdbscan"], float
                        ):
                            if score["dbcv-hdbscan"] > best_score:
                                best_score = score["dbcv-hdbscan"]
                                best_hyperparameters = {
                                    "min_cluster_size": min_cluster_size,
                                    "min_samples": min_samples,
                                    "cluster_selection_epsilon": cluster_selection_epsilon,
                                    "metric": metric,
                                }

                    except Exception as e:
                        logger.error(
                            f"|main| error computing hyperparameters => size: {min_cluster_size}, samples: {min_samples}, epsilon: {cluster_selection_epsilon}, metric: {metric} timestamp: [{time.asctime()}]: {e}"
                        )

    # Return the best score and hyperparameters
    return {
        "Best score": best_score,
        "Best hyperparameters": best_hyperparameters,
        "results": results,
    }


def get_clusters_from_tree(clusterer, cut_distance, min_cluster_size):
    labels = clusterer.single_linkage_tree_.get_clusters(
        cut_distance=cut_distance, min_cluster_size=min_cluster_size
    )
    return labels


def get_metadata(
    path,
    mode,
    data_collection_rate,
    neighbors,
    min_dist,
    densmap,
    cluster_size,
    samples,
    single_cluster,
    epsilon,
    approx_min_span_tree,
    full_file_names,
):
    start_time = time.asctime()
    return {
        "timestamp": {"start": start_time, "end": ""},
        "data_collection_rate": data_collection_rate,
        "mode": mode,
        "pca_num_components_95_percent_variance": "",
        "umap": {
            "neighbors": neighbors,
            "min_dist": min_dist,
            "densemap": densmap,
        },
        "hdbscan": {
            "min_cluster_size": cluster_size,
            "min_samples": samples,
            "allow_single_cluster": single_cluster,
            "cluster_selection_epsilon": epsilon,
            "approx_min_span_tree": approx_min_span_tree,
        },
        "labeled_data_percentage": "",
        "clustering_validation_score": "",
        "clustering_tunning_results": "",
        "data_path": path,
        "output": full_file_names,
    }


def save_metadata(metadata, file_name):
    class json_encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.int64):
                return int(obj)
            return super(json_encoder, self).default(obj)

    with open(file_name, "w") as f:
        json.dump(metadata, f, cls=json_encoder)


def get_reducer(model):
    reducer = load(model)
    return reducer

def map_data_to_embedding(data, model):
    reducer = get_reducer(model)
    embedding = da.from_array(reducer.transform(data))
    return embedding





@tictoc
def main(args) -> None:
    start_time = time.asctime()

    if args.rapids_cuml:
        print_color("\nUsing RAPIDS cuML\n", bold_color.GREEN)
        import cupy as np
        import cudf as pd
        from cuml.manifold.umap import UMAP as cuMLUMAP
        from cuml import PCA as cuMLPCA
        from cuml.cluster.hdbscan import HDBSCAN as cuMLHDBSCAN
        # Initialize PCA without n_components for RAPIDS
        pca = cuMLPCA()
    else:
        from umap import UMAP as skUMAP
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA as skPCA
        # Initialize PCA for sklearn
        pca = skPCA(n_components=None, svd_solver='full')

    path = os.path.normpath(args.path)
    mode = get_mode(path=path, embedding=args.passing_embedding)

    print_color(
        f"\nPath: {path} \nEmbedding: {args.passing_embedding}"
        + f"\nValidation: {args.validation_check} \nRapids: {args.rapids_cuml}\n",
        bold_color.GREEN,
    )

    # === PREPARE DATA === #
    original_data, features, parent_path, data_with_info = prepare_data(
        path=path, search_pattern=args.search_pattern, mode=mode, rapids_cuml=args.rapids_cuml, subsample_data_rate=args.subsample_data_rate,
        curated_list=args.curated_list
    )

    # === FILE NAMES FOR DATA SAVING === #
    if args.output_path:
        parent_path = args.output_path
        # Create directory if it doesn't exist, but don't modify the path
        if not os.path.exists(parent_path):
            os.makedirs(parent_path, exist_ok=True)
        print_color(f"Using specified output path: {parent_path}", bold_color.GREEN)
    else:
        parent_path = os.path.join(parent_path, "get_cluster_output")
        parent_path = create_directory(parent_path)
        print_color(f"Using default output path: {parent_path}", bold_color.GREEN)
    
    full_file_names = get_full_file_names(parent_path)
    print_color(f"Full file names: {full_file_names}", bold_color.YELLOW)
    metadata = get_metadata(
        path=args.path,
        mode=mode,
        data_collection_rate=args.data_collection_rate,
        neighbors=args.umap_n_neighbors,
        min_dist=args.min_dist,
        densmap=args.densmap,
        cluster_size=args.hdbscan_cluster_size,
        samples=args.hdbscan_samples,
        single_cluster=args.hdbscan_not_single_cluster,
        epsilon=args.hdbscan_epsilon,
        approx_min_span_tree=args.approx_min_span_tree,
        full_file_names=full_file_names,
    )
    save_metadata(metadata, full_file_names["metadata"])

    if mode in ["file", "directory"]:
        print_color("=== PROCESSING FILE/DIRECTORY MODE ===", bold_color.CYAN)
        # === DATA PROCESSING === #
        features_da, n_components = process_data(
            features, pca, pca_threshold=args.pca_threshold, rapids_cuml=args.rapids_cuml,
            n_components=args.n_components
        )

        n_components = n_components if not args.rapids_cuml else int(n_components)
        metadata["pca_num_components_95_percent_variance"] = n_components
        save_metadata(metadata, full_file_names["metadata"])

        # === DIMENSIONALITY REDUCTION UMAP EMBEDING === #
        if args.umap_embedding:
            embedding = map_data_to_embedding(features_da, args.umap_embedding)
        else:
            embedding, umap = get_umap_embedding(
                features_da, n_components, args.umap_n_neighbors, args.min_dist, args.densmap, umap_save_path=full_file_names["umap_save_path"],
                use_cuml=args.rapids_cuml
            )

        # === SAVE EMBEDDING === #
        # Note: Embedding will be saved later in save_data function

        # Use column names instead of iloc
        info_columns = ['ppid', 'session', 'trial', 'time']
        data_with_info_df = data_with_info[info_columns]
        
        if not args.rapids_cuml:
            data_with_info_df = data_with_info_df.compute()
        
        embedding_df = pd.DataFrame(embedding.compute())
        embedding_df.index = data_with_info_df.index
        
        embedding = pd.concat([data_with_info_df, embedding_df], axis=1)
        # Drop duplicate 'time' column if it exists
        time_columns = embedding.columns[embedding.columns == 'time']
        if len(time_columns) > 1:
            # Keep the first 'time' column and drop any duplicates
            embedding = embedding.loc[:, ~embedding.columns.duplicated()]
        
         # Reset index to ensure unique indexing
        embedding = embedding.reset_index(drop=True)

    else:
        print_color("=== PROCESSING EMBEDDING MODE ===", bold_color.CYAN)
        embedding = data_with_info.compute()

    scores = None
    if args.n_components:   
        args.n_components = str(args.n_components) + " components"

    # === CLUSTERING === #
    print_color("=== STARTING CLUSTERING SECTION ===", bold_color.CYAN)
    if args.clustering_tunning:
        approx_min_span_tree = False
        result = hyperparameter_search(
            embedding.iloc[:, 4:],
            subset=args.subset_hyper,
            subset_size=args.subset_size_hyper,
            approx_min_span_tree=approx_min_span_tree,
        )
        metadata["clustering_tunning_results"] = result["results"]
        scores = result["Best score"]
        hdbscan_params = result["Best hyperparameters"]
        cluster_size = hdbscan_params["min_cluster_size"]
        samples = hdbscan_params["min_samples"]
        epsilon = hdbscan_params["cluster_selection_epsilon"]

    approx_min_span_tree = False
    if args.subset_embedding:
        from sklearn.model_selection import train_test_split

        # Assuming 'data' is your dataset
        _, embedding = train_test_split(
            embedding,
            test_size=args.subset_embedding_size,
            random_state=0,
            shuffle=False,
        )
        embedding[:4] = embedding[:4].astype("int")
        embedding = pd.DataFrame(embedding)
        embedding = embedding.reset_index(drop=True)
    else:
        embedding = embedding

    print_color("=== CALLING perform_clustering ===", bold_color.CYAN)
    (
        labels,
        clusterer,
        percentage_labeled,
        hdbscan_params,
    ) = perform_clustering(
        embedding.iloc[:, 4:],
        cluster_size=args.hdbscan_cluster_size,
        samples=args.hdbscan_samples,
        single_cluster=args.hdbscan_not_single_cluster,
        approx_min_span_tree=args.approx_min_span_tree,
        epsilon=args.hdbscan_epsilon,
        rapids_cuml=args.rapids_cuml,
    )
    print_color("=== perform_clustering COMPLETED ===", bold_color.GREEN)
    try:
        dump(clusterer, full_file_names["clusterer"])
        print_color(f"Clusterer saved to: {full_file_names['clusterer']}", bold_color.GREEN)
        logger.info(f"|main| clusterer saved successfully")
    except Exception as e:
        print_color(f"Error saving clusterer: {e}", bold_color.RED)
        logger.error(f"|main| error saving clusterer: {e}")
        raise

    # === SAVE DATA === #
    # original_data = original_data.compute()
    print_color("=== STARTING SAVE DATA SECTION ===", bold_color.CYAN)
    print_color(f"About to save data. Embedding type: {type(embedding)}, Labels type: {type(labels)}, Features type: {type(features)}", bold_color.YELLOW)
    print_color(f"Embedding shape: {embedding.shape if hasattr(embedding, 'shape') else 'No shape'}", bold_color.YELLOW)
    print_color(f"Labels shape: {labels.shape if hasattr(labels, 'shape') else 'No shape'}", bold_color.YELLOW)
    
    try:
        embedding_full = save_data(embedding, labels, full_file_names, features=features)
        print_color("save_data completed successfully", bold_color.GREEN)
    except Exception as e:
        logger.error(f"Error in save_data: {e}")
        print_color(f"Error in save_data: {e}", bold_color.RED)
        raise

    # === VALIDATION === #
    if args.validation_check and not args.clustering_tunning:
        labels = pd.DataFrame(labels.compute())
        embedding_ = embedding.iloc[:, 4:]
        scores = validate_clusters(
            embedding_,
            labels,
            subset=args.subset_validation,
            subset_size=args.subset_validation_size,
            score_type=["dbcv-hdbscan"],
        )

    print_color(f"\nScores: {scores}\n", bold_color.DARKCYAN)

    # === SAVE METADATA === #
    metadata["timestamp"]["end"] = time.asctime()
    metadata["clustering_validation_score"] = scores
    metadata["labeled_data_percentage"] = percentage_labeled
    save_metadata(metadata, full_file_names["metadata"])

    # === PLOT DATA === #
    # if args.clusters_figure:
    plot_data(
        embedding_full,
        full_file_names,
        block=False,
        keep_figures_open=True,
        show=False,
        rapids_cuml=args.rapids_cuml,
    )
    
    # === VERIFY FILES SAVED === #
    print_color("=== STARTING FILE VERIFICATION ===", bold_color.CYAN)
    verify_files_saved(full_file_names)
    print_color("=== MAIN FUNCTION COMPLETED SUCCESSFULLY ===", bold_color.GREEN)


def convert_to_bool(answer):
    return True if answer.lower() == "y" else False


def parse_arguments():
    parser = argparse.ArgumentParser(description='Clustering analysis script')
    
    # Required arguments
    parser.add_argument('--path', type=str, required=True,
                        help='Input directory/file path')

    # Optional arguments with defaults matching your original code
    parser.add_argument('--pca-threshold', type=float, default=PCA_THRESHOLD,
                        help='PCA threshold (mutually exclusive with n-components)')
    parser.add_argument('--n-components', type=int, default=None,
                        help='Number of components (mutually exclusive with pca-threshold)')    
    parser.add_argument('--clusters-figure', action='store_true',
                        help='Whether to generate cluster figures')
    parser.add_argument('--search-pattern', nargs='+', default=["T\d{3}", ".csv"],
                        help='Search patterns for files')
    parser.add_argument('--passing-embedding', type=bool, default=False,
                        help='Whether passing an embedding')
    parser.add_argument('--validation-check', type=bool, default=False,
                        help='Whether to perform validation check')
    parser.add_argument('--rapids-cuml', action='store_true',
                        help='Whether to use RAPIDS cuML')
    parser.add_argument('--approx-min-span-tree', type=bool, default=False,
                        help='Whether to use approximate minimum spanning tree')
    parser.add_argument('--hdbscan-cluster-size', type=int, default=CLUSTER_SIZE,
                        help='Cluster size')
    parser.add_argument('--hdbscan-samples', type=Union[int, None], default=SAMPLES,
                        help='Number of samples')
    parser.add_argument('--hdbscan-epsilon', type=float, default=EPSILON,
                        help='Epsilon value')
    parser.add_argument('--hdbscan-not-single-cluster', action='store_true',
                        help='Whether to allow single cluster')
    parser.add_argument('--umap-n-neighbors', type=int, default=NEIGHBORS,
                        help='Number of neighbors')
    parser.add_argument('--min-dist', type=float, default=MIN_DIST,
                        help='Minimum distance')
    parser.add_argument('--densmap', type=bool, default=DENSMAP,
                        help='Whether to use densmap')
    parser.add_argument('--data-collection-rate', type=int, default=DATA_COLLECTION_RATE,
                        help='Data collection rate')
    parser.add_argument('--clustering-tunning', type=bool, default=False,
                        help='Whether to perform clustering tuning')
    parser.add_argument('--subset-hyper', type=bool, default=False,
                        help='Whether to use subset for hyperparameter search')
    parser.add_argument('--subset-size-hyper', type=float, default=0.99,
                        help='Size of subset for hyperparameter search')
    parser.add_argument('--subset-embedding', type=bool, default=False,
                        help='Whether to use subset for embedding')
    parser.add_argument('--subset-embedding-size', type=float, default=0.9,
                        help='Size of subset for embedding')
    parser.add_argument('--subset-validation', type=bool, default=False,
                        help='Whether to use subset for validation')
    parser.add_argument('--subset-validation-size', type=float, default=0.9,
                        help='Size of subset for validation')
    parser.add_argument('--umap-embedding', type=str, default=None,
                        help='Path to embedding model (joblib file)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Output directory path')
    parser.add_argument('--subsample-data-rate', type=int, default=None,
                        help='Sample every Nth row from each file (e.g., 2 means every second row)')
    parser.add_argument('--curated-list', type=str, default=None,
                        help='Curated list as a csv file. It expects the columns to be ppid, session, trial.')


    return parser.parse_args()


if __name__ == "__main__":
    os.system("cls||clear")
    
    args = parse_arguments()
    
    if args.rapids_cuml:
        print_color("\nUsing RAPIDS cuML\n", bold_color.GREEN)
        import cupy as np
        import cudf as pd
        from cuml.manifold.umap import UMAP as cuMLUMAP
        from cuml import PCA as cuMLPCA
        from cuml.cluster.hdbscan import HDBSCAN as cuMLHDBSCAN
    else:
        # from umap import UMAP as skUMAP
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA as skPCA
        from hdbscan import HDBSCAN as regularHDBSCAN
        from hdbscan import validity

    main(args)
