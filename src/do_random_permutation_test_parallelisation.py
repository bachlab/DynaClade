import os
import pdb
from get_syllable_transition_matrices import plot_matrix, encode_and_get_sequences, restructure_matrix
from pomegranate import MarkovChain
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from pickle import dump
from numba import jit, cuda
import numba
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
import random
import time
import os
import networkx as nx
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde, mannwhitneyu, ks_2samp, entropy
from sklearn.metrics import mutual_info_score

# Distance metrics as standalone functions
@jit(nopython=True)
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

@jit(nopython=True)
def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

@jit(nopython=True)
def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return 1 - (dot_product / (norm1 * norm2)) if norm1 * norm2 != 0 else 1

@jit(nopython=True)
def correlation_distance(vec1, vec2):
    return 1 - np.corrcoef(vec1, vec2)[0, 1] if len(vec1) > 1 else 1

@jit(nopython=True)
def kl_divergence_distance(vec1, vec2):
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    vec1 = vec1 + eps
    vec2 = vec2 + eps
    # Normalize to get probability distributions
    vec1 = vec1 / np.sum(vec1)
    vec2 = vec2 / np.sum(vec2)
    return np.sum(vec1 * np.log(vec1 / vec2))

# def compare_networks(graphs: dict) -> tuple:
#     """Compare two networks using graph edit distance."""
#     if len(graphs) != 2:
#         raise ValueError("Expected exactly two graphs for comparison")
    
#     # Get the two graphs
#     keys = list(graphs.keys())
#     key1, key2 = keys[0], keys[1]
#     graph1 = graphs[key1]
#     graph2 = graphs[key2]
    
#     print(f"\n\nComparing {key1} and {key2}")
#     print(f"Graph1: {graph1}")
#     print(f"Graph2: {graph2}")

#     # Define the cost functions for the graph edit distance
#     def node_del_cost(node, g1=graph1):
#         # Handle empty node case
#         if not node:
#             return 0
#         # Convert node to string if it's not already
#         node_str = str(node)
#         print(f"\n\nnode del cost: node: {node_str}")
#         if node_str in g1:
#             degree = g1.degree(node_str)
#             print(f"g1.degree({node_str}): {degree}")
#             return degree
#         return 0

#     def node_ins_cost(node, g2=graph2):
#         # Handle empty node case
#         if not node:
#             return 0
#         # Convert node to string if it's not already
#         node_str = str(node)
#         print(f"\n\nnode ins cost: node: {node_str}")
#         if node_str in g2:
#             degree = g2.degree(node_str)
#             print(f"g2.degree({node_str}): {degree}")
#             return degree
#         return 0

#     def node_subst_cost(node1, node2, g1=graph1, g2=graph2):
#         # Handle empty node cases
#         if not node1 or not node2:
#             return float('inf')  # High cost for invalid substitution
        
#         # Convert nodes to strings if they're not already
#         node1_str = str(node1)
#         node2_str = str(node2)
        
#         print(f"\n\nnode subst cost: node1: {node1_str}, node2: {node2_str}")
        
#         # Calculate degrees
#         deg1 = g1.degree(node1_str) if node1_str in g1 else 0
#         deg2 = g2.degree(node2_str) if node2_str in g2 else 0
        
#         print(f"g1.degree({node1_str}): {deg1}")
#         print(f"g2.degree({node2_str}): {deg2}")
#         print(f"abs(deg1 - deg2): {abs(deg1 - deg2)}")
        
#         return abs(deg1 - deg2)

#     def edge_subst_cost(edge1, edge2):
#         # Handle empty edge cases
#         if not edge1 or not edge2:
#             return float('inf')  # High cost for invalid substitution
            
#         weight1 = edge1.get('weight', 0) if isinstance(edge1, dict) else 0
#         weight2 = edge2.get('weight', 0) if isinstance(edge2, dict) else 0
#         return abs(weight1 - weight2)

#     if key1 != key2:  # Avoid comparing a network to itself
#         try:
#             ged_iter = nx.optimize_graph_edit_distance(
#                 graph1, 
#                 graph2, 
#                 node_subst_cost=node_subst_cost, 
#                 edge_subst_cost=edge_subst_cost, 
#                 node_del_cost=node_del_cost, 
#                 node_ins_cost=node_ins_cost
#             )
#             ged = next(ged_iter)  # get the first (and best) result
#             return (key1, key2), ged
#         except Exception as e:
#             print(f"Error in graph edit distance calculation: {str(e)}")
#             return (key1, key2), float('inf')  # Return infinity for failed comparison
    
#     return (key1, key2), 0.0  # Return 0 distance if comparing same network


def prepare_graph_for_comparison(G, graph_name_for_logging="graph"):
    """
    Adds 'degree' attribute to each node and ensures edges have 'weight'.
    This version assumes G.degree() (sum of in and out for DiGraph) is desired.
    """
    is_modified = False
    for node_id, data in G.nodes(data=True):
        if 'degree' not in data:
            current_degree = G.degree(node_id)
            G.nodes[node_id]['degree'] = current_degree
            # print(f"Info: Node {node_id} in {graph_name_for_logging} had 'degree' attribute calculated: {current_degree}")
            is_modified = True

    # Ensure edges have a default weight if 'weight' attribute is expected by edge cost functions
    for u, v, data in G.edges(data=True):
        if 'weight' not in data:
            G.edges[u,v]['weight'] = 1.0 # Default weight if missing
            # print(f"Info: Edge ({u},{v}) in {graph_name_for_logging} was assigned default weight 1.0")
            is_modified = True
            
    # if is_modified:
        # print(f"Graph {graph_name_for_logging} was modified to add necessary attributes.")
    return G


def compare_networks(graphs: dict) -> tuple:
    """Compare two networks using graph edit distance."""
    if len(graphs) != 2:
        raise ValueError("Expected exactly two graphs for comparison")

    keys = list(graphs.keys())
    key1, key2 = keys[0], keys[1]
    
    # It's crucial that graphs are prepared with necessary attributes BEFORE this function
    # or at the very beginning of it.
    graph1 = prepare_graph_for_comparison(graphs[key1].copy(), key1) # Use .copy() if you don't want to modify original dict graphs
    graph2 = prepare_graph_for_comparison(graphs[key2].copy(), key2)

    # print(f"\nComparing {key1} and {key2}")
    # For debugging, you can print node data to check attributes:
    # print(f"Graph1 ({key1}) nodes: {list(graph1.nodes(data=True))}")
    # print(f"Graph2 ({key2}) nodes: {list(graph2.nodes(data=True))}")

    # --- Define Cost Functions ---
    # These functions now expect attribute dictionaries as arguments.

    def node_del_cost(node_attr_g1: dict) -> float:
        """Cost of deleting a node from graph1."""
        # print(f"DEBUG node_del_cost: {node_attr_g1.get('degree', 1.0)}")
        return float(node_attr_g1.get('degree', 1.0))

    def node_ins_cost(node_attr_g2: dict) -> float:
        """Cost of inserting a node into graph2."""
        # print(f"DEBUG node_ins_cost: {node_attr_g2}")
        # print(f"DEBUG node_ins_cost: {node_attr_g2.get('degree', 1.0)}")
        return float(node_attr_g2.get('degree', 1.0))

    def node_subst_cost(node_attr_g1: dict, node_attr_g2: dict) -> float:
        """Cost of substituting a node from graph1 with a node from graph2."""
        # print(f"DEBUG node_subst_cost: G1_attr={node_attr_g1}, G2_attr={node_attr_g2}")   
        deg1 = float(node_attr_g1.get('degree', 0.0))
        deg2 = float(node_attr_g2.get('degree', 0.0))
        # print(f"DEBUG node_subst_cost: {deg1} - {deg2} = {abs(deg1 - deg2)}")
        return abs(deg1 - deg2)

    def edge_del_cost(edge_attr_g1: dict) -> float:
        """Cost of deleting an edge from graph1."""
        # print(f"DEBUG edge_del_cost: {edge_attr_g1}")
        # print(f"DEBUG edge_del_cost: {edge_attr_g1.get('weight', 1.0)}")
        return float(edge_attr_g1.get('weight', 1.0)) # Cost is its weight, or 1.0 if no weight

    def edge_ins_cost(edge_attr_g2: dict) -> float:
        """Cost of inserting an edge into graph2."""
        # print(f"DEBUG edge_ins_cost: {edge_attr_g2}") 
        # print(f"DEBUG edge_ins_cost: {edge_attr_g2.get('weight', 1.0)}")
        return float(edge_attr_g2.get('weight', 1.0))

    def edge_subst_cost(edge_attr_g1: dict, edge_attr_g2: dict) -> float:
        """Cost of substituting an edge from graph1 with an edge from graph2."""
        # print(f"DEBUG edge_subst_cost: G1_e_attr={edge_attr_g1}, G2_e_attr={edge_attr_g2}")
        weight1 = float(edge_attr_g1.get('weight', 0.0))
        weight2 = float(edge_attr_g2.get('weight', 0.0))
        # print(f"DEBUG edge_subst_cost: {weight1} - {weight2} = {abs(weight1 - weight2)}")
        return abs(weight1 - weight2)

    # --- Perform Comparison ---
    if key1 == key2:  # Avoid comparing a network to itself
        return (key1, key2), 0.0

    try:
        ged_iterator = nx.optimize_graph_edit_distance(
            graph1,
            graph2,
            node_subst_cost=node_subst_cost,
            node_del_cost=node_del_cost,
            node_ins_cost=node_ins_cost,
            edge_subst_cost=edge_subst_cost,
            edge_del_cost=edge_del_cost,
            edge_ins_cost=edge_ins_cost
        )
        
        # nx.optimize_graph_edit_distance returns an iterator.
        # The values yielded are progressively tighter upper bounds on the GED.
        # The final value is the true GED.
        # Taking next(ged_iterator) gives the first (often quickly computed) upper bound.
        # If you need the exact GED from this iterator, you'd normally iterate to the end:
        # ged_value = None
        # for value in ged_iterator:
        #     ged_value = value # The last one is the most accurate
        # If the iterator is exhausted quickly or your graphs are small, next() might be the exact one.
        ged_value = next(ged_iterator) 

        # print(f"GED (or upper bound) between {key1} and {key2}: {ged_value}")
        return (key1, key2), ged_value
    except StopIteration:
        print(f"Error: GED iterator for {key1} and {key2} was empty. This might indicate an issue or very dissimilar/empty graphs.")
        return (key1, key2), float('inf')
    except Exception as e:
        print(f"Error in graph edit distance calculation between {key1} and {key2}: {str(e)}")
        # For more detailed debugging:
        # import traceback
        # traceback.print_exc()
        return (key1, key2), float('inf')  # Return infinity for failed comparison

# Modified distance calculation function to support multiple metrics
@jit(nopython=True, parallel=True)
def calculate_distances(trials, ref_matrix, metric_name='euclidean'):
    distances = np.zeros(len(trials))
    for i in numba.prange(len(trials)):
        trial_vec = trials[i].flatten()
        ref_vec = ref_matrix.flatten()
        if len(trial_vec) == len(ref_vec):
            if metric_name == 'euclidean':
                distances[i] = euclidean_distance(trial_vec, ref_vec)
            elif metric_name == 'manhattan':
                distances[i] = manhattan_distance(trial_vec, ref_vec)
            elif metric_name == 'cosine':
                distances[i] = cosine_distance(trial_vec, ref_vec)
            elif metric_name == 'correlation':
                distances[i] = correlation_distance(trial_vec, ref_vec)
            elif metric_name == 'kl_divergence':
                distances[i] = kl_divergence_distance(trial_vec, ref_vec)
    return distances

def calculate_graph_distances(trials, ref_graph, template_matrix):
    """Calculate graph edit distances between trial graphs and reference graph."""
    distances = np.zeros(len(trials))
    for i in range(len(trials)):
        # Create graph from trial matrix
        trial_matrix = pd.DataFrame(trials[i], index=template_matrix.index, columns=template_matrix.columns)
        trial_graph = transition_matrix_to_graph(trial_matrix, template_matrix.columns)
        
        # Calculate graph edit distance
        try:
            (_, _), ged = compare_networks({'trial': trial_graph, 'ref': ref_graph})
            distances[i] = ged
        except Exception as e:
            print(f"Warning: Failed to calculate graph edit distance: {str(e)}")
            distances[i] = float('inf')
    return distances

# Modified permutation distances calculation
@jit(nopython=True, parallel=True)
def calculate_permutation_distances(trials, ref_matrix1, ref_matrix2, perm_indices1, perm_indices2, metric_name='euclidean'):
    n1 = len(perm_indices1)
    distances1 = np.zeros(n1)
    distances2 = np.zeros(len(perm_indices2))
    
    for i in numba.prange(n1):
        trial_vec = trials[perm_indices1[i]].flatten()
        ref_vec = ref_matrix1.flatten()
        if len(trial_vec) == len(ref_vec):
            if metric_name == 'euclidean':
                distances1[i] = euclidean_distance(trial_vec, ref_vec)
            elif metric_name == 'manhattan':
                distances1[i] = manhattan_distance(trial_vec, ref_vec)
            elif metric_name == 'cosine':
                distances1[i] = cosine_distance(trial_vec, ref_vec)
            elif metric_name == 'correlation':
                distances1[i] = correlation_distance(trial_vec, ref_vec)
            elif metric_name == 'kl_divergence':
                distances1[i] = kl_divergence_distance(trial_vec, ref_vec)
    
    for i in numba.prange(len(perm_indices2)):
        trial_vec = trials[perm_indices2[i]].flatten()
        ref_vec = ref_matrix2.flatten()
        if len(trial_vec) == len(ref_vec):
            if metric_name == 'euclidean':
                distances2[i] = euclidean_distance(trial_vec, ref_vec)
            elif metric_name == 'manhattan':
                distances2[i] = manhattan_distance(trial_vec, ref_vec)
            elif metric_name == 'cosine':
                distances2[i] = cosine_distance(trial_vec, ref_vec)
            elif metric_name == 'correlation':
                distances2[i] = correlation_distance(trial_vec, ref_vec)
            elif metric_name == 'kl_divergence':
                distances2[i] = kl_divergence_distance(trial_vec, ref_vec)
    
    return distances1, distances2

def tictoc(func):
    def inner(*args, **kwargs):
        # Get just the filename without path and extension
        current_file = os.path.splitext(os.path.basename(func.__code__.co_filename))[0].replace("_", " ").title()
        
        # Use time.perf_counter() for high-resolution timing
        T1 = time.perf_counter()
        T1_str = time.strftime("%H:%M:%S", time.localtime())
        print(f"\033[94m|{current_file}| START: {T1_str}\033[0m")
        func_out = func(*args, **kwargs)
        T2 = time.perf_counter()
        T2_str = time.strftime("%H:%M:%S", time.localtime())
        print(f"\033[94m|{current_file}| END: {T2_str}\033[0m")
        
        # Calculate elapsed time in seconds and format it
        elapsed_seconds = T2 - T1
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        T = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        print(f"\n\033[94m|{current_file}| TOTAL RUN TIME: {T}\033[0m\n")
        return func_out

    return inner

# Add GPU availability check
def get_gpu_available():
    try:
        return cuda.is_available()
    except:
        return False

GPU_AVAILABLE = get_gpu_available()

def pad_or_truncate_list(lst, n=8):
    return lst[:n] + [None] * max(0, n - len(lst))

def pad_or_truncate_list2(lst, n=4):
    return lst[:n] + [None] * max(0, n - len(lst))

def get_data(metadata_file, ppid_column: str, session_column: str, trial_column: str, target_columns: list, save_processed_data=False, label_column: str = "predicted_label", chunk_size: int = 10000) -> tuple[pd.DataFrame, list]:
    target_columns2 = target_columns.copy()
    target_columns = [
        ppid_column,
        session_column,
        trial_column,
        "time",
        "ppid_vr",
        "session_vr",
        "trial_vr",
        "ppid_coda",
        "session_coda",
        "trial_coda",
        "experiment",
        "start_time",
        "end_time",
        "episode",
        "episode_set_name",
        "timeout",
        "rewards",
        "end_state",
        "reward",
        "bank",
        "distance",
        "threat_action",
        "weapons",
        "safehouse",
        "day_time"
    ]

    dictionary_target_columns = target_columns2

    # Add label_column to target columns if it's not already there
    if label_column not in target_columns:
        target_columns.append(label_column)

    target_columns = target_columns + dictionary_target_columns 
    target_columns = list(set(target_columns))

    # Define dtypes for problematic columns
    dtypes = {
        'time': 'float64',
        'start_time': 'float64',
        'end_time': 'float64',
        'timeout': 'float64',
        'rewards': 'float64',
        'reward': 'float64',
        'bank': 'float64',
        'distance': 'str',  # Changed from float64 to str since it contains 'close'
        'scenario': 'str',
        'threat': 'str',
        'visibility': 'str',
        'threat_action': 'str',
        'weapons': 'str',
        'safehouse': 'str',
        'day_time': 'str',
        label_column: 'str',
        ppid_column: 'str',
        session_column: 'str',
        trial_column: 'str',
        'episode': 'str',
        'episode_set_name': 'str',
        'experiment': 'str',
        'end_state': 'str'
    }

    # Read data in chunks with specified dtypes
    chunks = []
    for chunk in pd.read_csv(metadata_file, chunksize=chunk_size, dtype=dtypes, low_memory=False):
        # Check if label_column exists in the chunk
        if label_column not in chunk.columns:
            raise KeyError(f"Column '{label_column}' not found in the input file. Please check the column name or provide the correct label column name using --label parameter.")
        
        # Filter columns
        chunk = chunk[[col for col in target_columns if col in chunk.columns]]
        
        # Check if dictionary columns are already present
        columns_already_present = all(col in chunk.columns for col in dictionary_target_columns)
        
        # Process chunk
        if 'episode_set_name' in chunk.columns:
            chunk["episode_set_name"] = chunk["episode_set_name"].replace(
                {
                    "Dictionary of Actions": "dictionary",
                    'Dictionary of Actions Replication - Main': 'dictionary',
                    'Dictionary of Actions Replication - Fixed Trials': 'dictionary',
                    'Dictionary of Actions Replication - Stump Trials': 'dictionary',
                    "Dictionary of Actions - Instructed Actions": "instructed",
                }
            )
            
            if not columns_already_present:
                chunk["episode_split"] = chunk["episode"].str.split("_")
                dictionary_df = chunk.loc[chunk["episode_set_name"] == "dictionary"].copy()
                dictionary_df[dictionary_target_columns] = pd.DataFrame(
                    dictionary_df["episode_split"].apply(pad_or_truncate_list).to_list(), 
                    index=dictionary_df.index
                )
                chunk = dictionary_df.drop(columns=["episode_split", "dummy"])
        
        chunks.append(chunk)
    
    # Concatenate chunks
    df = pd.concat(chunks, ignore_index=True)
    
    # Reorder columns
    front_columns = [ppid_column, session_column, trial_column, "time"] 
    remaining_columns = [col for col in df.columns if col not in front_columns]
    df = df[front_columns + remaining_columns]

    if save_processed_data:
        df.to_csv(
            metadata_file.replace(".csv", "_processed.csv"), index=False
        )

    return df

def flatten_matrix(matrix):
    return matrix.values.flatten()

def align_matrices(matrix1, template_matrix):
    # Convert all syllables to strings
    matrix1.columns = matrix1.columns.astype(str)
    matrix1.index = matrix1.index.astype(str)
    template_matrix.columns = template_matrix.columns.astype(str)
    template_matrix.index = template_matrix.index.astype(str)
    
    # Create new matrix with the template structure
    aligned_matrix1 = template_matrix.copy()
    
    # Fill in the values from original matrix
    for syllable in matrix1.index:
        if syllable in aligned_matrix1.index:
            # Get the row for this syllable from the original matrix
            row = matrix1.loc[syllable]
            # Only copy values for syllables that exist in the original matrix
            for target_syllable in row.index:
                if target_syllable in aligned_matrix1.columns:
                    aligned_matrix1.loc[syllable, target_syllable] = float(row[target_syllable])
    
    return aligned_matrix1

def create_transition_matrix(data_subset, label_col, template_matrix):
    # Get sequences and label encoder
    sequences, le = encode_and_get_sequences(data_subset, label_col)
    
    # Create Markov chain model
    model = MarkovChain.from_samples(sequences)
    matrix = restructure_matrix(model)
    
    # Convert labels to strings and ensure they match template
    matrix.columns = matrix.columns.astype(str)
    matrix.index = matrix.index.astype(str)
    
    # Create a new matrix with the template structure
    new_matrix = template_matrix.copy()
    
    # Fill in the values from the actual transitions
    for syllable in matrix.columns:
        if syllable in new_matrix.columns:
            # Get the row for this syllable from the original matrix
            row = matrix.loc[syllable]
            # Only copy values for syllables that exist in the original matrix
            for target_syllable in row.index:
                if target_syllable in new_matrix.columns:
                    new_matrix.loc[syllable, target_syllable] = float(row[target_syllable])
    
    # Normalize the matrix to get probabilities
    new_matrix = new_matrix.div(new_matrix.sum(axis=1), axis=0).fillna(0)
    
    return new_matrix

def get_transition_matrix(data, label_column, output_dir, save_model: bool = True, save_matrix: bool = True, file_name_prefix: str = "transition_matrix", save_matrix_figure: bool = True, template_matrix=None, ppid_column: str = "ppid", session_column: str = "session", trial_column: str = "trial"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequences, le = encode_and_get_sequences(data, label_column)
    
    # Get number of trials instead of observations
    n_trials = len(data.groupby([ppid_column, session_column, trial_column]).groups)

    # Create a Markov chain model
    model = MarkovChain.from_samples(sequences)

    mat_dir = os.path.join(output_dir, "transition_matrices")
    os.makedirs(mat_dir, exist_ok=True)

    if save_model:
        # Save the model
        model_filename = os.path.join(mat_dir, f'{file_name_prefix}_model_with_self_transitions.pkl')
        dump(model, model_filename)
    matrix_with_self_transitions = restructure_matrix(model)

    encoded_labels = matrix_with_self_transitions.columns
    decoded_labels = le.inverse_transform(encoded_labels)
    matrix_with_self_transitions.columns = decoded_labels
    matrix_with_self_transitions.index = decoded_labels

    # If template matrix is provided, align the matrix with it
    if template_matrix is not None:
        matrix_with_self_transitions = align_matrices(matrix_with_self_transitions, template_matrix)

    if save_matrix_figure:
        plot_matrix(matrix_with_self_transitions, labels=matrix_with_self_transitions.columns, output_dir=output_dir, title=f"{file_name_prefix} including self-transitions (N={n_trials} trials)")
    if save_matrix:
        matrix_with_self_transitions.to_csv(os.path.join(mat_dir, f'{file_name_prefix}_with_self_transitions_N{n_trials}_trials.csv'))

    # Group by trial and remove self-transitions per trial
    cleaned_trials = []
    for (ppid, session, trial), trial_df in data.groupby([ppid_column, session_column, trial_column]):
        labels = trial_df[label_column].values
        if len(labels) == 0:
            continue
        new_seq = [labels[0]]
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                new_seq.append(labels[i])
        if len(new_seq) > 1:
            # Rebuild a DataFrame for this trial with correct columns
            cleaned_trials.append(pd.DataFrame({
                label_column: new_seq,
                ppid_column: [ppid] * len(new_seq),
                session_column: [session] * len(new_seq),
                trial_column: [trial] * len(new_seq)
            }))

    # Concatenate all cleaned trials
    if cleaned_trials:
        data2 = pd.concat(cleaned_trials, ignore_index=True)
    else:
        data2 = pd.DataFrame(columns=[label_column, ppid_column, session_column, trial_column])
    
    # Get number of trials without self-transitions
    n_trials_no_self = len(data2.groupby([ppid_column, session_column, trial_column]).groups)

    sequences2, le2 = encode_and_get_sequences(data2, label_column, le)
    # Create a Markov chain model
    model2 = MarkovChain.from_samples(sequences2)
    if save_model:
        # Save the model
        model_filename2 = os.path.join(mat_dir, f'{file_name_prefix}_model_without_self_transitions.pkl')
        dump(model2, model_filename2)

    matrix_without_self_transitions = restructure_matrix(model2)
    matrix_without_self_transitions.columns = decoded_labels
    matrix_without_self_transitions.index = decoded_labels

    # If template matrix is provided, align the matrix without self-transitions with it
    if template_matrix is not None:
        matrix_without_self_transitions = align_matrices(matrix_without_self_transitions, template_matrix)

    if save_matrix_figure:
        plot_matrix(matrix_without_self_transitions, labels=matrix_without_self_transitions.columns, output_dir=output_dir, title=f"{file_name_prefix} excluding self-transitions (N={n_trials_no_self} trials)")
    if save_matrix:
        matrix_without_self_transitions.to_csv(os.path.join(mat_dir, f'{file_name_prefix}_excluding_self_transitions_N{n_trials_no_self}_trials.csv'))

    # print("Original number of trials:", len(data.groupby([ppid_column, session_column, trial_column]).groups))
    # print("Number of trials after removing self-transitions:", len(data2.groupby([ppid_column, session_column, trial_column]).groups))
    # print("Original number of observations:", len(data))
    # print("Number of observations after removing self-transitions:", len(data2))

    return matrix_with_self_transitions, matrix_without_self_transitions

def set_random_seed(seed=None):
    """Set random seed for all random number generators"""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    # Ensure seed is within valid range
    seed = int(seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    return seed


# Example of how you might modify your existing graph creation:
def transition_matrix_to_graph(matrix, labels):
    G = nx.DiGraph()
    str_labels = [str(label) for label in labels]

    for label in str_labels:
        G.add_node(label)

    for i in range(len(str_labels)):
        for j in range(len(str_labels)):
            if matrix.iloc[i, j] > 0: # Or your condition for an edge
                G.add_edge(str_labels[i], str_labels[j], weight=float(matrix.iloc[i, j]))

    # Now prepare it
    G = prepare_graph_for_comparison(G, "from_matrix")
    return G


def process_group(group, dictionary_df, ppid_column, session_column, trial_column, label_column, template_matrix, output_dir, seed=None, metric_name='euclidean', grid_search=False, test_mode=False):
    # Set random seed for this process
    process_seed = set_random_seed(seed)
    print(f"\n\033[34mAnalyzing group: {group} (seed: {process_seed})\033[0m")
    
    # Get unique levels of the group, excluding NaN
    levels = dictionary_df[group].dropna().unique()
    print(f"\nAll levels found in {group}: {levels}")
    
    if len(levels) < 2:
        print(f"Skipping {group} - only one level present")
        return {}
        
    # Print data statistics for this group
    total_obs = len(dictionary_df)
    nan_obs = dictionary_df[group].isna().sum()
    non_nan_obs = total_obs - nan_obs
    print(f"\nData statistics for {group}:")
    print(f"Total observations: {total_obs}")
    print(f"NaN observations: {nan_obs} ({nan_obs/total_obs*100:.2f}%)")
    print(f"Non-NaN observations: {non_nan_obs} ({non_nan_obs/total_obs*100:.2f}%)")
    print(f"Unique levels: {len(levels)}")
    
    # Get trials for each level
    level_trials = {}
    level_transition_matrices = {}  # Store transition matrices for each level
    level_graphs = {}  # Store networkx graphs for each level
    
    for level in levels:
        level_data = dictionary_df[dictionary_df[group] == level]
        # print(f"\nProcessing level: {level}")
        # print(f"Number of samples for {level}: {len(level_data)}")
        
        if len(level_data) < 10:  # Skip levels with too few samples
            print(f"Warning: Skipping {level} due to insufficient samples ({len(level_data)} < 10)")
            continue
        
        # Group by trial
        trials = level_data.groupby([ppid_column, session_column, trial_column])
        trial_data_list = [group for _, group in trials]
        
        if len(trial_data_list) < 2:  # Need at least 2 trials for comparison
            print(f"Warning: Skipping {level} due to insufficient trials ({len(trial_data_list)} < 2)")
            continue
        
        try:
            # Calculate transition matrix for all trials in this level
            _, level_matrix = get_transition_matrix(
                level_data,  # Use all data for this level
                label_column,
                output_dir,
                save_model=False,
                save_matrix=True,
                file_name_prefix=f"level_matrix_{group}_{level}",
                template_matrix=template_matrix
            )
            
            # Align and normalize the level matrix
            aligned_level_matrix = align_matrices(level_matrix, template_matrix)
            row_sums = aligned_level_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            normalized_level_matrix = aligned_level_matrix.div(row_sums, axis=0)
            
            # Create networkx graph from the transition matrix
            level_graph = transition_matrix_to_graph(normalized_level_matrix, normalized_level_matrix.columns)
            
            level_trials[level] = trial_data_list
            level_transition_matrices[level] = normalized_level_matrix
            level_graphs[level] = level_graph
            print(f"Successfully processed level {level} with {len(trial_data_list)} trials")
        except Exception as e:
            print(f"Error processing level {level}: {str(e)}")
            continue
    
    # print(f"\nSuccessfully processed levels: {list(level_transition_matrices.keys())}")
    
    if len(level_transition_matrices) < 2:
        print(f"Skipping {group} - less than 2 valid levels after filtering")
        return {}
    
    # Get valid levels that have both transition matrices and trials
    valid_levels = [level for level in level_transition_matrices.keys() 
                   if level in level_trials and len(level_trials[level]) >= 2]
    
    if len(valid_levels) < 2:
        print(f"Skipping {group} - less than 2 valid levels with sufficient trials")
        return {}
    
    # Store reference graphs for each level
    level_graphs = {}
    for level in valid_levels:
        level_matrix = level_transition_matrices[level]
        level_graphs[level] = transition_matrix_to_graph(level_matrix, level_matrix.columns)
    
    # Perform pairwise comparisons
    level_pairs = [(a, b) for i, a in enumerate(valid_levels) for b in valid_levels[i+1:]]
    if test_mode:
        level_pairs = level_pairs[:2]  # Only process first 2 pairs for testing
    # print(f"\nLevel pairs to compare: {level_pairs}")
    group_results = {}
    
    for level1, level2 in tqdm(level_pairs, desc=f'Pairwise comparisons for {group}'):
        # print(f"\nComparing {level1} vs {level2}")
        # print(f"Number of trials in {level1}: {len(level_trials[level1])}")
        # print(f"Number of trials in {level2}: {len(level_trials[level2])}")
        # print(f"Level graphs: {level_graphs}")
        # print(f"Level graph {level1}: {level_graphs[level1]}")
        # print(f"Level graph {level2}: {level_graphs[level2]}")
        
        # Set a new seed for each comparison
        comparison_hash = hash(f"{group}_{level1}_{level2}")
        comparison_seed = (process_seed + abs(comparison_hash)) % (2**32)
        print(f"Using seed {comparison_seed} for comparison {level1} vs {level2}")
        
        # Get the pre-calculated transition matrices for each level
        ref_matrix1 = level_transition_matrices[level1]
        ref_matrix2 = level_transition_matrices[level2]
        
        # Calculate network distance using stored graphs
        graphs_dict = {level1: level_graphs[level1], level2: level_graphs[level2]}
        try:
            (_, _), network_distance = compare_networks(graphs_dict)
        except Exception as e:
            print(f"Warning: Network comparison failed: {str(e)}")
            network_distance = None
        
        # Process individual trials for distance calculations
        all_trials1 = []
        for trial_data in level_trials[level1]:
            _, trial_matrix = get_transition_matrix(
                trial_data,
                label_column,
                output_dir,
                save_model=False,
                save_matrix=False,
                file_name_prefix=f"trial_matrix_{group}_{level1}",
                save_matrix_figure=False,
                template_matrix=template_matrix
            )
            aligned_trial_matrix = align_matrices(trial_matrix, template_matrix)
            row_sums = aligned_trial_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1
            normalized_trial_matrix = aligned_trial_matrix.div(row_sums, axis=0)
            all_trials1.append(normalized_trial_matrix.values)
        
        all_trials2 = []
        for trial_data in level_trials[level2]:
            _, trial_matrix = get_transition_matrix(
                trial_data,
                label_column,
                output_dir,
                save_model=False,
                save_matrix=False,
                file_name_prefix=f"trial_matrix_{group}_{level2}",
                save_matrix_figure=False,
                template_matrix=template_matrix
            )
            aligned_trial_matrix = align_matrices(trial_matrix, template_matrix)
            row_sums = aligned_trial_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1
            normalized_trial_matrix = aligned_trial_matrix.div(row_sums, axis=0)
            all_trials2.append(normalized_trial_matrix.values)
        
        # Convert to numpy arrays
        all_trials1 = np.array(all_trials1)
        all_trials2 = np.array(all_trials2)
        
        if grid_search:
            metrics = ['euclidean', 'manhattan', 'cosine', 'correlation', 'kl_divergence', 'graph_edit']
            metric_results = {}
            
            for metric in metrics:
                # print(f"\nTesting metric: {metric}")
                
                if metric == 'graph_edit':
                    # Use the pre-calculated graphs and matrices
                    graphs_dict = {level1: level_graphs[level1], level2: level_graphs[level2]}
                    try:
                        (_, _), network_distance = compare_networks(graphs_dict)
                    except Exception as e:
                        print(f"Warning: Network comparison failed: {str(e)}")
                        network_distance = None
                    
                    # For graph edit distance, we already have the network_distance
                    if network_distance is None:
                        print("Warning: Network distance calculation failed, skipping graph edit distance metric")
                        continue
                    
                    # Store the graph edit distance as the observed difference
                    observed_diff = network_distance
                    
                    # Combine all trials for permutation
                    all_trials = np.concatenate([all_trials1, all_trials2])
                    all_trial_data = level_trials[level1] + level_trials[level2]
                    n1 = len(all_trials1)
                    
                    # Perform permutations for graph edit distance
                    perm_diffs = []
                    for _ in range(1000):  # Number of permutations
                        # Create a random permutation of indices
                        perm_indices = np.random.permutation(len(all_trial_data))
                        
                        # Split into two groups using indices
                        perm_indices1 = perm_indices[:n1]
                        perm_indices2 = perm_indices[n1:]
                        
                        # Get permuted trial data
                        perm_data1 = pd.concat([all_trial_data[i] for i in perm_indices1])
                        perm_data2 = pd.concat([all_trial_data[i] for i in perm_indices2])
                        
                        # Calculate transition matrices for permuted groups
                        _, perm_matrix1 = get_transition_matrix(
                            perm_data1,
                            label_column,
                            output_dir,
                            save_model=False,
                            save_matrix=False,
                            file_name_prefix=f"perm_matrix_{group}_{level1}",
                            save_matrix_figure=False,
                            template_matrix=template_matrix
                        )
                        aligned_perm_matrix1 = align_matrices(perm_matrix1, template_matrix)
                        
                        _, perm_matrix2 = get_transition_matrix(
                            perm_data2,
                            label_column,
                            output_dir,
                            save_model=False,
                            save_matrix=False,
                            file_name_prefix=f"perm_matrix_{group}_{level2}",
                            save_matrix_figure=False,
                            template_matrix=template_matrix
                        )
                        aligned_perm_matrix2 = align_matrices(perm_matrix2, template_matrix)
                        
                        # Create graphs for permuted matrices
                        perm_graph1 = transition_matrix_to_graph(aligned_perm_matrix1, aligned_perm_matrix1.columns)
                        perm_graph2 = transition_matrix_to_graph(aligned_perm_matrix2, aligned_perm_matrix2.columns)
                        
                        # Calculate graph edit distance for permuted groups
                        try:
                            (_, _), perm_ged = compare_networks({level1: perm_graph1, level2: perm_graph2})
                            perm_diffs.append(perm_ged)
                        except Exception as e:
                            print(f"Warning: Failed to calculate graph edit distance for permutation: {str(e)}")
                            continue
                    
                    if not perm_diffs:
                        print("Warning: No valid permutations calculated for graph edit distance")
                        continue
                    
                    # Calculate p-value using two-tailed test
                    p_value = np.mean(np.abs(np.array(perm_diffs)) >= np.abs(observed_diff))
                    
                    # Calculate effect size (Cohen's d)
                    std_dist1 = np.std(perm_diffs)  # Use permutation distribution for std
                    std_dist2 = np.std(perm_diffs)  # Use same std for both conditions
                    pooled_std = np.sqrt((std_dist1**2 + std_dist2**2) / 2)
                    cohens_d = observed_diff / pooled_std if pooled_std != 0 else 0
                    
                    # Store results for graph edit distance
                    metric_results[metric] = {
                        "observed_difference": observed_diff,
                        "p_value": p_value,
                        "effect_size": cohens_d,
                        "permutation_differences": perm_diffs,
                        "network_distance": network_distance,
                        "condition1_stats": {
                            "n_observations": len(pd.concat(level_trials[level1])),
                            "n_trials": len(level_trials[level1]),
                            "mean_distance": None,
                            "std_distance": None
                        },
                        "condition2_stats": {
                            "n_observations": len(pd.concat(level_trials[level2])),
                            "n_trials": len(level_trials[level2]),
                            "mean_distance": None,
                            "std_distance": None
                        },
                        "distances1": distances1 if metric != 'graph_edit' else None,
                        "distances2": distances2 if metric != 'graph_edit' else None,
                        "matrix1": ref_matrix1.values if hasattr(ref_matrix1, 'values') else ref_matrix1,
                        "matrix2": ref_matrix2.values if hasattr(ref_matrix2, 'values') else ref_matrix2,
                        "graph1": level_graphs[level1],
                        "graph2": level_graphs[level2],
                    }
                    continue
                
                # For other metrics, proceed with the existing distance calculations
                distances1 = calculate_distances(all_trials1, ref_matrix1.values, metric)
                distances2 = calculate_distances(all_trials2, ref_matrix2.values, metric)
                
                if len(distances1) == 0 or len(distances2) == 0:
                    print(f"Warning: No valid distances calculated for metric {metric}")
                    continue
                
                # Calculate mean distance for each condition
                median_dist1 = np.median(distances1)
                median_dist2 = np.median(distances2)
                observed_diff = median_dist1 - median_dist2
                
                # Calculate additional statistics
                std_dist1 = np.std(distances1)
                std_dist2 = np.std(distances2)
                
                # Combine all trials for permutation
                all_trials = np.concatenate([all_trials1, all_trials2])
                all_trial_data = level_trials[level1] + level_trials[level2]
                n1 = len(all_trials1)
                
                # Perform permutations
                perm_diffs = []
                for _ in range(1000):  # Number of permutations
                    # Create a random permutation of indices
                    perm_indices = np.random.permutation(len(all_trials))
                    
                    # Split into two groups using indices
                    perm_indices1 = perm_indices[:n1]
                    perm_indices2 = perm_indices[n1:]
                    
                    # Get permuted trial matrices
                    perm_trials1 = [all_trials[i] for i in perm_indices1]
                    perm_trials2 = [all_trials[i] for i in perm_indices2]
                    
                    # Calculate distances using current metric
                    perm_dist1 = calculate_distances(np.array(perm_trials1), ref_matrix1.values, metric)
                    perm_dist2 = calculate_distances(np.array(perm_trials2), ref_matrix2.values, metric)
                    
                    if len(perm_dist1) == 0 or len(perm_dist2) == 0:
                        continue
                    
                    perm_median1 = np.median(perm_dist1)
                    perm_median2 = np.median(perm_dist2)
                    perm_diff = perm_median1 - perm_median2
                    perm_diffs.append(perm_diff)
                
                if not perm_diffs:
                    print(f"Warning: No valid permutations calculated for metric {metric}")
                    continue
                
                # Calculate p-value using two-tailed test
                p_value = np.mean(np.abs(np.array(perm_diffs)) >= np.abs(observed_diff))
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((std_dist1**2 + std_dist2**2) / 2)
                cohens_d = observed_diff / pooled_std if pooled_std != 0 else 0

                # print(f"Group: {group}, Level 1: {level1}, Level 2: {level2}, Metric: {metric_name}")
                # print(f"Observed Difference: {observed_diff}, P-Value: {p_value}, Effect Size: {cohens_d}")
                # print(f"Permutation Differences: {perm_diffs}")
                # print(f"Network Distance: {network_distance}")
                # print(f"Condition 1 Stats: {median_dist1}, {std_dist1}")
                # print(f"Condition 2 Stats: {median_dist2}, {std_dist2}")
                # print(f"Distances 1: {distances1}")
                # print(f"Distances 2: {distances2}")
                # print(f"Matrix 1: {ref_matrix1.values if hasattr(ref_matrix1, 'values') else ref_matrix1}")
                # print(f"Matrix 2: {ref_matrix2.values if hasattr(ref_matrix2, 'values') else ref_matrix2}")
                # print(f"Graph 1: {level_graphs[level1]}")
                # print(f"Graph 2: {level_graphs[level2]}")  
                
                # Store results for this metric
                metric_results[metric] = {
                    "observed_difference": observed_diff,
                    "p_value": p_value,
                    "effect_size": cohens_d,
                    "permutation_differences": perm_diffs,
                    "network_distance": network_distance,
                    "condition1_stats": {
                        "n_observations": len(pd.concat(level_trials[level1])),
                        "n_trials": len(level_trials[level1]),
                        "mean_distance": median_dist1,
                        "std_distance": std_dist1
                    },
                    "condition2_stats": {
                        "n_observations": len(pd.concat(level_trials[level2])),
                        "n_trials": len(level_trials[level2]),
                        "mean_distance": median_dist2,
                        "std_distance": std_dist2
                    },
                    "distances1": distances1 if metric != 'graph_edit' else None,
                    "distances2": distances2 if metric != 'graph_edit' else None,
                    "matrix1": ref_matrix1.values if hasattr(ref_matrix1, 'values') else ref_matrix1,
                    "matrix2": ref_matrix2.values if hasattr(ref_matrix2, 'values') else ref_matrix2,
                    "graph1": level_graphs[level1],
                    "graph2": level_graphs[level2],
                }
            
            # Store all metric results for this comparison
            group_results[f"{group}_{level1}_vs_{level2}"] = metric_results
            # print(f"Group results: {group_results}")
            
        else:
            # Original single metric code
            distances1 = calculate_distances(all_trials1, ref_matrix1.values, metric_name)
            distances2 = calculate_distances(all_trials2, ref_matrix2.values, metric_name)
            
            if len(distances1) == 0 or len(distances2) == 0:
                print("Warning: No valid distances calculated")
                continue
            
            # Calculate mean distance for each condition
            median_dist1 = np.median(distances1)
            median_dist2 = np.median(distances2)
            observed_diff = median_dist1 - median_dist2
            
            # Calculate additional statistics
            std_dist1 = np.std(distances1)
            std_dist2 = np.std(distances2)
            
            # Combine all trials for permutation
            all_trials = np.concatenate([all_trials1, all_trials2])
            all_trial_data = level_trials[level1] + level_trials[level2]
            n1 = len(all_trials1)
            
            # Perform permutations
            perm_diffs = []
            for _ in range(1000):  # Number of permutations
                # Create a random permutation of indices
                perm_indices = np.random.permutation(len(all_trials))
                
                # Split into two groups using indices
                perm_indices1 = perm_indices[:n1]
                perm_indices2 = perm_indices[n1:]
                
                # Get the permuted trial data
                perm_data1 = pd.concat([all_trial_data[i] for i in perm_indices1])
                perm_data2 = pd.concat([all_trial_data[i] for i in perm_indices2])
                
                # Calculate new reference matrices and graphs for permuted groups
                _, perm_ref1 = get_transition_matrix(
                    perm_data1,
                    label_column,
                    output_dir,
                    save_model=False,
                    save_matrix=False,
                    file_name_prefix=f"perm_ref_matrix_{group}_{level1}",
                    save_matrix_figure=False,
                    template_matrix=template_matrix
                )
                aligned_perm_ref1 = align_matrices(perm_ref1, template_matrix)
                perm_ref_graph1 = transition_matrix_to_graph(aligned_perm_ref1, template_matrix.columns)
                
                _, perm_ref2 = get_transition_matrix(
                    perm_data2,
                    label_column,
                    output_dir,
                    save_model=False,
                    save_matrix=False,
                    file_name_prefix=f"perm_ref_matrix_{group}_{level2}",
                    save_matrix_figure=False,
                    template_matrix=template_matrix
                )
                aligned_perm_ref2 = align_matrices(perm_ref2, template_matrix)
                perm_ref_graph2 = transition_matrix_to_graph(aligned_perm_ref2, template_matrix.columns)
                
                # Get permuted trial matrices
                perm_trials1 = [all_trials[i] for i in perm_indices1]
                perm_trials2 = [all_trials[i] for i in perm_indices2]
                
                # Calculate distances using current metric
                perm_dist1 = calculate_distances(np.array(perm_trials1), ref_matrix1, metric)
                perm_dist2 = calculate_distances(np.array(perm_trials2), ref_matrix2, metric)
                
                if len(perm_dist1) == 0 or len(perm_dist2) == 0:
                    continue
                
                perm_median1 = np.median(perm_dist1)
                perm_median2 = np.median(perm_dist2)
                perm_diff = perm_median1 - perm_median2
                
                perm_diffs.append(perm_diff)
            
            if not perm_diffs:
                print(f"Warning: No valid permutations calculated for metric {metric}")
                continue
            
            # Calculate p-value using two-tailed test
            p_value = np.mean(np.abs(np.array(perm_diffs)) >= np.abs(observed_diff))
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((std_dist1**2 + std_dist2**2) / 2)
            cohens_d = observed_diff / pooled_std if pooled_std != 0 else 0
      
            # Store results for this metric
            group_results[f"{group}_{level1}_vs_{level2}"] = {
                metric_name: {
                    "observed_difference": observed_diff,
                    "p_value": p_value,
                    "effect_size": cohens_d,
                    "permutation_differences": perm_diffs,
                    "network_distance": network_distance,
                    "condition1_stats": {
                        "n_observations": len(pd.concat(level_trials[level1])),
                        "n_trials": len(level_trials[level1]),
                        "mean_distance": median_dist1,
                        "std_distance": std_dist1
                    },
                    "condition2_stats": {
                        "n_observations": len(pd.concat(level_trials[level2])),
                        "n_trials": len(level_trials[level2]),
                        "mean_distance": median_dist2,
                        "std_distance": std_dist2
                    },
                    "distances1": distances1,
                    "distances2": distances2,
                    "matrix1": ref_matrix1.values if hasattr(ref_matrix1, 'values') else ref_matrix1,
                    "matrix2": ref_matrix2.values if hasattr(ref_matrix2, 'values') else ref_matrix2,
                    "graph1": level_graphs[level1],
                    "graph2": level_graphs[level2],
                }
            }
    
    return group_results

def plot_distance_distributions(distances1, distances2, condition1_name, condition2_name, output_dir, prefix):
    """Plot the distribution of distances for each condition."""
    plt.figure(figsize=(10, 6))
    
    # Create kernel density estimates
    kde1 = gaussian_kde(distances1)
    kde2 = gaussian_kde(distances2)
    
    # Create x values for plotting
    x_min = min(min(distances1), min(distances2))
    x_max = max(max(distances1), max(distances2))
    x = np.linspace(x_min, x_max, 1000)
    
    # Plot histograms and KDEs
    plt.hist(distances1, bins=30, alpha=0.5, density=True, label=f'{condition1_name} (N={len(distances1)} trials)')
    plt.hist(distances2, bins=30, alpha=0.5, density=True, label=f'{condition2_name} (N={len(distances2)} trials)')
    plt.plot(x, kde1(x), label=f'{condition1_name} KDE')
    plt.plot(x, kde2(x), label=f'{condition2_name} KDE')
    
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Distance Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{prefix}_distance_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_matrix(matrix, labels, output_dir, title, vmin=0, vmax=1):
    """Plot a transition matrix with consistent color scaling."""
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap with fixed color scale
    sns.heatmap(matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='plasma',
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'label': 'Transition Probability'})
    
    plt.title(title)
    plt.xlabel('To State')
    plt.ylabel('From State')
    
    # Save the plot
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{title.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_transition_difference_heatmap(matrix1, matrix2, labels, output_dir, prefix):
    """Plot a heatmap showing the difference in transition probabilities between conditions."""
    # Calculate the difference matrix
    diff_matrix = matrix1 - matrix2
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Plot the heatmap with fixed color scale
    sns.heatmap(diff_matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Probability Difference'})
    
    plt.title('Difference in Transition Probabilities')
    plt.xlabel('To State')
    plt.ylabel('From State')
    
    # Save the plot
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{prefix}_transition_differences.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_transition_network(matrix, labels, output_dir, prefix, threshold=0.1):
    """Create a network visualization of the transition matrix."""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, label in enumerate(labels):
        G.add_node(label)
    
    # Add edges with weights
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix.iloc[i, j] > threshold:
                G.add_edge(labels[i], labels[j], weight=matrix.iloc[i, j])
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Calculate node positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, 
                          width=[G[u][v]['weight'] * 3 for u, v in G.edges()],
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20)
    nx.draw_networkx_labels(G, pos)
    
    plt.title('Transition Network')
    plt.axis('off')
    
    # Save the plot
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{prefix}_transition_network.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_patterns(data, label_column, output_dir, prefix, window_size=10):
    """Plot temporal patterns of transitions."""
    # Convert data to sequence
    sequence = data[label_column].values
    
    # Get number of trials
    n_trials = len(data.groupby(['ppid', 'session', 'trial']).groups)
    
    # Create sliding window analysis
    n_windows = len(sequence) - window_size + 1
    transition_counts = []
    
    for i in range(n_windows):
        window = sequence[i:i+window_size]
        transitions = [(window[j], window[j+1]) for j in range(len(window)-1)]
        transition_counts.append(transitions)
    
    # Count transitions in each window
    unique_transitions = set()
    for window in transition_counts:
        unique_transitions.update(window)
    
    # Create transition matrix over time
    time_matrix = np.zeros((n_windows, len(unique_transitions)))
    for i, window in enumerate(transition_counts):
        for j, transition in enumerate(sorted(unique_transitions)):
            time_matrix[i, j] = window.count(transition)
    
    # Plot the temporal pattern
    plt.figure(figsize=(15, 8))
    plt.imshow(time_matrix.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Transition Count')
    plt.xlabel('Time Window')
    plt.ylabel('Transition Type')
    plt.title(f'Temporal Pattern of Transitions (N={n_trials} trials)')
    
    # Save the plot
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{prefix}_temporal_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_transition_entropy(matrix):
    """Calculate the entropy of transition probabilities."""
    # Flatten the matrix and remove zeros
    probs = matrix.values.flatten()
    probs = probs[probs > 0]
    return entropy(probs)

def calculate_mutual_information(matrix):
    """Calculate the mutual information between states."""
    # Convert matrix to joint probability distribution
    joint_prob = matrix.values / matrix.values.sum()
    # Calculate marginal probabilities
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    # Calculate mutual information
    mi = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if joint_prob[i,j] > 0:
                mi += joint_prob[i,j] * np.log2(joint_prob[i,j] / (p_x[i] * p_y[j]))
    return mi

def calculate_transition_regularity(matrix):
    """Calculate the regularity of transitions."""
    # Calculate the variance of transition probabilities
    return np.var(matrix.values.flatten())

def calculate_sequence_complexity(sequence):
    """Calculate the complexity of a sequence using Lempel-Ziv complexity."""
    n = len(sequence)
    if n == 0:
        return 0
    
    # Convert sequence to string for processing
    seq_str = ''.join(map(str, sequence))
    
    # Initialize variables
    complexity = 1
    pos = 1
    sub = seq_str[0]
    
    while pos < n:
        # Try to extend the current substring
        for l in range(1, n - pos + 1):
            sub_new = seq_str[pos:pos + l]
            if sub_new not in seq_str[:pos]:
                break
        else:
            l = 1
        
        pos += l
        complexity += 1
        sub = seq_str[:pos]
    
    # Normalize by sequence length
    return complexity / n

def analyze_transition_statistics(matrix1, matrix2, sequence1, sequence2, output_dir, prefix):
    """Calculate and save various transition statistics."""
    transition_stats = {}
    
    # Calculate entropies
    transition_stats['entropy1'] = calculate_transition_entropy(matrix1)
    transition_stats['entropy2'] = calculate_transition_entropy(matrix2)
    transition_stats['entropy_diff'] = transition_stats['entropy1'] - transition_stats['entropy2']
    
    # Calculate mutual information
    transition_stats['mi1'] = calculate_mutual_information(matrix1)
    transition_stats['mi2'] = calculate_mutual_information(matrix2)
    transition_stats['mi_diff'] = transition_stats['mi1'] - transition_stats['mi2']
    
    # Calculate transition regularity
    transition_stats['regularity1'] = calculate_transition_regularity(matrix1)
    transition_stats['regularity2'] = calculate_transition_regularity(matrix2)
    transition_stats['regularity_diff'] = transition_stats['regularity1'] - transition_stats['regularity2']
    
    # Calculate sequence complexity
    transition_stats['complexity1'] = calculate_sequence_complexity(sequence1)
    transition_stats['complexity2'] = calculate_sequence_complexity(sequence2)
    transition_stats['complexity_diff'] = transition_stats['complexity1'] - transition_stats['complexity2']
    
    # Perform statistical tests
    # Flatten matrices for comparison
    flat1 = matrix1.values.flatten()
    flat2 = matrix2.values.flatten()
    
    # Mann-Whitney U test
    transition_stats['mannwhitneyu_stat'], transition_stats['mannwhitneyu_p'] = mannwhitneyu(flat1, flat2)
    
    # Kolmogorov-Smirnov test
    transition_stats['ks_stat'], transition_stats['ks_p'] = ks_2samp(flat1, flat2)
    
    # Save statistics to file
    stats_df = pd.DataFrame([transition_stats])
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(results_dir, f'{prefix}_transition_statistics.csv'), index=False)
    
    return transition_stats

def combine_transition_matrices(matrices):
    """Combine multiple transition matrices by summing their values and normalizing."""
    if not matrices:
        return None
    
    # Sum all matrices
    combined = np.sum(matrices, axis=0)
    
    # Normalize by row sums
    row_sums = combined.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = combined / row_sums
    
    return normalized

def process_trials_chunk(trials_chunk, label_col, output_dir, prefix, template_matrix):
    chunk_results = []
    for trial_data in trials_chunk:
        # Calculate transition matrix for this trial
        _, trial_matrix = get_transition_matrix(
            trial_data, 
            label_col, 
            output_dir, 
            save_model=False, 
            save_matrix=False, 
            file_name_prefix=prefix,
            save_matrix_figure=False,
            template_matrix=template_matrix
        )
        
        # Align with template and normalize
        aligned_trial_matrix = align_matrices(trial_matrix, template_matrix)
        
        # Ensure proper normalization
        row_sums = aligned_trial_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_matrix = aligned_trial_matrix.div(row_sums, axis=0)
        
        chunk_results.append(normalized_matrix.values)
    
    return chunk_results

def permutation_test(condition1_trials, condition2_trials, template_matrix, label_col, output_dir, group, level1, level2, n_permutations=1000, chunk_size=1000, seed=None, metric_name='euclidean', grid_search=False):
    # Set random seed for this test
    set_random_seed(seed)
    
    print(f"Template matrix shape: {template_matrix.shape}")
    
    # Create output directory for reference matrices
    ref_matrices_dir = os.path.join(output_dir, "reference_matrices")
    os.makedirs(ref_matrices_dir, exist_ok=True)
    
    # Concatenate all trials for each condition
    condition1_data = pd.concat(condition1_trials)
    condition2_data = pd.concat(condition2_trials)
    
    # Get number of observations for each condition
    n_obs1 = len(condition1_data)
    n_obs2 = len(condition2_data)
    
    # Get reference matrices (both with and without self-transitions)
    ref_matrix1_with, ref_matrix1_without = get_transition_matrix(
        condition1_data, 
        label_col, 
        ref_matrices_dir, 
        save_model=False, 
        save_matrix=True, 
        file_name_prefix=f"ref_matrix_{group}_{level1}",
        template_matrix=template_matrix
    )
    
    ref_matrix2_with, ref_matrix2_without = get_transition_matrix(
        condition2_data, 
        label_col, 
        ref_matrices_dir, 
        save_model=False, 
        save_matrix=True, 
        file_name_prefix=f"ref_matrix_{group}_{level2}",
        template_matrix=template_matrix
    )
    
    # Align matrices without self-transitions with template for the permutation test
    aligned_ref_matrix1 = align_matrices(ref_matrix1_without, template_matrix)
    aligned_ref_matrix2 = align_matrices(ref_matrix2_without, template_matrix)
    
    # Create reference graphs for graph edit distance
    ref_graph1 = transition_matrix_to_graph(aligned_ref_matrix1, template_matrix.columns)
    ref_graph2 = transition_matrix_to_graph(aligned_ref_matrix2, template_matrix.columns)
    
    # Process trials in chunks
    all_trials1 = []
    for chunk in [condition1_trials[i:i + chunk_size] for i in range(0, len(condition1_trials), chunk_size)]:
        chunk_results = process_trials_chunk(chunk, label_col, output_dir, f"trial_matrix_{group}_{level1}", template_matrix)
        all_trials1.extend(chunk_results)
    
    all_trials2 = []
    for chunk in [condition2_trials[i:i + chunk_size] for i in range(0, len(condition2_trials), chunk_size)]:
        chunk_results = process_trials_chunk(chunk, label_col, output_dir, f"trial_matrix_{group}_{level2}", template_matrix)
        all_trials2.extend(chunk_results)
    
    # Convert to numpy arrays for Numba
    all_trials1 = np.array(all_trials1)
    all_trials2 = np.array(all_trials2)
    ref_matrix1 = aligned_ref_matrix1.values
    ref_matrix2 = aligned_ref_matrix2.values
    
    # Create reference graphs for each level
    level_graphs = {
        level1: transition_matrix_to_graph(aligned_ref_matrix1, template_matrix.columns),
        level2: transition_matrix_to_graph(aligned_ref_matrix2, template_matrix.columns)
    }
    
    if grid_search:
        metrics = ['euclidean', 'manhattan', 'cosine', 'correlation', 'kl_divergence', 'graph_edit']
        metric_results = {}
        
        for metric in metrics:
            print(f"\nTesting metric: {metric}")
            
            if metric == 'graph_edit':
                # Use the pre-calculated graphs and matrices
                graphs_dict = {level1: level_graphs[level1], level2: level_graphs[level2]}
                try:
                    (_, _), network_distance = compare_networks(graphs_dict)
                except Exception as e:
                    print(f"Warning: Network comparison failed: {str(e)}")
                    network_distance = None
                
                # For graph edit distance, we already have the network_distance
                if network_distance is None:
                    print("Warning: Network distance calculation failed, skipping graph edit distance metric")
                    continue
                
                # Store the graph edit distance as the observed difference
                observed_diff = network_distance
                
                # Combine all trials for permutation
                all_trials = np.concatenate([all_trials1, all_trials2])
                all_trial_data = condition1_trials + condition2_trials
                n1 = len(all_trials1)
                
                # Perform permutations for graph edit distance
                perm_diffs = []
                for _ in range(1000):  # Number of permutations
                    # Create a random permutation of indices
                    perm_indices = np.random.permutation(len(all_trial_data))
                    
                    # Split into two groups using indices
                    perm_indices1 = perm_indices[:n1]
                    perm_indices2 = perm_indices[n1:]
                    
                    # Get permuted trial data
                    perm_data1 = pd.concat([all_trial_data[i] for i in perm_indices1])
                    perm_data2 = pd.concat([all_trial_data[i] for i in perm_indices2])
                    
                    # Calculate transition matrices for permuted groups
                    _, perm_matrix1 = get_transition_matrix(
                        perm_data1,
                        label_col,
                        output_dir,
                        save_model=False,
                        save_matrix=False,
                        file_name_prefix=f"perm_matrix_{group}_{level1}",
                        save_matrix_figure=False,
                        template_matrix=template_matrix
                    )
                    aligned_perm_matrix1 = align_matrices(perm_matrix1, template_matrix)
                    
                    _, perm_matrix2 = get_transition_matrix(
                        perm_data2,
                        label_col,
                        output_dir,
                        save_model=False,
                        save_matrix=False,
                        file_name_prefix=f"perm_matrix_{group}_{level2}",
                        save_matrix_figure=False,
                        template_matrix=template_matrix
                    )
                    aligned_perm_matrix2 = align_matrices(perm_matrix2, template_matrix)
                    
                    # Create graphs for permuted matrices
                    perm_graph1 = transition_matrix_to_graph(aligned_perm_matrix1, aligned_perm_matrix1.columns)
                    perm_graph2 = transition_matrix_to_graph(aligned_perm_matrix2, aligned_perm_matrix2.columns)
                    
                    # Calculate graph edit distance for permuted groups
                    try:
                        (_, _), perm_ged = compare_networks({level1: perm_graph1, level2: perm_graph2})
                        perm_diffs.append(perm_ged)
                    except Exception as e:
                        print(f"Warning: Failed to calculate graph edit distance for permutation: {str(e)}")
                        continue
                
                if not perm_diffs:
                    print("Warning: No valid permutations calculated for graph edit distance")
                    continue
                
                # Calculate p-value using two-tailed test
                p_value = np.mean(np.abs(np.array(perm_diffs)) >= np.abs(observed_diff))
                
                # Calculate effect size (Cohen's d)
                std_dist1 = np.std(perm_diffs)  # Use permutation distribution for std
                std_dist2 = np.std(perm_diffs)  # Use same std for both conditions
                pooled_std = np.sqrt((std_dist1**2 + std_dist2**2) / 2)
                cohens_d = observed_diff / pooled_std if pooled_std != 0 else 0
                
                # Store results for graph edit distance
                metric_results[metric] = {
                    "observed_difference": observed_diff,
                    "p_value": p_value,
                    "effect_size": cohens_d,
                    "permutation_differences": perm_diffs,
                    "network_distance": observed_diff,  # Use observed_diff instead of network_distance
                    "condition1_stats": {
                        "n_observations": len(pd.concat(condition1_trials)),
                        "n_trials": len(condition1_trials),
                        "mean_distance": None,
                        "std_distance": None
                    },
                    "condition2_stats": {
                        "n_observations": len(pd.concat(condition2_trials)),
                        "n_trials": len(condition2_trials),
                        "mean_distance": None,
                        "std_distance": None
                    },
                    "distances1": distances1 if metric != 'graph_edit' else None,
                    "distances2": distances2 if metric != 'graph_edit' else None,
                    "matrix1": ref_matrix1.values if hasattr(ref_matrix1, 'values') else ref_matrix1,
                    "matrix2": ref_matrix2.values if hasattr(ref_matrix2, 'values') else ref_matrix2,
                    "graph1": level_graphs[level1],
                    "graph2": level_graphs[level2],
                }
                continue
            
            # For other metrics, proceed with the existing distance calculations
            distances1 = calculate_distances(all_trials1, ref_matrix1, metric)
            distances2 = calculate_distances(all_trials2, ref_matrix2, metric)
            
            if len(distances1) == 0 or len(distances2) == 0:
                print(f"Warning: No valid distances calculated for metric {metric}")
                continue
            
            # Calculate mean distance for each condition
            median_dist1 = np.median(distances1)
            median_dist2 = np.median(distances2)
            observed_diff = median_dist1 - median_dist2
            
            # Calculate additional statistics
            std_dist1 = np.std(distances1)
            std_dist2 = np.std(distances2)
            
            # Combine all trials for permutation
            all_trials = np.concatenate([all_trials1, all_trials2])
            all_trial_data = condition1_trials + condition2_trials
            n1 = len(all_trials1)
            
            # Perform permutations
            perm_diffs = []
            for _ in range(n_permutations):
                # Create a random permutation of indices
                perm_indices = np.random.permutation(len(all_trials))
                
                # Split into two groups using indices
                perm_indices1 = perm_indices[:n1]
                perm_indices2 = perm_indices[n1:]
                
                # Get the permuted trial data
                perm_data1 = pd.concat([all_trial_data[i] for i in perm_indices1])
                perm_data2 = pd.concat([all_trial_data[i] for i in perm_indices2])
                
                # Calculate new reference matrices and graphs for permuted groups
                _, perm_ref1 = get_transition_matrix(
                    perm_data1,
                    label_col,
                    output_dir,
                    save_model=False,
                    save_matrix=False,
                    file_name_prefix=f"perm_ref_matrix_{group}_{level1}",
                    save_matrix_figure=False,
                    template_matrix=template_matrix
                )
                aligned_perm_ref1 = align_matrices(perm_ref1, template_matrix)
                perm_ref_graph1 = transition_matrix_to_graph(aligned_perm_ref1, template_matrix.columns)
                
                _, perm_ref2 = get_transition_matrix(
                    perm_data2,
                    label_col,
                    output_dir,
                    save_model=False,
                    save_matrix=False,
                    file_name_prefix=f"perm_ref_matrix_{group}_{level2}",
                    save_matrix_figure=False,
                    template_matrix=template_matrix
                )
                aligned_perm_ref2 = align_matrices(perm_ref2, template_matrix)
                perm_ref_graph2 = transition_matrix_to_graph(aligned_perm_ref2, template_matrix.columns)
                
                # Get permuted trial matrices
                perm_trials1 = [all_trials[i] for i in perm_indices1]
                perm_trials2 = [all_trials[i] for i in perm_indices2]
                
                # Calculate distances using current metric
                perm_dist1 = calculate_distances(np.array(perm_trials1), ref_matrix1, metric)
                perm_dist2 = calculate_distances(np.array(perm_trials2), ref_matrix2, metric)
                
                if len(perm_dist1) == 0 or len(perm_dist2) == 0:
                    continue
                
                perm_median1 = np.median(perm_dist1)
                perm_median2 = np.median(perm_dist2)
                perm_diff = perm_median1 - perm_median2
                
                perm_diffs.append(perm_diff)
            
            if not perm_diffs:
                print(f"Warning: No valid permutations calculated for metric {metric}")
                continue
            
            # Calculate p-value using two-tailed test
            p_value = np.mean(np.abs(np.array(perm_diffs)) >= np.abs(observed_diff))
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((std_dist1**2 + std_dist2**2) / 2)
            cohens_d = observed_diff / pooled_std if pooled_std != 0 else 0
            
            # Store results for this metric
            metric_results[metric] = {
                "observed_difference": observed_diff,
                "p_value": p_value,
                "effect_size": cohens_d,
                "permutation_differences": perm_diffs,
                "network_distance": network_distance,
                "condition1_stats": {
                    "n_observations": n_obs1,
                    "n_trials": len(condition1_trials),
                    "mean_distance": median_dist1,
                    "std_distance": std_dist1
                },
                "condition2_stats": {
                    "n_observations": n_obs2,
                    "n_trials": len(condition2_trials),
                    "mean_distance": median_dist2,
                    "std_distance": std_dist2
                },
                "distances1": distances1 if metric != 'graph_edit' else None,
                "distances2": distances2 if metric != 'graph_edit' else None,
                "matrix1": ref_matrix1.values if hasattr(ref_matrix1, 'values') else ref_matrix1,
                "matrix2": ref_matrix2.values if hasattr(ref_matrix2, 'values') else ref_matrix2,
                "graph1": level_graphs[level1],
                "graph2": level_graphs[level2],
            }
        
        return metric_results
    
    else:
        # Original single metric code
        distances1 = calculate_distances(all_trials1, ref_matrix1, metric_name)
        distances2 = calculate_distances(all_trials2, ref_matrix2, metric_name)
        
        if len(distances1) == 0 or len(distances2) == 0:
            print("Warning: No valid distances calculated")
            return 0.0, 1.0, []
        
        # Calculate mean distance for each condition
        median_dist1 = np.median(distances1)
        median_dist2 = np.median(distances2)
        observed_diff = median_dist1 - median_dist2
        
        # Calculate additional statistics
        std_dist1 = np.std(distances1)
        std_dist2 = np.std(distances2)
        min_dist1 = np.min(distances1)
        min_dist2 = np.min(distances2)
        max_dist1 = np.max(distances1)
        max_dist2 = np.max(distances2)
        
        # Combine all trials for permutation
        all_trials = np.concatenate([all_trials1, all_trials2])
        all_trial_data = condition1_trials + condition2_trials
        n1 = len(all_trials1)
        
        # Perform permutations
        perm_diffs = []
        for _ in range(n_permutations):
            # Create a random permutation of indices
            perm_indices = np.random.permutation(len(all_trials))
            
            # Split into two groups using indices
            perm_indices1 = perm_indices[:n1]
            perm_indices2 = perm_indices[n1:]
            
            # Get the permuted trial data
            perm_data1 = pd.concat([all_trial_data[i] for i in perm_indices1])
            perm_data2 = pd.concat([all_trial_data[i] for i in perm_indices2])
            
            # Calculate new reference matrices and graphs for permuted groups
            _, perm_ref1 = get_transition_matrix(
                perm_data1,
                label_col,
                output_dir,
                save_model=False,
                save_matrix=False,
                file_name_prefix=f"perm_ref_matrix_{group}_{level1}",
                save_matrix_figure=False,
                template_matrix=template_matrix
            )
            aligned_perm_ref1 = align_matrices(perm_ref1, template_matrix)
            perm_ref_graph1 = transition_matrix_to_graph(aligned_perm_ref1, template_matrix.columns)
            
            _, perm_ref2 = get_transition_matrix(
                perm_data2,
                label_col,
                output_dir,
                save_model=False,
                save_matrix=False,
                file_name_prefix=f"perm_ref_matrix_{group}_{level2}",
                save_matrix_figure=False,
                template_matrix=template_matrix
            )
            aligned_perm_ref2 = align_matrices(perm_ref2, template_matrix)
            perm_ref_graph2 = transition_matrix_to_graph(aligned_perm_ref2, template_matrix.columns)
            
            # Get permuted trial matrices
            perm_trials1 = [all_trials[i] for i in perm_indices1]
            perm_trials2 = [all_trials[i] for i in perm_indices2]
            
            # Calculate distances using current metric
            perm_dist1 = calculate_distances(np.array(perm_trials1), ref_matrix1, metric_name)
            perm_dist2 = calculate_distances(np.array(perm_trials2), ref_matrix2, metric_name)
            
            if len(perm_dist1) == 0 or len(perm_dist2) == 0:
                continue
            
            # Calculate mean distances
            perm_median1 = np.median(perm_dist1)
            perm_median2 = np.median(perm_dist2)
            perm_diff = perm_median1 - perm_median2
            
            perm_diffs.append(perm_diff)
        
        if not perm_diffs:
            print("Warning: No valid permutations calculated")
            return 0.0, 1.0, []
        
        # Calculate p-value using two-tailed test
        p_value = np.mean(np.abs(np.array(perm_diffs)) >= np.abs(observed_diff))
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((std_dist1**2 + std_dist2**2) / 2)
        cohens_d = observed_diff / pooled_std if pooled_std != 0 else 0
        
        # Calculate additional statistics
        perm_median = np.median(perm_diffs)
        perm_std = np.std(perm_diffs)
        perm_ci = np.percentile(perm_diffs, [2.5, 97.5])  # 95% confidence interval
        
        # Create visualization of permutation distribution
        plt.figure(figsize=(12, 8))
        
        # Plot histogram of permuted differences
        plt.hist(perm_diffs, bins=50, alpha=0.7, density=True, 
                 label=f'Permutation Distribution (N={len(perm_diffs)})')
        
        # Add vertical lines for observed difference and confidence interval
        plt.axvline(x=observed_diff, color='red', linestyle='--', 
                    label=f'Observed Difference: {observed_diff:.4f}')
        plt.axvline(x=-observed_diff, color='red', linestyle='--')
        plt.axvline(x=perm_ci[0], color='gray', linestyle=':', 
                    label='95% CI')
        plt.axvline(x=perm_ci[1], color='gray', linestyle=':')
        
        # Add statistical annotations
        stats_text = (f'Observed Difference: {observed_diff:.4f}\n'
                     f'p-value: {p_value:.4f}\n'
                     f"Cohen's d: {cohens_d:.4f}\n"
                     f'Permutation Median: {perm_median:.4f}\n'
                     f'Permutation SD: {perm_std:.4f}\n'
                     f'95% CI: [{perm_ci[0]:.4f}, {perm_ci[1]:.4f}]')
        
        plt.annotate(stats_text, 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.xlabel('Difference in Median Distances')
        plt.ylabel('Density')
        plt.title('Permutation Test Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the permutation distribution plot
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{group}_{level1}_vs_{level2}_permutation_distribution.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # Create QQ plot to check normality of permutation distribution
        plt.figure(figsize=(8, 8))
        stats.probplot(perm_diffs, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Permutation Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, f'{group}_{level1}_vs_{level2}_permutation_qqplot.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # Create box plot comparing conditions
        plt.figure(figsize=(8, 6))
        plt.boxplot([distances1, distances2], 
                    labels=[f'{level1} (N={len(distances1)})', 
                           f'{level2} (N={len(distances2)})'])
        plt.title('Distance Distribution by Condition')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, f'{group}_{level1}_vs_{level2}_distance_boxplot.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # Calculate additional statistics
        transition_stats = analyze_transition_statistics(
            ref_matrix1_without,
            ref_matrix2_without,
            condition1_data[label_col].values,
            condition2_data[label_col].values,
            output_dir,
            f"{group}_{level1}_vs_{level2}"
        )
        
        # Return results for single metric
        return {
            "observed_difference": observed_diff,
            "p_value": p_value,
            "permutation_differences": perm_diffs,
            "permutation_stats": {
                "median": perm_median,
                "std": perm_std,
                "ci_lower": perm_ci[0],
                "ci_upper": perm_ci[1]
            },
            "effect_size": cohens_d,
            "condition1_stats": {
                "n_observations": n_obs1,
                "n_trials": len(condition1_trials),
                "mean_distance": median_dist1,
                "std_distance": std_dist1,
                "min_distance": min_dist1,
                "max_distance": max_dist1,
                "entropy": transition_stats['entropy1'],
                "mutual_information": transition_stats['mi1'],
                "regularity": transition_stats['regularity1'],
                "complexity": transition_stats['complexity1']
            },
            "condition2_stats": {
                "n_observations": n_obs2,
                "n_trials": len(condition2_trials),
                "mean_distance": median_dist2,
                "std_distance": std_dist2,
                "min_distance": min_dist2,
                "max_distance": max_dist2,
                "entropy": transition_stats['entropy2'],
                "mutual_information": transition_stats['mi2'],
                "regularity": transition_stats['regularity2'],
                "complexity": transition_stats['complexity2']
            },
            "statistical_tests": {
                "mannwhitneyu": {
                    "statistic": transition_stats['mannwhitneyu_stat'],
                    "p_value": transition_stats['mannwhitneyu_p']
                },
                "kolmogorov_smirnov": {
                    "statistic": transition_stats['ks_stat'],
                    "p_value": transition_stats['ks_p']
                }
            },
            "n_permutations": n_permutations,
            "seed": seed,
            "metric": metric_name
        }

def create_output_directories(output_dir):
    """Create all necessary output directories for plots and results."""
    directories = [
        os.path.join(output_dir, 'results'),
        os.path.join(output_dir, 'results', 'figures'),
        os.path.join(output_dir, 'results', 'figures', 'metrics'),
        os.path.join(output_dir, 'results', 'figures', 'permutations'),
        os.path.join(output_dir, 'results', 'figures', 'distributions'),
        os.path.join(output_dir, 'results', 'figures', 'transition_matrices'),
        os.path.join(output_dir, 'results', 'figures', 'networks'),
        os.path.join(output_dir, 'results', 'figures', 'temporal'),
        os.path.join(output_dir, 'results', 'reference_matrices'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

@tictoc
def main():
    print("\033[34mCode has started running\033[0m\n")
    
    parser = argparse.ArgumentParser(description='Process transition matrices with metadata')
    parser.add_argument('--metadata', required=True, help='Path to the metadata CSV file')
    parser.add_argument('--output-dir', required=False, default=None, help='Path to the output directory')
    parser.add_argument('--ppid', required=False, default="ppid", help='Name of the ppid column')
    parser.add_argument('--session', required=False, default="session", help='Name of the session column')
    parser.add_argument('--trial', required=False, default="trial", help='Name of the trial column')
    parser.add_argument('--label', required=False, default="predicted_label", help='Name of the label column')
    parser.add_argument('--n-jobs', type=int, default=None, help='Number of parallel jobs to run. Defaults to number of CPU cores - 1')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--grid-search', action='store_true', help='Perform grid search over distance metrics')
    parser.add_argument('--metric', type=str, default='euclidean', 
                       choices=['euclidean', 'manhattan', 'cosine', 'correlation', 'kl_divergence'],
                       help='Distance metric to use when not doing grid search')
    parser.add_argument('--test-mode', action='store_true', help='Run only a couple of pairwise comparisons per group for testing')

    args = parser.parse_args()
    
    if args.n_jobs is None:
        # Default to using all cores minus 2, ensuring at least 1 core is used
        args.n_jobs = max(1, cpu_count() - 2)
    elif args.n_jobs == -1:
        # Special value -1 means use all cores
        args.n_jobs = cpu_count()
    elif args.n_jobs == -2:
        # Special value -2 means use all cores minus 1
        args.n_jobs = max(1, cpu_count() - 1)
    elif args.n_jobs <= 0:
        # Invalid, default to 1 core
        print(f"Warning: Invalid n_jobs value ({args.n_jobs}). Defaulting to 1 core.")
        args.n_jobs = 1
    # If args.n_jobs is a positive number, it will be used directly.

    print(f"Number of parallel jobs set to: {args.n_jobs}")
    
    # Set global random seed
    global_seed = set_random_seed(args.seed)
    print(f"Using global seed: {global_seed}")
    
    metadata_file = args.metadata
    ppid_column = args.ppid
    session_column = args.session
    trial_column = args.trial
    label_column = args.label
    target_groups = ["instructed_action"]

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.metadata), 'random_permutation_test_output')
    else:
        output_dir = args.output_dir

    # Create all necessary output directories
    create_output_directories(output_dir)

    try:
        df = get_data(
            metadata_file, ppid_column, session_column, trial_column, target_columns=target_groups, label_column=label_column, save_processed_data=True
        )
    except KeyError as e:
        print(f"\033[91mError: {str(e)}\033[0m")
        print("Available columns in the input file:")
        # Read just the header to show available columns
        df_sample = pd.read_csv(metadata_file, nrows=0)
        print("\n".join(df_sample.columns))
        return

    # Get the list of participants
    dictionary_df = df.copy()

    # First, get all unique syllables from the entire dataset and convert to strings
    all_syllables = sorted([str(s) for s in dictionary_df[label_column].unique()])
    print(f"\nTotal unique syllables: {len(all_syllables)}")
    print(f"Sample syllables: {list(all_syllables)[:5]}...")
    
    # Create a template matrix with all possible syllables
    template_matrix = pd.DataFrame(0.0, index=all_syllables, columns=all_syllables, dtype=float)
    
    # Process groups in parallel
    with Pool(args.n_jobs) as pool:
        # Create a list of tuples containing all arguments for each process
        process_args = []
        for i, group in enumerate(target_groups):
            process_seed = global_seed + i
            process_args.append((
                group,
                dictionary_df,
                ppid_column,
                session_column,
                trial_column,
                label_column,
                template_matrix,
                output_dir,
                process_seed,
                args.metric,
                args.grid_search,
                args.test_mode  # pass test_mode
            ))
        
        # Map the process_group function to the arguments
        all_results = dict(zip(target_groups, pool.starmap(process_group, process_args)))
    
    # Save results
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create summary DataFrame with one row per comparison, all metrics as columns
    summary_rows = []
    for group, group_results in all_results.items():
        for level, level_results in group_results.items():
            row = {'group': group, 'level': level}
            for metric, metric_results in level_results.items():
                row[f'{metric}_observed_difference'] = metric_results.get('observed_difference')
                row[f'{metric}_p_value'] = metric_results.get('p_value')
                row[f'{metric}_effect_size'] = metric_results.get('effect_size')
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    # Reorder columns: group, level, then all metric columns sorted
    metric_cols = sorted([col for col in summary_df.columns if col not in ['group', 'level']])
    summary_df = summary_df[['group', 'level'] + metric_cols]
    summary_df.to_csv(os.path.join(results_dir, "comparison_summary.csv"), index=False)
    
    print("\n\033[34mThe code has finished running\033[0m\n")

    # Generate summary plots
    plot_metric_summary(summary_df, results_dir)
    plot_all_permutation_histograms(summary_df, all_results, results_dir)
    plot_all_distance_distributions(all_results, results_dir)
    plot_all_transition_matrices(all_results, results_dir)
    plot_all_network_graphs(all_results, results_dir)
    plot_all_temporal_patterns(all_results, results_dir)

def plot_metric_summary(summary_df, output_dir):
    fig_dir = os.path.join(output_dir, 'figures', 'metrics')
    os.makedirs(fig_dir, exist_ok=True)
    metrics = sorted(set('_'.join(col.split('_')[:-2]) for col in summary_df.columns if '_observed_difference' in col))
    for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc='Metric summary plots'):
        level = row['level']
        group = row['group']
        pvals = [row.get(f'{m}_p_value', np.nan) for m in metrics]
        effects = [row.get(f'{m}_effect_size', np.nan) for m in metrics]
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        axs[0].bar(metrics, pvals)
        axs[0].axhline(0.05, color='r', linestyle='--')
        axs[0].set_ylabel('p-value')
        axs[0].set_title(f'P-values by Distance Metric\n{group} {level}')
        axs[1].bar(metrics, effects)
        axs[1].set_ylabel("Cohen's d")
        axs[1].set_title(f'Effect Sizes by Distance Metric\n{group} {level}')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'{group}_{level}_metric_summary.png'), dpi=300)
        plt.close()

def plot_all_permutation_histograms(summary_df, results, output_dir):
    """Plot permutation histograms for all comparisons."""
    fig_dir = os.path.join(output_dir, 'figures', 'permutations')
    os.makedirs(fig_dir, exist_ok=True)
    
    total = sum(len(metric_dict) for group_results in results.values() for metric_dict in group_results.values())
    pbar = tqdm(total=total, desc='Permutation histograms')
    
    for group, group_results in results.items():
        for level, metric_dict in group_results.items():
            for metric, metric_results in metric_dict.items():
                if 'permutation_differences' in metric_results:
                    perm_diffs = metric_results['permutation_differences']
                    observed_diff = metric_results.get('observed_difference', None)
                    
                    # Create figure
                    plt.figure(figsize=(10, 6))
                    plt.hist(perm_diffs, bins=30, alpha=0.7, density=True)
                    if observed_diff is not None:
                        plt.axvline(x=observed_diff, color='red', linestyle='--', 
                                    label=f'Observed Difference: {observed_diff:.4f}')
                        plt.axvline(x=-observed_diff, color='red', linestyle='--', 
                                    label=f'Observed Difference: {-observed_diff:.4f}')
                    plt.title(f'Permutation Distribution: {group} {level} {metric}')
                    plt.xlabel('Difference')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.tight_layout()
                    
                    # Create shorter filename
                    # Extract comparison from level (e.g., "walk_forwards_vs_run_forwards" -> "walk_vs_run")
                    comparison = level.split('_vs_')
                    if len(comparison) == 2:
                        cond1 = comparison[0].split('_')[-1]  # Get last word
                        cond2 = comparison[1].split('_')[-1]  # Get last word
                        filename = f'perm_{cond1}_vs_{cond2}_{metric}.png'
                    else:
                        filename = f'perm_{level}_{metric}.png'
                    
                    filepath = os.path.join(fig_dir, filename)
                    
                    # Ensure directory exists again before saving
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    # Save figure
                    plt.savefig(filepath, dpi=300)
                    plt.close()
                pbar.update(1)
    pbar.close()

def plot_all_distance_distributions(results, output_dir):
    """Plot distance distributions for all comparisons."""
    fig_dir = os.path.join(output_dir, 'figures', 'distributions')

    os.makedirs(fig_dir, exist_ok=True)
    
    total = sum(len(metric_dict) for group_results in results.values() for metric_dict in group_results.values())
    pbar = tqdm(total=total, desc='Distance distributions')
    
    for group, group_results in results.items():
        for level, metric_dict in group_results.items():
            for metric, metric_results in metric_dict.items():
                d1 = metric_results.get('distances1', None)
                d2 = metric_results.get('distances2', None)
                if d1 is not None and d2 is not None:
                    plt.figure(figsize=(10, 6))
                    plt.hist(d1, bins=30, alpha=0.5, density=True, label='Condition 1')
                    plt.hist(d2, bins=30, alpha=0.5, density=True, label='Condition 2')
                    sns.kdeplot(d1, color='g', label='Cond1 KDE')
                    sns.kdeplot(d2, color='r', label='Cond2 KDE')
                    plt.title(f'Distance Distributions: {group} {level} {metric}')
                    plt.xlabel('Distance')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.tight_layout()
                    
                    # Create shorter filename
                    comparison = level.split('_vs_')
                    if len(comparison) == 2:
                        cond1 = comparison[0].split('_')[-1]
                        cond2 = comparison[1].split('_')[-1]
                        filename = f'dist_{cond1}_vs_{cond2}_{metric}.png'
                    else:
                        filename = f'dist_{level}_{metric}.png'
                    
                    filepath = os.path.join(fig_dir, filename)
                    
                    # Ensure directory exists again before saving
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    # Save figure
                    plt.savefig(filepath, dpi=300)
                    plt.close()
                pbar.update(1)
    pbar.close()

def plot_all_transition_matrices(results, output_dir):
    """Plot transition matrices for all comparisons."""
    fig_dir = os.path.join(output_dir, 'figures', 'transition_matrices')
    os.makedirs(fig_dir, exist_ok=True)
    
    total = sum(len(metric_dict) for group_results in results.values() for metric_dict in group_results.values())
    pbar = tqdm(total=total, desc='Transition matrix heatmaps')
    
    for group, group_results in results.items():
        for level, metric_dict in group_results.items():
            for metric, metric_results in metric_dict.items():
                m1 = metric_results.get('matrix1', None)
                m2 = metric_results.get('matrix2', None)
                if m1 is not None and m2 is not None:
                    # Create shorter filenames
                    comparison = level.split('_vs_')
                    if len(comparison) == 2:
                        cond1 = comparison[0].split('_')[-1]
                        cond2 = comparison[1].split('_')[-1]
                        base_name = f'matrix_{cond1}_vs_{cond2}_{metric}'
                    else:
                        base_name = f'matrix_{level}_{metric}'
                    
                    # Plot matrix 1
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(m1, cmap='plasma', square=True)
                    plt.title(f'Transition Matrix 1: {group} {level} {metric}')
                    plt.tight_layout()
                    
                    filename1 = f'{base_name}_1.png'
                    filepath1 = os.path.join(fig_dir, filename1)
                    os.makedirs(os.path.dirname(filepath1), exist_ok=True)
                    plt.savefig(filepath1, dpi=300)
                    plt.close()
                    
                    # Plot matrix 2
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(m2, cmap='plasma', square=True)
                    plt.title(f'Transition Matrix 2: {group} {level} {metric}')
                    plt.tight_layout()
                    
                    filename2 = f'{base_name}_2.png'
                    filepath2 = os.path.join(fig_dir, filename2)
                    os.makedirs(os.path.dirname(filepath2), exist_ok=True)
                    plt.savefig(filepath2, dpi=300)
                    plt.close()
                    
                    # Plot difference matrix
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(m1 - m2, cmap='RdBu_r', center=0, square=True)
                    plt.title(f'Transition Matrix Difference: {group} {level} {metric}')
                    plt.tight_layout()
                    
                    filename_diff = f'{base_name}_diff.png'
                    filepath_diff = os.path.join(fig_dir, filename_diff)
                    os.makedirs(os.path.dirname(filepath_diff), exist_ok=True)
                    plt.savefig(filepath_diff, dpi=300)
                    plt.close()
                pbar.update(1)
    pbar.close()

def plot_all_network_graphs(results, output_dir):
    """Plot network graphs for all comparisons, only including nodes with edges."""
    fig_dir = os.path.join(output_dir, 'figures', 'networks')
    os.makedirs(fig_dir, exist_ok=True)
    
    total = sum(len(metric_dict) for group_results in results.values() for metric_dict in group_results.values())
    pbar = tqdm(total=total, desc='Network graphs')
    
    for group, group_results in results.items():
        for level, metric_dict in group_results.items():
            for metric, metric_results in metric_dict.items():
                g1 = metric_results.get('graph1', None)
                g2 = metric_results.get('graph2', None)
                
                # Create shorter filenames
                comparison = level.split('_vs_')
                if len(comparison) == 2:
                    cond1 = comparison[0].split('_')[-1]
                    cond2 = comparison[1].split('_')[-1]
                    base_name = f'net_{cond1}_vs_{cond2}_{metric}'
                else:
                    base_name = f'net_{level}_{metric}'
                
                if g1 is not None:
                    # Create a subgraph of g1 with only nodes that have edges
                    g1_subgraph = g1.subgraph([n for n in g1.nodes() if g1.degree(n) > 0])
                    
                    plt.figure(figsize=(10, 8))
                    nx.draw_networkx(g1_subgraph, with_labels=True, node_color='lightblue', edge_color='gray')
                    plt.title(f'Network Graph 1: {group} {level} {metric}')
                    plt.tight_layout()
                    
                    filename1 = f'{base_name}_1.png'
                    filepath1 = os.path.join(fig_dir, filename1)
                    os.makedirs(os.path.dirname(filepath1), exist_ok=True)
                    plt.savefig(filepath1, dpi=300)
                    plt.close()
                
                if g2 is not None:
                    # Create a subgraph of g2 with only nodes that have edges
                    g2_subgraph = g2.subgraph([n for n in g2.nodes() if g2.degree(n) > 0])
                    
                    plt.figure(figsize=(10, 8))
                    nx.draw_networkx(g2_subgraph, with_labels=True, node_color='lightgreen', edge_color='gray')
                    plt.title(f'Network Graph 2: {group} {level} {metric}')
                    plt.tight_layout()
                    
                    filename2 = f'{base_name}_2.png'
                    filepath2 = os.path.join(fig_dir, filename2)
                    os.makedirs(os.path.dirname(filepath2), exist_ok=True)
                    plt.savefig(filepath2, dpi=300)
                    plt.close()
                pbar.update(1)
    pbar.close()

def plot_all_temporal_patterns(results, output_dir):
    """Plot temporal patterns for all comparisons."""
    fig_dir = os.path.join(output_dir, 'figures', 'temporal')
    os.makedirs(fig_dir, exist_ok=True)
    
    total = sum(len(metric_dict) for group_results in results.values() for metric_dict in group_results.values())
    pbar = tqdm(total=total, desc='Temporal pattern heatmaps')
    
    for group, group_results in results.items():
        for level, metric_dict in group_results.items():
            for metric, metric_results in metric_dict.items():
                seq1 = metric_results.get('sequence1', None)
                seq2 = metric_results.get('sequence2', None)
                
                # Create shorter filenames
                comparison = level.split('_vs_')
                if len(comparison) == 2:
                    cond1 = comparison[0].split('_')[-1]
                    cond2 = comparison[1].split('_')[-1]
                    base_name = f'temp_{cond1}_vs_{cond2}_{metric}'
                else:
                    base_name = f'temp_{level}_{metric}'
                
                if seq1 is not None:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(seq1, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Transition Count')
                    plt.title(f'Temporal Pattern 1: {group} {level} {metric}')
                    plt.tight_layout()
                    
                    filename1 = f'{base_name}_1.png'
                    filepath1 = os.path.join(fig_dir, filename1)
                    os.makedirs(os.path.dirname(filepath1), exist_ok=True)
                    plt.savefig(filepath1, dpi=300)
                    plt.close()
                
                if seq2 is not None:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(seq2, aspect='auto', cmap='viridis')
                    plt.title(f'Temporal Pattern 2: {group} {level} {metric}')
                    plt.tight_layout()
                    
                    filename2 = f'{base_name}_2.png'
                    filepath2 = os.path.join(fig_dir, filename2)
                    os.makedirs(os.path.dirname(filepath2), exist_ok=True)
                    plt.savefig(filepath2, dpi=300)
                    plt.close()
                pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    os.system("cls||clear")
    print("\n\033[35mcode by uSerratos\033[0m\n\n")
    main()
