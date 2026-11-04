"""Module for analyzing syllable transition matrices as graph representations."""

# ==
# author: uSerratos (based on transition matrices module)
# date: Nov 2024
# ==

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import hashlib
import re

# Compatibility layer for pomegranate 1.1.2
try:
    from pomegranate import MarkovChain
    POMEGRANATE_OLD = True
except ImportError:
    # Use pomegranate 1.1.2 API
    from pomegranate.hmm import DenseHMM
    from pomegranate.distributions import Categorical
    POMEGRANATE_OLD = False
    
    class MarkovChain:
        """Compatibility class for pomegranate 1.1.2"""
        def __init__(self, transition_matrix, state_map):
            self.transition_matrix = transition_matrix
            self.state_map = state_map
            
        @classmethod
        def from_samples(cls, sequences):
            # Convert sequences to the format expected by DenseHMM
            # sequences is a list of lists, we need to flatten and create states
            flat_sequences = []
            for seq in sequences:
                flat_sequences.extend(seq)
            
            # Get unique states
            unique_states = sorted(set(flat_sequences))
            n_states = len(unique_states)
            
            # Create state mapping
            state_map = {state: i for i, state in enumerate(unique_states)}
            
            # Convert sequences to numeric format
            numeric_sequences = []
            for seq in sequences:
                numeric_seq = [state_map[state] for state in seq]
                numeric_sequences.append(numeric_seq)
            
            # Create transition matrix from sequences
            transition_matrix = np.zeros((n_states, n_states))
            for seq in numeric_sequences:
                for i in range(len(seq) - 1):
                    current_state = seq[i]
                    next_state = seq[i + 1]
                    transition_matrix[current_state, next_state] += 1
            
            # Normalize rows to get probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_matrix = transition_matrix / row_sums
            
            return cls(transition_matrix, state_map)
        
        @property
        def distributions(self):
            # Return a mock distributions object for compatibility
            class MockDistributions:
                def __init__(self, transition_matrix, state_map):
                    self.transition_matrix = transition_matrix
                    self.state_map = state_map
                    
                @property
                def parameters(self):
                    # Return transition matrix in the format expected by restructure_matrix
                    return [self.transition_matrix]
            
            return [None, MockDistributions(self.transition_matrix, self.state_map)]

import os
from joblib import dump, Parallel, delayed
import matplotlib.pyplot as plt
import time
import getpass
import sys
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import re
import multiprocessing
try:
    from networkx.algorithms import community
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

# Only import utils if it hasn't been imported already
if 'utils' not in sys.modules:
    if os.environ.get('RUNNING_AS_SUBPROCESS') != 'true':
        os.system("cls||clear")
    from utils import *

warnings.filterwarnings("ignore")

# SYSTEM COLORS
class bold_color:
    """
    Color Options
    ---
    PURPLE, CYAN, DARKBLUE, BLUE, GREEN, YELLOW, RED
    """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_color(text: str, color: bold_color) -> str:
    """
    prints lines of specified colour

    Color Options
    ---
    PURPLE, CYAN, DARKBLUE, BLUE, GREEN, YELLOW, RED
    """
    print(color + color + text + bold_color.END)
    return ""


def tictoc(func):
    def inner(*args, **kwargs):
        T1: time = time.time()
        print(f"\n\033[92mFunction: {func.__name__} started at {time.strftime('%H:%M:%S', time.gmtime(T1))}\033[0m\n")
        func_out = func(*args, **kwargs)
        T2: time = time.time()
        T: time = time.strftime("%H:%M:%S", time.gmtime(T2 - T1))
        print(f"\n\033[92mFunction: {func.__name__} ended, total runTime {T}\033[0m\n")
        return func_out
    return inner


def create_color_map(le: LabelEncoder, colour_map_arg=None, label_description_dictionary=None) -> dict:
    # (Same as in counts/durations)
    if isinstance(colour_map_arg, str) and colour_map_arg.endswith('.pkl'):
        try:
            import pickle
            with open(colour_map_arg, 'rb') as f:
                colour_map_arg = pickle.load(f)
        
        except Exception as e:
            print(f"Warning: Could not load color map from {colour_map_arg}: {e}")
            colour_map_arg = None
    if isinstance(colour_map_arg, dict):
        if label_description_dictionary:
            colour_map_arg = {str(label_description_dictionary.get(label)): colour for label, colour in colour_map_arg.items()}
        return colour_map_arg
    if colour_map_arg:
        try:
            color_map = plt.colormaps.get_cmap(colour_map_arg)
        except ValueError as e:
            print(f"Warning: Invalid colormap name '{colour_map_arg}': {e}")
            color_map = create_pastel_rainbow_cmap()
    else:
        color_map = create_pastel_rainbow_cmap()
    if label_description_dictionary:
        label_description_dictonary_unique_values = list(set(label_description_dictionary.values()))
        normalized_labels = [i / (len(label_description_dictonary_unique_values) - 1) for i in range(len(label_description_dictonary_unique_values))]
    else:
        normalized_labels = le.transform(le.classes_) / (len(le.classes_) - 1)
    cmap = [color_map(norm_label) for norm_label in normalized_labels]
    if label_description_dictionary:
        color_map_dict = {str(label): cmap[i] for i, label in enumerate(label_description_dictonary_unique_values)}
    else:
        color_map_dict = {str(label): cmap[i] for i, label in enumerate(le.classes_)}
    color_map_dict['other'] = '#D3D3D3'
    color_map_dict[-1] = '#D3D3D3'
    color_map_dict["-1"] = '#D3D3D3'
    return color_map_dict


def print_welcome_message():
    # Get the username
    user_name = getpass.getuser()

    # Get the full path of the initial script
    full_path = os.path.realpath(sys.argv[0])

    # Get the file name only, without extension
    file_name_without_extension = os.path.splitext(
        os.path.basename(full_path)
    )[0]

    # Replace underscores with spaces and capitalize every word
    formatted_file_name = file_name_without_extension.replace(
        "_", " "
    ).title()

    # Print the welcome message
    print_color(
        f"{formatted_file_name} Code [by uSerratos] Started \n",
        bold_color.GREEN
    )


def restructure_matrix(model):
    """Restructure the Markov chain model into a readable transition matrix."""
    transition_matrix = model.distributions[1].parameters[0]
    # The transition_matrix is already a 2D array, so we can directly create a DataFrame
    # We need to get the state labels from the model
    if hasattr(model, 'state_map'):
        # For the compatibility class
        state_labels = list(model.state_map.keys())
    else:
        # For the original pomegranate class, we need to get the state labels differently
        # This is a fallback - we'll use numeric indices
        state_labels = list(range(transition_matrix.shape[0]))
    
    # Create DataFrame with proper row and column labels
    matrix = pd.DataFrame(transition_matrix, 
                         index=state_labels, 
                         columns=state_labels)
    return matrix


def encode_and_get_sequences(data, label_column, le=None):
    """Encode labels and create sequences for Markov chain analysis."""
    if le is None:
        le = LabelEncoder()
        # Create a copy of the data to avoid SettingWithCopyWarning
        data = data.copy()
        data.loc[:, label_column] = le.fit_transform(data[label_column])
    else:
        # Create a copy of the data to avoid SettingWithCopyWarning
        data = data.copy()
        data.loc[:, label_column] = le.transform(data[label_column])
    sequences = [data[label_column].tolist()]
    return sequences, le


def draw_edges_with_node_margin(ax, G, pos, node_sizes, edge_widths, edge_color, alpha, arrowsize, arrowstyle, connectionstyle):
    # node_sizes is a dict {node: size_in_points^2}
    # pos is a dict {node: (x, y)}
    for (u, v), width in zip(G.edges(), edge_widths):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        size1 = node_sizes[u]
        size2 = node_sizes[v]
        # Convert node size (points^2) to radius in data coordinates
        radius1 = (size1 ** 0.5) / 2 / 72  # 72 points per inch
        radius2 = (size2 ** 0.5) / 2 / 72
        # Calculate direction
        dx, dy = x2 - x1, y2 - y1
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist == 0:
            continue
        # Offset start and end by node radii
        x1a = x1 + radius1 * dx / dist
        y1a = y1 + radius1 * dy / dist
        x2a = x2 - radius2 * dx / dist
        y2a = y2 - radius2 * dy / dist
        arrow = FancyArrowPatch((x1a, y1a), (x2a, y2a),
                                arrowstyle=arrowstyle,
                                color=edge_color,
                                linewidth=width,
                                alpha=alpha,
                                mutation_scale=arrowsize,
                                connectionstyle=connectionstyle)
        ax.add_patch(arrow)


def create_transition_graph(matrix, threshold=0.0, title="Transition Graph", output_dir=None, show=False):
    """
    Create a graph representation of the transition matrix.
    
    Parameters:
    -----------
    matrix : pd.DataFrame
        Transition matrix with probabilities
    threshold : float
        Minimum probability threshold for including edges (0.0 to 1.0)
    title : str
        Title for the graph
    output_dir : str
        Directory to save the graph
    show : bool
        Whether to display the graph
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in matrix.index:
        G.add_node(node)
    
    # Add edges based on threshold
    edge_data = []
    for i, source in enumerate(matrix.index):
        for j, target in enumerate(matrix.columns):
            prob = matrix.iloc[i, j]
            if prob > threshold and not np.isnan(prob):
                G.add_edge(source, target, weight=prob)
                edge_data.append((source, target, prob))
    
    # Check if graph has any edges
    if not edge_data:
        print(f"Warning: No edges found for {title} with threshold {threshold:.3f}")
        return G
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue', 
                          node_size=2000, 
                          alpha=0.8)
    node_sizes_dict = {n: 2000 for n in G.nodes()}
    
    # Draw edges with varying thickness based on probability
    if edge_data:
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [2 + 8 * (w / max_weight) for w in weights]
        try:
            nx.draw_networkx_edges(G, pos, 
                                  width=edge_widths,
                                  edge_color='gray',
                                  alpha=0.6,
                                  arrows=True,
                                  arrowsize=20,
                                  arrowstyle='->',
                                  connectionstyle='arc3,rad=0.1',
                                  min_source_margin=20,
                                  min_target_margin=20)
        except TypeError:
            ax = plt.gca()
            draw_edges_with_node_margin(ax, G, pos, node_sizes_dict, edge_widths, 'gray', 0.6, 20, '->', 'arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add edge labels for significant transitions
    edge_labels = {}
    for u, v, w in edge_data:
        if w > threshold + 0.05:  # Only show labels for transitions > threshold + 0.05
            edge_labels[(u, v)] = f'{w:.2f}'
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    # Add title and statistics
    plt.title(f"{title}\nThreshold: {threshold:.3f} | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend for edge weights
    if edge_data:
        legend_elements = [
            mpatches.Patch(color='gray', alpha=0.6, label=f'Edge weight: {min(weights):.3f} - {max(weights):.3f}')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        filename = f"{title.replace(' ', '_')}_threshold_{threshold:.3f}.png"
        fig_name = get_safe_filename(output_dir, filename)
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
    
    return G


def safe_filename(s):
    import re
    return re.sub(r'[^a-zA-Z0-9_\-]', '', s)


def _separate_cluster_layout(G, communities, base_k=1.0, cluster_spacing=3.5):
    import math
    pos = {}
    n_clusters = len(communities)
    angle_step = 2 * math.pi / n_clusters
    cluster_centers = [
        (cluster_spacing * math.cos(i * angle_step), cluster_spacing * math.sin(i * angle_step))
        for i in range(n_clusters)
    ]
    for idx, comm in enumerate(communities):
        subG = G.subgraph(comm)
        sub_pos = nx.spring_layout(subG, k=base_k, iterations=100, seed=42)
        centroid = np.mean(list(sub_pos.values()), axis=0)
        for node in sub_pos:
            sub_pos[node] = sub_pos[node] + (np.array(cluster_centers[idx]) - centroid) * 0.7
        pos.update(sub_pos)
    return pos


def create_heatmap_graph(matrix, threshold=0.0, title="Transition Heatmap", output_dir=None, show=False, file_tag=None, color_map=None):
    """
    Create a heatmap-style graph representation with node sizes based on in-degree and node color based on community.
    """
    # Create directed graph
    G = nx.DiGraph()
    for node in matrix.index:
        G.add_node(node)
    for i, source in enumerate(matrix.index):
        for j, target in enumerate(matrix.columns):
            prob = matrix.iloc[i, j]
            if prob > threshold and not np.isnan(prob):
                G.add_edge(source, target, weight=prob)
    if not G.edges():
        print_color(f"Warning: No edges found for {title} with threshold {threshold:.3f}", bold_color.YELLOW)
        return G
    plt.figure(figsize=(18, 14))  # Larger figure for better readability
    # Community detection and layout
    if COMMUNITY_AVAILABLE:
        try:
            communities = list(community.louvain_communities(G.to_undirected(), weight='weight'))
            community_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i
            pos = _separate_cluster_layout(G, communities, base_k=8.0, cluster_spacing=3.0)
            n_communities = len(communities)
            community_colors = plt.cm.Set3(np.linspace(0, 1, n_communities))
        except Exception as e:
            print_color(f"Warning: Community detection failed, using spring layout: {e}", bold_color.YELLOW)
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            community_map = None
            communities = None
    else:
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        community_map = None
        communities = None
    
    # Use syllable colors for nodes if color_map is provided, otherwise use default
    if color_map:
        node_colors = [color_map.get(str(node), '#D3D3D3') for node in G.nodes()]
    else:
        node_colors = 'lightblue'
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    composite = {node: in_degrees.get(node, 0) + out_degrees.get(node, 0) for node in G.nodes()}
    max_composite = max(composite.values()) if composite else 1
    node_sizes = [1000 + 3000 * (composite[node] / max_composite) for node in G.nodes()]
    node_sizes_dict = {n: s for n, s in zip(G.nodes(), node_sizes)}
    # Draw community background shading if communities are available
    if COMMUNITY_AVAILABLE and community_map and communities:
        ax = plt.gca()
        for i, comm in enumerate(communities):
            if len(comm) > 1:  # Only draw background for communities with multiple nodes
                # Get positions of nodes in this community
                comm_positions = [pos[node] for node in comm if node in pos]
                if len(comm_positions) > 1:
                    # Calculate bounding box for the community
                    x_coords = [p[0] for p in comm_positions]
                    y_coords = [p[1] for p in comm_positions]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Add padding around the community
                    padding = 0.1
                    x_padding = (x_max - x_min) * padding
                    y_padding = (y_max - y_min) * padding
                    
                    # Draw a light grey background circle/ellipse for the community
                    from matplotlib.patches import Ellipse
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width = (x_max - x_min) + 2 * x_padding
                    height = (y_max - y_min) + 2 * y_padding
                    
                    ellipse = Ellipse((center_x, center_y), width, height, 
                                    facecolor='lightgrey', alpha=0.6, zorder=0)
                    ax.add_patch(ellipse)
    
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_sizes,
                                  node_color=node_colors,
                                  alpha=0.9,
                                  edgecolors='black',
                                  linewidths=1.5)
    if G.edges():
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [1.5 + 6 * (w / max_weight) for w in weights]  # Thicker edges for better visibility
        try:
            nx.draw_networkx_edges(G, pos, 
                                  width=edge_widths,
                                  edge_color='darkgray',
                                  alpha=0.8,
                                  arrows=True,
                                  arrowsize=20,
                                  arrowstyle='->',
                                  connectionstyle='arc3,rad=0.2',
                                  min_source_margin=25,
                                  min_target_margin=25)
        except TypeError:
            ax = plt.gca()
            draw_edges_with_node_margin(ax, G, pos, node_sizes_dict, edge_widths, 'darkgray', 0.8, 20, '->', 'arc3,rad=0.2')
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    # Add legend for communities and node coloring
    if COMMUNITY_AVAILABLE and community_map and communities:
        from matplotlib.patches import Patch
        legend_elements = []
        
        # Add community background legend
        legend_elements.append(Patch(facecolor='lightgrey', alpha=0.6, edgecolor='black',
                                     label='Community groups (background shading)'))
        
        # Add node coloring legend
        if color_map:
            legend_elements.append(Patch(facecolor='blue', alpha=0.8, edgecolor='black',
                                         label='Node colors: syllable types'))
        else:
            legend_elements.append(Patch(facecolor='lightblue', alpha=0.8, edgecolor='black',
                                         label='Node colors: default'))
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
    else:
        # Fallback legend when no communities
        from matplotlib.patches import Patch
        legend_elements = []
        if color_map:
            legend_elements.append(Patch(facecolor='blue', alpha=0.8, edgecolor='black',
                                         label='Node colors: syllable types'))
        else:
            legend_elements.append(Patch(facecolor='lightblue', alpha=0.8, edgecolor='black',
                                         label='Node colors: default'))
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
    if file_tag:
        full_title = f"{title}\nThreshold: {threshold:.3f} | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}\n"
        full_title += f"Node size: composite degree | {file_tag}"
    else:
        full_title = f"{title}\nThreshold: {threshold:.3f} | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}\n"
        full_title += f"Node size: composite degree"
    plt.title(full_title, fontsize=16, fontweight='bold', pad=25)
    plt.axis('off')
    plt.tight_layout()
    if output_dir:
        safe_tag = safe_filename(file_tag) if file_tag else ''
        if file_tag:
            filename = f"{safe_tag}.png"
        else:
            filename = f"heatmap.png"
        fig_name = get_safe_filename(output_dir, filename)
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return G


def process_single_trial_graph(group_data, args, matrix2, output_dir, le, threshold):
    """Process a single trial to create its transition graph."""
    name = group_data[0]  # tuple of (ppid, session, trial)
    group = group_data[1]  # dataframe for this group
    
    name_current = f'transition_graph_P{name[0]:03}_S{name[1]:03}_T{name[2]:03}'

    # Create sequence without self-transitions using original labels
    original_labels = group[args.label_column].tolist()
    new_sequence = [original_labels[0]]
    for i in range(1, len(original_labels)):
        if original_labels[i] != original_labels[i - 1]:
            new_sequence.append(original_labels[i])

    current_data = pd.DataFrame(new_sequence, columns=[args.label_column])
    sequences3, _ = encode_and_get_sequences(current_data, args.label_column, le)
    model3 = MarkovChain.from_samples(sequences3)
    matrix3 = restructure_matrix(model3)

    # Get the labels in the same order as the matrix
    encoded_labels3 = matrix3.columns
    decoded_labels3 = le.inverse_transform(encoded_labels3)

    # Create a new matrix with the same size and structure as matrix2, but initialized with zeros
    new_matrix = pd.DataFrame(np.zeros(matrix2.shape), index=matrix2.index, columns=matrix2.columns)

    # Iterate over the columns of matrix3
    for col in matrix3.columns:
        # Check if the column exists in new_matrix
        if col in new_matrix.columns:
            # Assign the values from matrix3 to the corresponding rows in new_matrix
            for i, value in enumerate(matrix3[col]):
                new_matrix.at[decoded_labels3[i], decoded_labels3[col]] = value

    # Create graphs for this trial
    if not args.no_plots:
        # Check if there are any transitions above threshold before creating graphs
        has_transitions = False
        for i, source in enumerate(new_matrix.index):
            for j, target in enumerate(new_matrix.columns):
                prob = new_matrix.iloc[i, j]
                if prob > threshold and not np.isnan(prob):
                    has_transitions = True
                    break
            if has_transitions:
                break
        
        if has_transitions:
            create_transition_graph(new_matrix, threshold=threshold, 
                                   title=name_current.replace('_', ' '), 
                                   output_dir=output_dir)
            
            create_heatmap_graph(new_matrix, threshold=threshold, 
                                title=name_current.replace('_', ' '), 
                                output_dir=output_dir)
        else:
            print(f"Warning: No transitions above threshold {threshold:.3f} for {name_current}")
    
    # Save the matrix as CSV
    csv_filename = f'{name_current}.csv'
    csv_path = get_safe_filename(output_dir, csv_filename)
    new_matrix.to_csv(csv_path)


def single_permutation(label_sequences, matrix):
    permuted_counts = np.zeros(matrix.shape)
    for labels in label_sequences:
        shuffled = np.random.permutation(labels)
        for i in range(len(shuffled) - 1):
            src = shuffled[i]
            tgt = shuffled[i + 1]
            if src in matrix.index and tgt in matrix.columns:
                src_idx = matrix.index.get_loc(src)
                tgt_idx = matrix.columns.get_loc(tgt)
                permuted_counts[src_idx, tgt_idx] += 1
    row_sums = permuted_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        permuted_probs = np.divide(permuted_counts, row_sums, out=np.zeros_like(permuted_counts), where=row_sums!=0)
    return permuted_probs


def permutation_edge_threshold(matrix, label_sequences, n_perm=1000, quantile=0.95, random_state=42):
    """
    For each edge, compute a null distribution of transition probabilities by permuting the label sequences.
    Only keep edges whose observed probability is above the specified quantile of the null distribution.
    Returns a boolean mask matrix of the same shape as matrix.
    """
    np.random.seed(random_state)
    observed = matrix.values.copy()
    n_jobs = max(1, multiprocessing.cpu_count() - 3)
    null_distributions = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(single_permutation)(label_sequences, matrix) for _ in range(n_perm)
    )
    null_distributions = np.stack(null_distributions)
    # Compute quantile threshold for each edge
    thresholds = np.quantile(null_distributions, quantile, axis=0)
    mask = observed > thresholds
    return mask


def top_percent_edge_threshold(matrix, top_percent=80.0):
    """
    For each node, keep the smallest set of edges such that their cumulative sum accounts for at least top_percent% of the total in- or out-edge weight.
    Returns a boolean mask matrix of the same shape as matrix.
    """
    arr = matrix.values.copy()
    n = arr.shape[0]
    mask = np.zeros_like(arr, dtype=bool)
    # Outgoing (rows)
    for i in range(n):
        row = arr[i, :]
        total = row.sum()
        if total <= 0:
            continue
        idx_sorted = np.argsort(row)[::-1]
        cumsum = np.cumsum(row[idx_sorted])
        keep = cumsum < top_percent / 100.0 * total
        # Always keep at least one edge if any exist
        if np.any(row > 0):
            keep[np.argmax(keep == False)] = True
        mask[i, idx_sorted[keep]] = True
    # Incoming (columns)
    for j in range(n):
        col = arr[:, j]
        total = col.sum()
        if total <= 0:
            continue
        idx_sorted = np.argsort(col)[::-1]
        cumsum = np.cumsum(col[idx_sorted])
        keep = cumsum < top_percent / 100.0 * total
        if np.any(col > 0):
            keep[np.argmax(keep == False)] = True
        mask[idx_sorted[keep], j] = True
    return mask


def get_short_path(path, max_length=200):
    """Get a shorter version of the path if it's too long for Windows."""
    if len(path) <= max_length:
        return path
    
    # Try to use a shorter base directory
    # Extract the last few parts of the path
    parts = path.split(os.sep)
    if len(parts) > 3:
        # Keep only the last 3 parts
        short_parts = parts[-3:]
        short_path = os.sep.join(short_parts)
        if len(short_path) <= max_length:
            return short_path
    
    # If still too long, use a hash-based short name
    import hashlib
    path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
    return f"short_{path_hash}"


def get_safe_filename(output_dir, filename, max_length=260):
    """Get a filename that's safe for Windows path length limits."""
    full_path = os.path.join(output_dir, filename)
    
    if len(full_path) < max_length:
        return full_path
    
    # If too long, create a meaningful shortened filename
    name, ext = os.path.splitext(filename)
    short_name = create_compressed_filename(name, ext, max_length - len(output_dir) - 1)
    short_path = os.path.join(output_dir, short_name)
    
    print(f"WARNING: Filename too long ({len(full_path)} chars), using compressed name: {short_name}")
    print(f"New path length: {len(short_path)} characters")
    
    return short_path


def create_compressed_filename(name, ext, max_name_length):
    """Create a compressed but meaningful filename."""
    # Common abbreviations for transition matrix analysis
    abbreviations = {
        'transition_matrix': 'tm',
        'all_trials': 'all',
        'exclself': 'excl',
        'permthresh': 'perm',
        'threshold': 'th',
        'model_with_self_transitions': 'm1',
        'model_without_self_transitions': 'm2',
        'compare': 'cmp',
        'three_panel': '3p',
        'heatmap': 'hm',
        'threshold': 'th',
        'Individual': 'ind',
        'Comparison': 'cmp',
        'Data': 'd',
        'Plots': 'p',
        'Pickles': 'pk'
    }
    
    # Apply abbreviations
    compressed = name
    for full_word, abbrev in abbreviations.items():
        compressed = compressed.replace(full_word, abbrev)
    
    # If still too long, try more aggressive compression
    if len(compressed) > max_name_length:
        # Remove common separators and compress numbers
        compressed = compressed.replace('_', '')
        compressed = compressed.replace('-', '')
        
        # Compress number patterns like "n1081" to "n1081" or "q95" to "q95"
        import re
        # Keep important numbers but remove redundant ones
        compressed = re.sub(r'n\d+', lambda m: f"n{m.group(0)[1:]}", compressed)
        compressed = re.sub(r'q\d+', lambda m: f"q{m.group(0)[1:]}", compressed)
    
    # If still too long, use a meaningful hash
    if len(compressed) > max_name_length:
        # Extract key info: type, n_obs, method
        key_parts = []
        if 'tm' in compressed:
            key_parts.append('tm')
        if re.search(r'n\d+', compressed):
            n_match = re.search(r'n(\d+)', compressed)
            if n_match:
                key_parts.append(f"n{n_match.group(1)}")
        if 'perm' in compressed:
            key_parts.append('perm')
        elif 'th' in compressed:
            key_parts.append('th')
        elif 'top' in compressed:
            key_parts.append('top')
        
        # Create meaningful short name
        base_name = '_'.join(key_parts) if key_parts else 'data'
        # Add hash for uniqueness but keep it short
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]
        compressed = f"{base_name}_{hash_suffix}"
    
    return f"{compressed}{ext}"


def get_output_subfolders(output_dir, mode_tag, category=None):
    if category:
        base = os.path.join(output_dir, mode_tag, category)
    else:
        base = os.path.join(output_dir, mode_tag)
    
    # Check if path is too long and use shorter paths if needed
    if len(base) > 200:
        print(f"WARNING: Path too long ({len(base)} chars), using shorter paths")
        # Use a shorter base directory
        short_base = get_short_path(base)
        base = os.path.join(os.path.dirname(output_dir), short_base)
        print(f"Using shorter base path: {base}")
    
    data_dir = os.path.join(base, 'Data')
    plots_dir = os.path.join(base, 'Plots')
    pickles_dir = os.path.join(base, 'Pickles')
    
    print(f"Creating subfolders in: {base}")
    print(f"  - Data: {data_dir}")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Pickles: {pickles_dir}")
    
    try:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(pickles_dir, exist_ok=True)
        print("All subfolders created successfully")
    except Exception as e:
        print(f"Error creating subfolders: {e}")
        raise
    
    return data_dir, plots_dir, pickles_dir


def get_transition_graphs(data, label_column, output_dir, threshold=0.0, save_model=True, save_matrix=True, 
                         file_name_prefix="transition_graph", le=None, save_plots=True, 
                         permutation_threshold=False, n_perm=1000, perm_quantile=0.95, top_percent_threshold=False, top_percent=80.0, category='Individual', color_map=None):
    """Generate transition graphs with and without self-transitions."""
    # Determine mode tag for folder naming
    if permutation_threshold:
        mode_tag = f"permthresh_q{int(perm_quantile*100)}_n{n_perm}"
    elif top_percent_threshold:
        mode_tag = f"top{int(top_percent)}pct"
    else:
        mode_tag = f"thresh{int(round(threshold*100))}"
    data_dir, plots_dir, pickles_dir = get_output_subfolders(output_dir, mode_tag, category=category)

    sequences, le = encode_and_get_sequences(data, label_column, le)

    # Create a Markov chain model
    model = MarkovChain.from_samples(sequences)
    if save_model:
        # Ensure the pickles directory exists before saving
        print(f"Creating directory: {pickles_dir}")
        print(f"Path length: {len(pickles_dir)} characters")
        try:
            os.makedirs(pickles_dir, exist_ok=True)
            print(f"Directory created successfully: {pickles_dir}")
            # Verify directory actually exists
            if not os.path.exists(pickles_dir):
                print(f"ERROR: Directory does not exist after creation: {pickles_dir}")
                raise FileNotFoundError(f"Directory creation failed: {pickles_dir}")
        except Exception as e:
            print(f"Error creating directory {pickles_dir}: {e}")
            raise
        # Save the model
        model_filename = os.path.join(pickles_dir, f'{file_name_prefix}_model_with_self_transitions.pkl')
        print(f"Saving model to: {model_filename}")
        print(f"Model file path length: {len(model_filename)} characters")
        
        # Check if path is too long and shorten filename if needed
        if len(model_filename) >= 260:
            print(f"WARNING: File path too long ({len(model_filename)} chars), shortening filename")
            # Create a shorter filename
            short_filename = f'model1_{hashlib.md5(file_name_prefix.encode()).hexdigest()[:8]}.pkl'
            model_filename = os.path.join(pickles_dir, short_filename)
            print(f"Using shorter filename: {model_filename}")
            print(f"New path length: {len(model_filename)} characters")
        
        # Verify parent directory exists before saving
        if not os.path.exists(os.path.dirname(model_filename)):
            print(f"ERROR: Parent directory does not exist: {os.path.dirname(model_filename)}")
            raise FileNotFoundError(f"Parent directory does not exist: {os.path.dirname(model_filename)}")
        dump(model, model_filename)
    matrix_with_self_transitions = restructure_matrix(model)

    encoded_labels = matrix_with_self_transitions.columns
    decoded_labels = le.inverse_transform(encoded_labels)
    matrix_with_self_transitions.columns = decoded_labels
    matrix_with_self_transitions.index = decoded_labels
    
    # Only save CSV for matrix with self-transitions
    if save_matrix:
        csv_filename = f'{file_name_prefix}_with_self_transitions.csv'
        csv_path = get_safe_filename(data_dir, csv_filename)
        matrix_with_self_transitions.to_csv(csv_path)

    # Create sequence without self-transitions using original labels
    original_labels = le.inverse_transform(sequences[0])
    new_sequence = [original_labels[0]]
    for i in range(1, len(original_labels)):
        if original_labels[i] != original_labels[i - 1]:
            new_sequence.append(original_labels[i])

    data2 = pd.DataFrame(new_sequence, columns=[label_column])

    sequences2, le2 = encode_and_get_sequences(data2, label_column, le)
    # Create a Markov chain model
    model2 = MarkovChain.from_samples(sequences2)
    if save_model:
        # Ensure the pickles directory exists before saving
        print(f"Creating directory (second model): {pickles_dir}")
        print(f"Path length (second model): {len(pickles_dir)} characters")
        try:
            os.makedirs(pickles_dir, exist_ok=True)
            print(f"Directory created successfully (second model): {pickles_dir}")
            # Verify directory actually exists
            if not os.path.exists(pickles_dir):
                print(f"ERROR: Directory does not exist after creation (second model): {pickles_dir}")
                raise FileNotFoundError(f"Directory creation failed: {pickles_dir}")
        except Exception as e:
            print(f"Error creating directory {pickles_dir}: {e}")
            raise
        # Save the model
        model_filename2 = os.path.join(pickles_dir, f'{file_name_prefix}_model_without_self_transitions.pkl')
        print(f"Saving second model to: {model_filename2}")
        print(f"Second model file path length: {len(model_filename2)} characters")
        
        # Check if path is too long and shorten filename if needed
        if len(model_filename2) >= 260:
            print(f"WARNING: Second model file path too long ({len(model_filename2)} chars), shortening filename")
            # Create a shorter filename
            short_filename2 = f'model2_{hashlib.md5(file_name_prefix.encode()).hexdigest()[:8]}.pkl'
            model_filename2 = os.path.join(pickles_dir, short_filename2)
            print(f"Using shorter filename for second model: {model_filename2}")
            print(f"New second model path length: {len(model_filename2)} characters")
        
        # Verify parent directory exists before saving
        if not os.path.exists(os.path.dirname(model_filename2)):
            print(f"ERROR: Parent directory does not exist (second model): {os.path.dirname(model_filename2)}")
            raise FileNotFoundError(f"Parent directory does not exist: {os.path.dirname(model_filename2)}")
        dump(model2, model_filename2)

    matrix_without_self_transitions = restructure_matrix(model2)
    matrix_without_self_transitions.columns = decoded_labels
    matrix_without_self_transitions.index = decoded_labels
    
    filtered_matrix = None
    if save_plots:
        base_tag = file_name_prefix + "_exclself"
        if permutation_threshold:
            label_sequences = [trial[label_column].values for _, trial in data.groupby(['ppid', 'session', 'trial'])]
            mask = permutation_edge_threshold(matrix_without_self_transitions, label_sequences, n_perm=n_perm, quantile=perm_quantile)
            filtered_matrix = matrix_without_self_transitions.copy()
            filtered_matrix.values[~mask] = 0.0
            perm_tag = f"permthresh_q{int(perm_quantile*100)}_n{n_perm}"
            create_heatmap_graph(filtered_matrix, threshold=0.0,
                                title=f"{base_tag} ({perm_tag})",
                                output_dir=plots_dir,
                                file_tag=f"{base_tag}_{perm_tag}",
                                color_map=color_map)
            if save_matrix:
                csv_filename = f'{base_tag}_{perm_tag}.csv'
                csv_path = get_safe_filename(data_dir, csv_filename)
                filtered_matrix.to_csv(csv_path)
        elif top_percent_threshold:
            mask = top_percent_edge_threshold(matrix_without_self_transitions, top_percent=top_percent)
            filtered_matrix = matrix_without_self_transitions.copy()
            filtered_matrix.values[~mask] = 0.0
            top_tag = f"top{int(top_percent)}pct"
            create_heatmap_graph(filtered_matrix, threshold=0.0,
                                title=f"{base_tag} ({top_tag})",
                                output_dir=plots_dir,
                                file_tag=f"{base_tag}_{top_tag}",
                                color_map=color_map)
            if save_matrix:
                csv_filename = f'{base_tag}_{top_tag}.csv'
                csv_path = get_safe_filename(data_dir, csv_filename)
                filtered_matrix.to_csv(csv_path)
        else:
            thresh_pct = int(round(threshold * 100))
            hard_tag = f"thresh{thresh_pct}"
            filtered_matrix = matrix_without_self_transitions.copy()
            filtered_matrix[filtered_matrix < threshold] = 0.0
            create_heatmap_graph(filtered_matrix, threshold=threshold,
                                title=f"{base_tag} ({hard_tag})",
                                output_dir=plots_dir,
                                file_tag=f"{base_tag}_{hard_tag}",
                                color_map=color_map)
            if save_matrix:
                csv_filename = f'{base_tag}_{hard_tag}.csv'
                csv_path = get_safe_filename(data_dir, csv_filename)
                filtered_matrix.to_csv(csv_path)
    else:
        # If not plotting, still return a filtered matrix for downstream use
        if permutation_threshold:
            label_sequences = [trial[label_column].values for _, trial in data.groupby(['ppid', 'session', 'trial'])]
            mask = permutation_edge_threshold(matrix_without_self_transitions, label_sequences, n_perm=n_perm, quantile=perm_quantile)
            filtered_matrix = matrix_without_self_transitions.copy()
            filtered_matrix.values[~mask] = 0.0
        elif top_percent_threshold:
            mask = top_percent_edge_threshold(matrix_without_self_transitions, top_percent=top_percent)
            filtered_matrix = matrix_without_self_transitions.copy()
            filtered_matrix.values[~mask] = 0.0
        else:
            filtered_matrix = matrix_without_self_transitions.copy()
            filtered_matrix[filtered_matrix < threshold] = 0.0
    return matrix_with_self_transitions, matrix_without_self_transitions, le, filtered_matrix


def plot_block_comparison_figure(matrix1, matrix2, labels, output_dir, block1_name, block2_name, file_tag=None, layout_k=4.0, cluster_spacing=8.0, n_obs1=None, n_obs2=None):
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np
    # Ensure output_dir exists
    os.makedirs(output_dir, exist_ok=True)
    # Compute common node set and fixed order
    # Only include nodes present in either thresholded matrix (nonzero rows/cols)
    nonzero_nodes1 = set(matrix1.index[(matrix1.abs().sum(axis=1) + matrix1.abs().sum(axis=0)) > 0])
    nonzero_nodes2 = set(matrix2.index[(matrix2.abs().sum(axis=1) + matrix2.abs().sum(axis=0)) > 0])
    common_labels = sorted(nonzero_nodes1 | nonzero_nodes2)
    # Reindex both matrices to common_labels for both rows and columns
    mat1 = matrix1.reindex(index=common_labels, columns=common_labels, fill_value=0)
    mat2 = matrix2.reindex(index=common_labels, columns=common_labels, fill_value=0)
    # Compute node layout from reference block (mat1) using only these nodes
    G1 = nx.from_pandas_adjacency(mat1, create_using=nx.DiGraph())
    node_size_base = 600
    node_size_scale = 2000
    try:
        from networkx.algorithms import community
        communities = list(community.louvain_communities(G1.to_undirected(), weight='weight'))
        pos = _separate_cluster_layout(G1, communities, base_k=layout_k, cluster_spacing=cluster_spacing)
    except Exception:
        pos = nx.spring_layout(G1, k=layout_k, iterations=100, seed=42)
    fig, axes = plt.subplots(1, 3, figsize=(30, 12))  # Larger figure
    
    # Create titles with observation counts if provided
    title1 = block1_name
    title2 = block2_name
    if n_obs1 is not None:
        title1 = f"{block1_name} (n={n_obs1})"
    if n_obs2 is not None:
        title2 = f"{block2_name} (n={n_obs2})"
    
    panels = [(mat1, axes[0], title1, 'Blues'),
              (mat2, axes[1], title2, 'Blues')]
    def edge_widths(weights, max_width=8, min_width=0.5):
        max_w = max(abs(w) for w in weights) if weights else 1.0
        return [min_width + (max_width - min_width) * abs(w) / max_w for w in weights]
    def get_node_sizes(G):
        return [node_size_base + node_size_scale * (G.in_degree(n) + G.out_degree(n)) / (1 + max(G.in_degree(n) + G.out_degree(n) for n in G.nodes())) for n in G.nodes()]
    def edge_indices(G, edgelist):
        edge_idx = {e: i for i, e in enumerate(G.edges())}
        return [edge_idx[e] for e in edgelist]
    def get_perpendicular_offset(pos, u, v, magnitude=0.12):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            return (0, 0)
        # Perpendicular vector
        perp = (-dy / length, dx / length)
        return (perp[0] * magnitude, perp[1] * magnitude)
    # Panel 1 & 2: block graphs
    for idx, (mat, ax, title, cmap) in enumerate(panels):
        aplha = 0.6
        arrow_style = '->'
        pos_radius = 'arc3,rad=0.4'
        neg_radius = 'arc3,rad=0.4'
        unidir_radius = 'arc3,rad=0.2'
        G = nx.from_pandas_adjacency(mat, create_using=nx.DiGraph())
        node_sizes = get_node_sizes(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color='white', edgecolors='black', linewidths=2, alpha=1.0)
        edge_colors = ['#2b4a8b'] * G.number_of_edges()
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        widths = edge_widths(weights)
        # Identify bidirectional pairs
        bidir_pairs = set()
        for u, v in G.edges():
            if G.has_edge(v, u) and u < v:
                bidir_pairs.add((u, v))
        # Draw bidirectional edges with perpendicular offset
        for u, v in bidir_pairs:
            idx_uv = list(G.edges()).index((u, v))
            idx_vu = list(G.edges()).index((v, u))
            offset = get_perpendicular_offset(pos, u, v, magnitude=0.12)
            # Offset for u->v
            pos_uv = pos.copy()
            pos_uv[u] = (pos[u][0] + offset[0], pos[u][1] + offset[1])
            pos_uv[v] = (pos[v][0] + offset[0], pos[v][1] + offset[1])
            arrows1 = nx.draw_networkx_edges(
                G, pos_uv, ax=ax,
                edgelist=[(u, v)],
                edge_color=[edge_colors[idx_uv]],
                width=[widths[idx_uv]],
                alpha=aplha,
                arrows=True,
                arrowsize=20,
                arrowstyle=arrow_style,
                min_source_margin=15,
                min_target_margin=15,
                connectionstyle=pos_radius
            )
            if arrows1 is not None:
                for arrow in arrows1:
                    arrow.set_edgecolor(arrow.get_facecolor())
            # Offset for v->u (opposite direction)
            pos_vu = pos.copy()
            pos_vu[u] = (pos[u][0] - offset[0], pos[u][1] - offset[1])
            pos_vu[v] = (pos[v][0] - offset[0], pos[v][1] - offset[1])
            arrows2 = nx.draw_networkx_edges(
                G, pos_vu, ax=ax,
                edgelist=[(v, u)],
                edge_color=[edge_colors[idx_vu]],
                width=[widths[idx_vu]],
                alpha=aplha,
                arrows=True,
                arrowsize=20,
                arrowstyle=arrow_style,
                min_source_margin=15,
                min_target_margin=15,
                connectionstyle=neg_radius
            )
            if arrows2 is not None:
                for arrow in arrows2:
                    arrow.set_edgecolor(arrow.get_facecolor())
        # Draw unidirectional edges (not part of any bidir pair)
        bidir_edges = set()
        for u, v in bidir_pairs:
            bidir_edges.add((u, v))
            bidir_edges.add((v, u))
        unidir_edges = [(u, v) for u, v in G.edges() if (u, v) not in bidir_edges]
        if unidir_edges:
            idxs = edge_indices(G, unidir_edges)
            arrows3 = nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=unidir_edges,
                edge_color=[edge_colors[i] for i in idxs],
                width=[widths[i] for i in idxs],
                alpha=aplha,
                arrows=True,
                arrowsize=20,
                arrowstyle=arrow_style,
                min_source_margin=15,
                min_target_margin=15,
                connectionstyle=pos_radius
            )
            if arrows3 is not None:
                for arrow in arrows3:
                    arrow.set_edgecolor(arrow.get_facecolor())
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.axis('off')
        legend = ax.legend([Line2D([0], [0], color='#2b4a8b', lw=w) for w in [8, 4, 1]],
                  ['2.5%', '1.0%', '0.1%'],
                  title='Bigram Probability', loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=12, title_fontsize=14, frameon=False)
        ax.add_artist(legend)
    # Panel 3: difference graph (only for edges present after thresholding)
    # Only keep edges present in either thresholded matrix
    diff = mat2 - mat1
    # Remove edges not present in either thresholded matrix
    mask = ((mat1.abs() > 0) | (mat2.abs() > 0))
    diff = diff.where(mask, 0)
    Gd = nx.from_pandas_adjacency(diff, create_using=nx.DiGraph())
    node_sizes = get_node_sizes(Gd)
    nx.draw_networkx_nodes(Gd, pos, ax=axes[2], node_size=node_sizes, node_color='white', edgecolors='black', linewidths=2, alpha=1.0)
    edge_colors = ['#2b4a8b' if Gd[u][v]['weight'] > 0 else '#b22222' for u, v in Gd.edges()]
    weights = [Gd[u][v]['weight'] for u, v in Gd.edges()]
    widths = edge_widths(weights)
    bidir_pairs = set()
    for u, v in Gd.edges():
        if Gd.has_edge(v, u) and u < v:
            bidir_pairs.add((u, v))
    for u, v in bidir_pairs:
        idx_uv = list(Gd.edges()).index((u, v))
        idx_vu = list(Gd.edges()).index((v, u))
        offset = get_perpendicular_offset(pos, u, v, magnitude=0.12)
        # Offset for u->v
        pos_uv = pos.copy()
        pos_uv[u] = (pos[u][0] + offset[0], pos[u][1] + offset[1])
        pos_uv[v] = (pos[v][0] + offset[0], pos[v][1] + offset[1])
        arrows1 = nx.draw_networkx_edges(
            Gd, pos_uv, ax=axes[2],
            edgelist=[(u, v)],
            edge_color=[edge_colors[idx_uv]],
            width=[widths[idx_uv]],
            alpha=aplha,
            arrows=True,
            arrowsize=20,
            arrowstyle=arrow_style,
            min_source_margin=15,
            min_target_margin=15,
            connectionstyle=pos_radius
        )
        if arrows1 is not None:
            for arrow in arrows1:
                arrow.set_edgecolor(arrow.get_facecolor())
        # Offset for v->u (opposite direction)
        pos_vu = pos.copy()
        pos_vu[u] = (pos[u][0] - offset[0], pos[u][1] - offset[1])
        pos_vu[v] = (pos[v][0] - offset[0], pos[v][1] - offset[1])
        arrows2 = nx.draw_networkx_edges(
            Gd, pos_vu, ax=axes[2],
            edgelist=[(v, u)],
            edge_color=[edge_colors[idx_vu]],
            width=[widths[idx_vu]],
            alpha=aplha,
            arrows=True,
            arrowsize=20,
            arrowstyle=arrow_style,
            min_source_margin=15,
            min_target_margin=15,
            connectionstyle=neg_radius
        )
        if arrows2 is not None:
            for arrow in arrows2:
                arrow.set_edgecolor(arrow.get_facecolor())
    bidir_edges = set()
    for u, v in bidir_pairs:
        bidir_edges.add((u, v))
        bidir_edges.add((v, u))
    unidir_edges = [(u, v) for u, v in Gd.edges() if (u, v) not in bidir_edges]
    if unidir_edges:
        idxs = edge_indices(Gd, unidir_edges)
        arrows3 = nx.draw_networkx_edges(
            Gd, pos, ax=axes[2],
            edgelist=unidir_edges,
            edge_color=[edge_colors[i] for i in idxs],
            width=[widths[i] for i in idxs],
            alpha=aplha,
            arrows=True,
            arrowsize=20,
            arrowstyle=arrow_style,
            min_source_margin=15,
            min_target_margin=15,
            connectionstyle=unidir_radius
        )
        if arrows3 is not None:
            for arrow in arrows3:
                arrow.set_edgecolor(arrow.get_facecolor())
    nx.draw_networkx_labels(Gd, pos, ax=axes[2], font_size=10, font_weight='bold')
    
    # Create difference panel title with observation counts if provided
    diff_title = f"{block2_name} - {block1_name}"
    if n_obs1 is not None and n_obs2 is not None:
        diff_title = f"{title2} - {title1}"
    elif n_obs2 is not None:
        diff_title = f"{title2} - {block1_name}"
    elif n_obs1 is not None:
        diff_title = f"{block2_name} - {title1}"
    
    axes[2].set_title(diff_title, fontsize=20, fontweight='bold')
    axes[2].axis('off')
    legend_diff = axes[2].legend([Line2D([0], [0], color='#2b4a8b', lw=8),
                    Line2D([0], [0], color='#2b4a8b', lw=4),
                    Line2D([0], [0], color='#b22222', lw=4),
                    Line2D([0], [0], color='#b22222', lw=8)],
                   ['+2.5%', '+1.0%', '-1.0%', '-2.5%'],
                   title='Bigram Probability Change', loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=4, fontsize=12, title_fontsize=14, frameon=False)
    axes[2].add_artist(legend_diff)
    # Add color legend for difference direction
    color_legend = [
        Line2D([0], [0], color='#2b4a8b', lw=4, label='More probable in second block'),
        Line2D([0], [0], color='#b22222', lw=4, label='More probable in first block')
    ]
    axes[2].legend(
        handles=color_legend,
        loc='lower left',
        bbox_to_anchor=(0, -0.08),
        fontsize=14,
        frameon=False,
        title='Edge color: Difference direction'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if file_tag is None:
        file_tag = f"compare_{block2_name}_vs_{block1_name}"
    filename = f"{file_tag}_three_panel.png"
    fig_path = get_safe_filename(output_dir, filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print_color(f"Saved three-panel comparison figure: {fig_path}", bold_color.GREEN)


@tictoc
def main():
    parser = argparse.ArgumentParser(description='Generate syllable transition graphs from labeled data using new metadata approach.')
    parser.add_argument('--labels-path', type=str, required=True, help='Path to the labels dataset CSV file')
    parser.add_argument('--label-column', type=str, required=True, help='Name of the column containing the labels')
    parser.add_argument('--metadata-path', type=str, required=True, help='Path to the new metadata dataset CSV file')
    parser.add_argument('--output-dir', type=str, required=False, default=None, help='Path to the output directory')
    parser.add_argument('--colour-map-object', type=str, required=False, default=None, help='Path to the colour map object')
    parser.add_argument('--threshold', type=float, required=False, default=0.0, help='Minimum probability threshold for including edges in graphs (0.0 to 1.0)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--exclude-block', type=str, required=False, default=None, help='Name of the block to exclude when creating "all_but_excluded" group (optional)')
    parser.add_argument('--permutation-threshold', action='store_true', help='Use permutation-based thresholding for edges')
    parser.add_argument('--n-perm', type=int, required=False, default=1000, help='Number of permutations for permutation thresholding')
    parser.add_argument('--perm-quantile', type=float, required=False, default=0.99, help='Quantile for permutation thresholding (e.g., 0.95 for 95th percentile)')
    parser.add_argument('--top-percent-threshold', action='store_true', help='Use top NN%% in/out edge weight thresholding')
    parser.add_argument('--top-percent', type=float, required=False, default=80.0, help='Top NN%% of cumulative in/out edge weights to keep (default: 80)')
    parser.add_argument('--layout-k', type=float, required=False, default=20.0, help='Node repulsion parameter for layout (higher = more separation, default: 4.0)')
    parser.add_argument('--cluster-spacing', type=float, required=False, default=5.0, help='Cluster spacing for community layout (default: 8.0)')
    parser.add_argument('--failed-trials-path', type=str, required=False, default=None, help='Path to CSV file containing metadata of failed trials (must have ppid, session, trial columns)')
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")
    
    if args.output_dir is None:
        output_dir = os.path.dirname(args.metadata_path)
    else:
        output_dir = args.output_dir
    output_dir = os.path.join(output_dir, 'markov_transition_graphs_output')
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    dataset1 = pd.read_csv(args.labels_path)
    dataset2 = pd.read_csv(args.metadata_path)
    
    # Load failed trials data if provided
    failed_trials_data = None
    if args.failed_trials_path:
        try:
            failed_trials_data = pd.read_csv(args.failed_trials_path)
            
            # Check for different column naming conventions
            possible_ppid_cols = ['ppid', 'ppid_vr', 'ppid_coda']
            possible_session_cols = ['session', 'session_vr', 'session_coda']
            possible_trial_cols = ['trial', 'trial_vr', 'trial_coda']
            
            # Find the actual column names
            ppid_col = next((col for col in possible_ppid_cols if col in failed_trials_data.columns), None)
            session_col = next((col for col in possible_session_cols if col in failed_trials_data.columns), None)
            trial_col = next((col for col in possible_trial_cols if col in failed_trials_data.columns), None)
            
            if ppid_col and session_col and trial_col:
                # Rename columns to standard names for consistency
                failed_trials_data = failed_trials_data.rename(columns={
                    ppid_col: 'ppid',
                    session_col: 'session', 
                    trial_col: 'trial'
                })
                print_color(f"Loaded failed trials data with {len(failed_trials_data)} trials", bold_color.GREEN)
                print_color(f"Using columns: {ppid_col}->ppid, {session_col}->session, {trial_col}->trial", bold_color.BLUE)
            else:
                missing_cols = []
                if not ppid_col:
                    missing_cols.append("ppid (or ppid_vr, ppid_coda)")
                if not session_col:
                    missing_cols.append("session (or session_vr, session_coda)")
                if not trial_col:
                    missing_cols.append("trial (or trial_vr, trial_coda)")
                print_color(f"Warning: Failed trials CSV missing required columns: {missing_cols}. Failed trials will be ignored.", bold_color.YELLOW)
                failed_trials_data = None
        except Exception as e:
            print_color(f"Warning: Could not load failed trials CSV: {str(e)}. Failed trials will be ignored.", bold_color.YELLOW)
            failed_trials_data = None
    
    # Verify required columns exist in metadata
    required_columns = ['end_state', 'reward', 'instructed_action', 'day_time', 'scenario', 
                       'scenario_id', 'threat', 'visibility', 'threat_distance', 'threat_action', 
                       'weapons', 'safehouse', 'threat2', 'threat_action2', 'fixed_order', 'episode_set_name2']
    
    missing_columns = [col for col in required_columns if col not in dataset2.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in metadata file: {missing_columns}")
    
    # Use columns ending with "2" for specific fields as requested
    # For threat, threat_action, and episode_set_name, use the "2" versions
    threat_column = 'threat2'
    threat_action_column = 'threat_action2'
    episode_set_name_column = 'episode_set_name2'
    
    print_color(f"Using columns: {threat_column}, {threat_action_column}, {episode_set_name_column}", bold_color.GREEN)
    # Mutually exclusive thresholding info
    if args.permutation_threshold:
        print_color(f"Thresholding mode: Permutation-based (n_perm={args.n_perm}, quantile={args.perm_quantile})", bold_color.GREEN)
    elif args.top_percent_threshold:
        print_color(f"Thresholding mode: Top cumulative percent (top_percent={args.top_percent}%)", bold_color.GREEN)
    else:
        print_color(f"Thresholding mode: Hard threshold (threshold={args.threshold})", bold_color.GREEN)
    
    # Filter out invalid labels
    dataset1 = dataset1[~dataset1[args.label_column].isin([-1, '-1'])]
    
    # First, create a label encoder that knows all possible labels
    le = LabelEncoder()
    le.fit(dataset1[args.label_column].unique())
    
    # Create color map
    unique_syllables = sorted(dataset1[args.label_column].unique())
    color_map = create_color_map(le, colour_map_arg=args.colour_map_object)
    
    # Count number of unique observations (ppid-session-trial combinations)
    n_observations = len(dataset1.groupby(['ppid', 'session', 'trial']))

    # Define the blocks of interest
    block_names_of_interest = ['instructed', 'dictionary', 'stump', 'fixed', 'failed']

    # Always create 'all_trials' (all data)
    analysis_groups = {'all_trials': dataset1.copy()}

    # Always create 'all_main' (all data except 'instructed' and 'failed')
    if episode_set_name_column in dataset2.columns:
        # Find all trials belonging to 'instructed' or 'failed' blocks
        excluded_blocks = ['instructed', 'failed']
        excluded_trials = set()
        for block_name in excluded_blocks:
            block_trials = set(
                tuple(x) for x in dataset2[dataset2[episode_set_name_column] == block_name][['ppid', 'session', 'trial']].values
            )
            excluded_trials.update(block_trials)
        mask = ~dataset1[['ppid', 'session', 'trial']].apply(tuple, axis=1).isin(excluded_trials)
        analysis_groups['all_main'] = dataset1[mask].copy()
    else:
        analysis_groups['all_main'] = dataset1.copy()

    # For each block of interest, create a group if present in metadata
    if episode_set_name_column in dataset2.columns:
        for block_name in block_names_of_interest:
            block_trials = dataset2[dataset2[episode_set_name_column] == block_name][['ppid', 'session', 'trial']]
            if not block_trials.empty:
                block_data = dataset1.merge(block_trials, on=['ppid', 'session', 'trial'], how='inner')
                analysis_groups[block_name] = block_data

    # Filter out failed trials if failed trials data is provided
    if failed_trials_data is not None:
        print_color("\nFiltering out failed trials from analysis groups...", bold_color.BLUE)
        
        # Create a set of failed trial identifiers for efficient lookup
        failed_trial_set = set(
            tuple(x) for x in failed_trials_data[['ppid', 'session', 'trial']].values
        )
        
        # Filter each analysis group to remove failed trials
        for group_name, group_data in analysis_groups.items():
            if not group_data.empty:
                # Create mask to exclude failed trials
                failed_mask = group_data[['ppid', 'session', 'trial']].apply(tuple, axis=1).isin(failed_trial_set)
                
                if failed_mask.sum() > 0:
                    print_color(f"Removing {failed_mask.sum()} failed trials from {group_name} group", bold_color.YELLOW)
                    analysis_groups[group_name] = group_data[~failed_mask].copy()
                else:
                    print_color(f"No failed trials found in {group_name} group", bold_color.GREEN)
    
    # Print summary of created groups
    print_color("\nSummary of analysis groups:", bold_color.CYAN)
    for group_name, group_data in analysis_groups.items():
        print_color(f"  {group_name}: {len(group_data.groupby(['ppid', 'session', 'trial']))} unique trials", bold_color.GREEN)

    # Process all trials together
    print_color("\nProcessing all trials together:", bold_color.CYAN)
    
    # Get the transition graphs and label encoder for all trials
    matrix_with_self_transitions, matrix_without_self_transitions, _, filtered_matrix = get_transition_graphs(
        analysis_groups['all_trials'], args.label_column, output_dir, threshold=args.threshold,
        file_name_prefix=f"transition_matrix_n{n_observations}_all_trials",
        save_matrix=not args.no_plots, le=le, save_plots=not args.no_plots,
        permutation_threshold=args.permutation_threshold, n_perm=args.n_perm, perm_quantile=args.perm_quantile,
        top_percent_threshold=args.top_percent_threshold, top_percent=args.top_percent,
        category='Individual', color_map=color_map
    )

    # After running get_transition_graphs for each group, collect block matrices
    block_matrices = {}
    block_observation_counts = {}  # Store observation counts for each block
    for group_name, group_data in analysis_groups.items():
        if group_data.empty:
            continue
        n_observations_group = len(group_data.groupby(['ppid', 'session', 'trial']))
        block_observation_counts[group_name] = n_observations_group
        matrix_with_self_transitions_group, matrix_without_self_transitions_group, _, filtered_matrix_group = get_transition_graphs(
            group_data, args.label_column, output_dir, threshold=args.threshold,
            file_name_prefix=f"transition_matrix_n{n_observations_group}_{group_name}",
            save_matrix=not args.no_plots, le=le, save_plots=not args.no_plots,
            permutation_threshold=args.permutation_threshold, n_perm=args.n_perm, perm_quantile=args.perm_quantile,
            top_percent_threshold=args.top_percent_threshold, top_percent=args.top_percent,
            category='Individual', color_map=color_map
        )
        block_matrices[group_name] = filtered_matrix_group

    # Now compare all pairs of blocks and plot three-panel figures
    block_names = list(block_matrices.keys())
    for i in range(len(block_names)):
        for j in range(len(block_names)):
            if i == j:
                continue
            name1, name2 = block_names[i], block_names[j]
            mat1, mat2 = block_matrices[name1], block_matrices[name2]
            # Only compare if both matrices are non-empty and have overlapping labels
            if mat1.empty or mat2.empty:
                continue
            common_labels = sorted(set(mat1.index) & set(mat2.index))
            if not common_labels:
                continue
            # Use the same Plots directory as other plots
            mode_tag = f"permthresh_q{int(args.perm_quantile*100)}_n{args.n_perm}" if args.permutation_threshold else f"top{int(args.top_percent)}pct" if args.top_percent_threshold else f"thresh{int(round(args.threshold*100))}"
            _, plots_dir, _ = get_output_subfolders(output_dir, mode_tag, category='Comparison')
            plot_block_comparison_figure(
                matrix1=mat1,
                matrix2=mat2,
                labels=common_labels,
                output_dir=plots_dir,
                block1_name=name1,
                block2_name=name2,
                file_tag=f"compare_{name2}_vs_{name1}",
                layout_k=args.layout_k,
                cluster_spacing=args.cluster_spacing,
                n_obs1=block_observation_counts.get(name1),
                n_obs2=block_observation_counts.get(name2)
            )

    print_color("\nTransition graphs analysis completed successfully!", bold_color.BLUE)
    print_color(f"\nThe data can be found in: ".capitalize() + f"{output_dir}", bold_color.BLUE)


if __name__ == "__main__":
    print_welcome_message()
    main() 