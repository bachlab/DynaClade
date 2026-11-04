import os
import glob
import re
import networkx as nx
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import art
import time
import inspect
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
from natsort import natsorted
import itertools
import seaborn as sns
import pandas as pd
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# taskkill /IM python.exe /F

def print_welcome_message():
    print(art.text2art("Network Analysis", font='standard'))
    print("\033[1;36m" + "by uSerratos " + "\033[0m")
    print("\n" + "="*80)
    print("\nThis script will compare networks using the Graph Edit Distance (GED)")
    print("\n" + "="*80 + "\n")

    return None


def tictoc(func):
    def inner(*args, **kwargs):
        T1 = time.time()
        print(f"\n\033[92mStart {time.strftime('%H:%M:%S', time.gmtime(T1))} | Function: {func.__name__} | File: {inspect.getfile(func)}\033[0m\n")
        func_out = func(*args, **kwargs)
        T2 = time.time()
        T = time.strftime("%H:%M:%S", time.gmtime(T2 - T1))
        print(f"\n\033[92mEnd {time.strftime('%H:%M:%S', time.gmtime(T1))} | Function: {func.__name__} | Total runTime {T}\033[0m\n")
        return func_out
    return inner


def df_to_networkx(df) -> object:
    df = df.T
    G = nx.DiGraph()
    for target, row in df.iterrows():
        for source, prob in row.items():
            if prob > 0:  # Only add edges with non-zero probability
                G.add_edge(source, target, weight=prob)
                G.nodes[source]['name'] = source
                G.nodes[target]['name'] = target
    return G


def draw_graph(G, key, show:bool=False, save:bool=False, save_path:str=None) -> None:
    # Get all nodes in the graph that are part of at least one edge
    connected_nodes = set(edge for edges in G.edges() for edge in edges)

    # Create a subgraph that contains only the connected nodes
    subgraph = G.subgraph(connected_nodes)

    plt.figure(figsize=(15, 15))

    # Calculate the positions for all nodes in the graph
    pos = nx.spring_layout(subgraph)

    node_color = ((255/256, 0, 127/256),)  
    edge_color = ((0.3, 0.3, 0.3),)  
    edge_alpha = 0.7
    node_size = 1500
    alpha = 0.6  

    # Draw the nodes
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_size, node_color=node_color, alpha=alpha, edgecolors='none')

    # Get edge weights
    weights = [subgraph[u][v]['weight'] for u,v in subgraph.edges()]

    # Normalize weights to be in the range [0, 1]
    weights = [float(i)/max(weights) for i in weights]

    # Multiply by a factor to increase visibility of edges
    weights = [i * 5 for i in weights]
 
    edges = nx.draw_networkx_edges(subgraph, pos, node_size=node_size, arrowstyle='-|>', 
                                arrowsize=10, width=weights, edge_color=edge_color,
                                connectionstyle='arc3,rad=0.1', alpha=edge_alpha)
    

    nx.draw_networkx_labels(subgraph, pos)
    edge_labels = nx.get_edge_attributes(subgraph, 'weight')
    
    edge_labels_pos = {}
    for edge in subgraph.edges():
        node, neighbor = edge
        edge_labels_pos[edge] = [(pos[node][0] + pos[neighbor][0]) / 2 + 0.01, 
                                (pos[node][1] + pos[neighbor][1]) / 2 + 0.01]
    

    plt.axis('off')
    plt.title(f'Transition matrix for {key.replace("_"," ")}', fontsize=20)
    plt.tight_layout()

    if save:
        if not save_path:
            print("The graph will not be saved as the save path is not provided.")
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            graph_name = os.path.join(save_path, f'graph_{key.replace("_transition_matrix"," ")}.png')
            plt.savefig(graph_name, format='png', dpi=300)

    if show: plt.show()

    plt.close()

    return None


def find_csv_files(directory, pattern:str="transition_matrix_P*.csv", patterns:str="P\d+_S\d+_T\d+") -> dict:
    # First, escape any regex special characters in the pattern
    pattern = re.escape(pattern).replace('\\*', '.*')
    
    # Find all CSV files recursively
    matching_files = glob.glob(f"{directory}/**/*.csv", recursive=True)
    
    # Filter files that match both patterns
    matching_files = [file for file in matching_files if re.search(patterns, file)]
    matching_files = [file for file in matching_files if re.search(pattern, os.path.basename(file))]
    
    # Create dictionary with file data
    data = {}
    for file in matching_files:
        key = re.search(patterns, file).group().replace(".", "")
        data[key] = {'data': pd.read_csv(file)}
    
    return data


def compare_networks(pair) -> tuple:
    i, graph1, j, graph2 = pair
    key1 = list(graph1.keys())[0]
    key2 = list(graph2.keys())[0]
    # print(f"\n\nComparing {key1} and {key2}")
    graph1 = graph1[key1]
    graph2 = graph2[key2]

    def node_del_cost(node1) -> float:
   
        edges = graph1.edges(node1['name'], data=True)
        weights = [edge[2].get('weight', None) for edge in edges]
        degree = graph1.degree()[node1['name']]
        # cost = degree * sum(weights)
        cost = degree

        # print(f"\ndelete: {node1} | degree: {graph1.degree()[node1['name']]} | weights: {weights} |cost:{cost}")

        return cost

    def node_ins_cost(node2) -> float:

        edges = graph2.edges(node2['name'], data=True)
        weights = [edge[2].get('weight', None) for edge in edges]
        degree = graph2.degree()[node2['name']]
        # cost = degree * sum(weights)
        cost = degree 

        # print(f"\ninsert: {node2} | degree: {graph2.degree()[node2['name']]} | weights: {weights} |cost:{cost}")

        return cost

    def node_subst_cost(node1, node2) -> float:
        # Get the degree attribute of each node, default to 0 if not present
        degree1 = graph1.degree()[node1['name']]
        degree2 = graph2.degree()[node2['name']]
        return abs(degree1 - degree2)

    def edge_subst_cost(edge1, edge2) -> float:
        # Get the weight attribute of each edge, default to 0 if not present
        weight1 = edge1['weight']
        weight2 = edge2['weight']
        return abs(weight1 - weight2)

    if i != j:  # Avoid comparing a network to itself
        ged = nx.optimize_graph_edit_distance(graph1, graph2, node_subst_cost=node_subst_cost, edge_subst_cost=edge_subst_cost, node_del_cost=node_del_cost, node_ins_cost=node_ins_cost)
        ged = next(ged)  # get the first (and best) result
        return (key1, key2), ged
    

def process_graph(args):
    key, df = args
    df_current = df['data'].iloc[:,1:].copy().set_index(df['data'].columns[1:])
    G = df_to_networkx(df_current)
    
    result = {
        'graph': G,
        'degree_centrality': nx.degree_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G),
        'closeness_centrality': nx.closeness_centrality(G)
    }
    
    try:
        result['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=5000)
    except (nx.NetworkXPointlessConcept, nx.NetworkXError):
        result['eigenvector_centrality'] = {node: 0.0 for node in G.nodes()}
        print(f"Warning: Graph for {key} is empty or has no edges. Setting eigenvector centrality to 0.")
    
    return key, result

@tictoc
def get_graphs(data: dict, output_dir: str) -> dict:
    output_dir_graphs = os.path.join(output_dir, 'graphs')
    if not os.path.exists(output_dir_graphs):
        os.makedirs(output_dir_graphs)

    # Prepare arguments for parallel processing
    args = [(key, df) for key, df in data.items()]
    
    # Use number of CPUs minus 1 to avoid overloading
    n_cores = max(1, cpu_count() - 1)
    
    # Process graphs in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_graph, arg) for arg in args]
        
        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing graphs'):
            key, result = future.result()
            data[key].update(result)
            
            # Draw graph (this needs to be done in the main process due to matplotlib limitations)
            draw_graph(result['graph'], key, show=False, save=True, save_path=output_dir_graphs)
             
            # Plot and save heatmap
            title = f'GED Eigen Vector Centrality -- {key} '
            import pdb; pdb.set_trace()
            plot_heatmap(result['eigenvector_centrality'], nodes, show=False, save=True, 
                        save_path=output_dir_graphs,
                        title=title, annot=True)
    
    return data

@tictoc
def compare_all_networks(data: dict) -> dict:
    networks = [{f'{key}':df['graph']} for key, df in data.items()]
    pairs = list(itertools.product(range(len(networks)), repeat=2))
    networks_pairs = [(i, networks[i], j, networks[j]) for i, j in pairs if i != j]
    
    # Use number of CPUs minus 1 to avoid overloading
    n_cores = max(1, cpu_count() - 1)
    
    output = {}
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Submit all comparison tasks
        future_to_pair = {executor.submit(compare_networks, pair): pair for pair in networks_pairs}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_pair), total=len(networks_pairs), desc="Calculating GED"):
            try:
                key, ged = future.result()
                output[key] = ged
            except Exception as e:
                print(f"Error processing comparison: {str(e)}")
    
    return output


def plot_heatmap(data:np.ndarray, network_names: list, show:bool=False, save_path:str=None,title:str='Graph Edit Distance Heatmap', save:bool=False, annot:bool=False) -> None:
    plt.figure(figsize=(20, 20))  # Increased figure size for better readability
    
    import pdb; pdb.set_trace()
    # Ensure data is a DataFrame with proper indices
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, index=network_names, columns=network_names)
    
    # Plot the heatmap with proper tick labels
    ax = sns.heatmap(data, cmap='plasma', annot=annot, fmt='.2f', 
                     cbar_kws={'label': 'Graph Edit Distance'},
                     xticklabels=True, yticklabels=True)

    # Rotate labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout()

    if show: plt.show()

    if save and save_path:
        heatmaps_dir = os.path.join(save_path, 'heatmaps')
        if not os.path.exists(heatmaps_dir):
            os.makedirs(heatmaps_dir)
        plt.savefig(os.path.join(heatmaps_dir, f'{title.lower().replace(" ","_")}.png'), 
                    format='png', dpi=300, bbox_inches='tight')

    plt.close()

    return None


@tictoc
def process_network_comparisons(data: dict, plot_show:bool=False, plot_save:bool=False, plot_annot:bool=False, output_dir_name:str=None) -> None:

    ged_matrix = compare_all_networks(data=data)

    ged_matrix = {(f'{key[0].replace("_mean_transition_matrix", "")}', f'{key[1].replace("_mean_transition_matrix", "")}'):value for key, value in ged_matrix.items()}

    # Get the unique network names from the keys of ged_matrix
    network_names = natsorted(list(set(name for names in ged_matrix.keys() for name in names)))

    # Create a mapping from network names to indices
    network_indices = {name: i for i, name in enumerate(network_names)}
    
    # Initialize a 2D array with zeros
    ged_array = np.zeros((len(network_names), len(network_names)))
    
    # Fill in the array with GED values
    for (name1, name2), ged in ged_matrix.items():
        i = network_indices[name1]
        j = network_indices[name2]
        ged_array[i, j] = ged
        ged_array[j, i] = ged  # Assuming the GED is symmetric

    # Create a DataFrame from ged_array with network_names as the index and columns
    ged_df = pd.DataFrame(ged_array, index=network_names, columns=network_names)

    # Fit the scaler and transform ged_array
    ged_array_normalized = ged_array/max(ged_array.max(axis=0))

    # Create a DataFrame from ged_array with network_names as the index and columns
    ged_df_normalised = pd.DataFrame(ged_array_normalized, index=network_names, columns=network_names)

    graph_edit_distance_output_path = os.path.join(output_dir_name, 'graph_edit_distance')
    if not os.path.exists(graph_edit_distance_output_path):
        os.makedirs(graph_edit_distance_output_path)

    graph_edit_distance_matices_output_path = os.path.join(graph_edit_distance_output_path, 'matrices')
    if not os.path.exists(graph_edit_distance_matices_output_path):
        os.makedirs(graph_edit_distance_matices_output_path)

    condition = network_names[0].split('--')[0]
    ged_df.to_csv(os.path.join(graph_edit_distance_matices_output_path, f'ged_{condition}.csv'))
    ged_df_normalised.to_csv(os.path.join(graph_edit_distance_matices_output_path, f'gedNormalised_{condition}.csv'))

    plot_heatmap(ged_df, network_names, show=plot_show, title=f'Graph Edit Distance _{condition}', save=plot_save, save_path=graph_edit_distance_output_path, annot=plot_annot)
    plot_heatmap(ged_df_normalised, network_names, show=plot_show, title=f'Graph Edit Distance Normalised _{condition}', save=plot_save, save_path=graph_edit_distance_output_path, annot=plot_annot)

    return None


@tictoc
def main():
    data_path = input("Enter the path to the data folder: ")
    output_dir_name = f'{data_path}_network_analysis'
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    pattern = "*_mean_*.csv"
    patterns = r'([^/\\]+)\.' # Get the filename without the extension

    data = find_csv_files(data_path, pattern=pattern, patterns=patterns)
    
    if not data:
        print("\nNo matching files found. Please check the patterns and directory path.")
        return

    data_graphs = get_graphs(data=data, output_dir=output_dir_name)
    
    # Convert graph data to JSON-serializable format
    serializable_data = {}
    for key, value in data_graphs.items():
        serializable_data[key] = {
            'degree_centrality': {str(k): float(v) for k, v in value['degree_centrality'].items()},
            'betweenness_centrality': {str(k): float(v) for k, v in value['betweenness_centrality'].items()},
            'closeness_centrality': {str(k): float(v) for k, v in value['closeness_centrality'].items()},
            'eigenvector_centrality': {str(k): float(v) for k, v in value['eigenvector_centrality'].items()}
        }
    
    # Save as JSON
    with open(os.path.join(output_dir_name, 'graphs.json'), 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    process_network_comparisons(data_graphs, plot_show=False, plot_save=True, plot_annot=True, output_dir_name=output_dir_name)

if __name__ == "__main__":
    os.system('cls||clear')
    print_welcome_message()
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        # os.system('shutdown /h')  # Hibernate on Windows
