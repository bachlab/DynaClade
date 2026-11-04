"""Module for plotting syllable sequences and visualizations."""

# ==
# author: uSerratos
# date: Nov 2024
# ==

import os
import sys
import argparse
import pickle
import shutil
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Only import utils if it hasn't been imported already
if 'utils' not in sys.modules:
    if os.environ.get('RUNNING_AS_SUBPROCESS') != 'true':
        os.system("cls||clear")
    from utils import *

warnings.filterwarnings('ignore', category=RuntimeWarning)

def group_and_encode(df: pd.DataFrame, label_column: str) -> tuple:
    """Group data by trial info and encode labels.
    
    Args:
        df: DataFrame containing the data
        label_column: Name of the column containing labels
        
    Returns:
        tuple: (LabelEncoder, list of group keys, grouped DataFrame)
    """
    grouped = df.groupby(["ppid", "session", "trial"])
    group_keys = list(grouped.groups.keys())
    le = LabelEncoder()
    le.fit(df[label_column])

    return le, group_keys, grouped


def create_color_map(le: LabelEncoder, colour_map_arg=None, label_description_dictionary=None) -> dict:
    """Create a color map for labels.
    
    Args:
        le: Fitted LabelEncoder object
        colour_map_arg: Optional colormap name, dictionary, or file path
        label_description_dictionary: Optional dictionary with label descriptions

    Returns:
        dict: Mapping of labels to colors
    """
    # If colour_map_arg is a dictionary, use it directly
    if isinstance(colour_map_arg, dict):
        if label_description_dictionary:
            # Convert label descriptions to strings to ensure consistent key types
            colour_map_arg = {str(label_description_dictionary.get(label)): colour for label, colour in colour_map_arg.items()}
        return colour_map_arg

    # If colour_map_arg is a string and a valid file path, try to load it
    if isinstance(colour_map_arg, str) and os.path.exists(colour_map_arg):
        try:
            with open(colour_map_arg, "rb") as f:
                loaded_map = pickle.load(f)
            # If the loaded object is a dictionary, use it directly
            if isinstance(loaded_map, dict):
                if label_description_dictionary:
                    loaded_map = {str(label_description_dictionary.get(label)): colour for label, colour in loaded_map.items()}
                return loaded_map
            # If it's not a dictionary, use it as a colormap name
            colour_map_arg = loaded_map
        except Exception as e:
            print_color(f"\nError loading color map from {colour_map_arg}: {str(e)}", bold_color.RED)
            colour_map_arg = None

    # If we have a colormap name or None, create a new colormap
    if colour_map_arg:
        try:
            color_map = plt.colormaps.get_cmap(colour_map_arg)
        except ValueError:
            print_color(f"\nInvalid colormap name: {colour_map_arg}. Using default pastel rainbow.", bold_color.YELLOW)
            color_map = create_pastel_rainbow_cmap()
    else:
        color_map = create_pastel_rainbow_cmap()

    # Get all unique labels that need colors
    all_labels = set()
    if label_description_dictionary:
        all_labels.update(label_description_dictionary.values())
    all_labels.update(le.classes_)
    
    # Convert all labels to strings and remove any None values
    all_labels = {str(label) for label in all_labels if label is not None}
    
    # Normalize label indices to the range [0, 1]
    normalized_labels = [i / (len(all_labels) - 1) for i in range(len(all_labels))]
    
    # Get colors using the colormap
    cmap = [color_map(norm_label) for norm_label in normalized_labels]
    
    # Create label to color mapping
    color_map_dict = {label: cmap[i] for i, label in enumerate(sorted(all_labels))}

    # Ensure consistent special cases
    color_map_dict['other'] = '#D3D3D3'
    color_map_dict[-1] = '#D3D3D3'
    color_map_dict["-1"] = '#D3D3D3'

    return color_map_dict


def get_full_file_names(parent_path: str) -> dict:
    """Get full file paths for all output files.
    
    Args:
        parent_path: Base directory path
        
    Returns:
        dict: Mapping of file types to full paths
    """
    file_names = {
        "embedding": "embedding.csv",
        "metadata": "metadata.json",
        "embedding_full": "embedding_full.csv",
        "clusters_figure": "clusters_figure.png",
        "embedding_figure": "embedding_figure.png",
        "clusterer": "clusterer.joblib",
        "colour_map": "colour_map.pkl",
        "mapped_data": "mapped_data.csv",
    }

    return {k: os.path.join(parent_path, v) for k, v in file_names.items()}


def plot_label_sequence(
    x: np.ndarray,
    labels: np.ndarray,
    colors: list,
    color_map: dict,
    legend_args: dict
) -> None:
    """Plot a sequence of labels as colored bars.
    
    Args:
        x: Time points for x-axis
        labels: Sequence of labels
        colors: List of colors for each label
        color_map: Mapping of labels to colors
        legend_args: Arguments for legend formatting
    """
    plt.figure(figsize=(8, 1))
    plt.bar(x, [1] * len(labels), color=colors)

    # Set axis limits and ticks
    plt.xlim([min(x), max(x)])
    plt.ylim([0, 1])

    ticks = np.arange(min(x), max(x), step=5)
    plt.xticks(
        ticks, 
        labels=[f"{int(tick)}s" for tick in ticks], 
        fontsize=10
    )

    # Remove plot borders
    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tick_params(axis="x", colors="grey")

    # Create legend patches
    patches = [
        plt.Line2D([0], [0], color=color, lw=5, label=label)
        for label, color in color_map.items()
    ]

    # Update and set legend arguments
    legend_args.update({
        "handles": patches,
        "prop": {"size": 4},
        "ncol": 2,
        "labelspacing": 0.05,
        "borderpad": 0.4,
        "fontsize": 4,
        "bbox_to_anchor": (1, 1.4),
    })

    plt.legend(**legend_args)
    plt.yticks([])


def plot_clusters(
    embedding: np.ndarray | pd.DataFrame,
    labels: np.ndarray | pd.DataFrame = None,
    title: str = None,
    marker_size: float = 2,
    marker_size_fixed: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot clusters in 3D space.
    
    Args:
        embedding: Data points to plot
        labels: Optional cluster labels
        title: Plot title
        marker_size: Size of markers
        marker_size_fixed: Whether to use fixed marker size
        show: Whether to display the plot
        
    Returns:
        plt.Figure: The created figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Convert inputs to DataFrames if needed
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
        unique_labels = labels.squeeze().unique()
        num_labels = len(unique_labels)
        plt.gca().set_prop_cycle(
            plt.cycler("color", plt.cm.jet(np.linspace(0, 1, num_labels)))
        )

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

    if show:
        plt.show()

    return fig

def process_single_group(args_tuple):
    """Process a single group of data for plotting.
    
    Args:
        args_tuple: Tuple containing (name, group, parent_dir_bar, labels_column, color_map, legend_args, show, block, DPI)
        
    Returns:
        None
    """
    name, group, parent_dir_bar, labels_column, label_description_column, color_map, legend_args, show, block, DPI = args_tuple
    ppid, session, trial = name
    ppid = int(ppid)
    session_str = str(int(session)).zfill(3)
    trial_str = str(int(trial)).zfill(3)
    
    figure_file_name_bar = f"label_sequence_{ppid}_S{session_str}_T{trial_str}.png"
    figure_full_filename_bar = os.path.join(parent_dir_bar, figure_file_name_bar)

    # Use the color map created outside of the loop
    # if label_description_column:
    #     labels_column = label_description_column
    
    labels = group[labels_column]   
  
    # Use a default color (grey) for any labels not in the color map
    colors = [color_map.get(label, 'grey') for label in labels]

    # Create a color map for the current group
    group_labels = np.unique(group[labels_column])
    group_color_map = {
        label: color_map.get(label, 'grey')
        for label in group_labels
    }

    # Create a custom legend
    patches = [
        plt.Line2D([0], [0], color=color, lw=4, label=label)
        for label, color in group_color_map.items()
    ]

    # Update legend arguments for this group
    group_legend_args = legend_args.copy()
    group_legend_args["handles"] = patches
    group_legend_args["ncol"] = 1
    group_legend_args["labelspacing"] = 0.2
    group_legend_args["borderpad"] = 0.4
    group_legend_args["bbox_to_anchor"] = (1, 1)
    group_legend_args["prop"] = {"size": 8}
    group_legend_args["fontsize"] = 5

    # Plot the label sequence
    plot_label_sequence(
        group["time"],
        group[labels_column],
        colors,
        color_map=group_color_map,
        legend_args=group_legend_args,
    )
    
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

@tictoc
def plot_data(
    data: pd.DataFrame,
    filenames: dict,
    output_path: str,
    args: argparse.Namespace,
    show: bool = False,
    block: bool = True,
    save_figures: bool = True,
    keep_figures_open: bool = False,
    labels_scatter: bool = False,
    labels_column: int = None,
    label_description_column: int = None,
    plot_all_data_figures: bool = True,
) -> None:
    """Plot syllable sequence data.
    
    Args:
        data: DataFrame containing the data
        filenames: Dictionary of output file paths
        output_path: Path to save output
        args: Command line arguments
        show: Whether to display plots
        block: Whether to block execution while showing plots
        save_figures: Whether to save figures
        keep_figures_open: Whether to keep figures open
        labels_scatter: Whether to create scatter plots
        labels_column: Column containing labels
        plot_all_data_figures: Whether to plot all figures
    """
    CMAP_COLORMAP = create_pastel_rainbow_cmap()
    SCATTER_MARKER_SIZE = 9
    DPI = 300
    
    if "Time" in data.columns:
        data = data.rename(columns={"Time": "time"})

    data[args.labels_column] = data[args.labels_column].astype(str)

    if args.label_description_data:
        print_color(f"\nLoading label description data from {args.label_description_data}...\n", bold_color.BLUE)
        label_description = pd.read_csv(args.label_description_data, dtype=str)
        label_description_dictionary = dict(zip(label_description['label'], label_description[args.label_description_column]))
        label_description_dictionary["-1"] = "-1"
    else:
        label_description_dictionary = None

    if args.label_description_column and not args.label_description_data:
        labels_column = args.label_description_column
    elif args.label_description_data and args.label_description_column:
        data_mapped = data.copy()
        data_mapped['label_description'] = data[args.labels_column].map(label_description_dictionary)

        if args.save_mapped_data:
            data_mapped.to_csv(os.path.join(output_path, "mapped_data_" + args.label_description_column + ".csv"), index=False)

        labels_column = 'label_description' 
        data_raw = data_mapped.copy()
        data = data_mapped.copy()

    else:
        data_raw = data.copy()

    le, group_keys, grouped = group_and_encode(
        data_raw, 
        label_column=labels_column
    )
    
    # Load color map
    if args.colour_map_object:
        with open(args.colour_map_object, "rb") as f:
            colour_map_object = pickle.load(f)
    else:
        colour_map_object = None

    
    color_map = create_color_map(le, colour_map_arg=colour_map_object, label_description_dictionary=label_description_dictionary)

    # Save LabelEncoder object
    label_encoder_path = os.path.join(output_path, "label_encoder.pkl")
    with open(label_encoder_path, "wb") as le_file:
        pickle.dump(le, le_file)
    
    # Save color map object
    colour_map_path = os.path.join(output_path, "colour_map.pkl")
    with open(colour_map_path, "wb") as cm_file:
        pickle.dump(color_map, cm_file)

    parent_dir = output_path
    parent_dir_bar = create_directory(
        os.path.join(parent_dir, "sequence_plots")
    )

    legend_args = {
        "handles": "",
        "bbox_to_anchor": (1, 1),
        "loc": "upper left",
        "handlelength": 0.1,
        "fontsize": 5,
        "frameon": False,
        "borderpad": 0.1,
        "labelspacing": 0.03,
        "prop": {"size": 8},
    }

    embedding = data.iloc[:, 4:]

    if labels_column:
        labels = data.loc[:, labels_column]
    else:
        labels = data.iloc[:, -1]

    x_seconds = pd.DataFrame(embedding).index / args.data_collection_rate
    colors = [color_map.get(label) for label in labels]

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

        ticks = np.arange(min(x_seconds), max(x_seconds), step=1000)
        plt.xticks(ticks, labels=[f"{int(tick)}s" for tick in ticks], fontsize=8)
        plt.title("Clusters | All", fontweight="bold")
        plt.xlabel("time", fontweight="bold")
        plt.ylabel("label", fontweight="bold")


        patches = [
            plt.Line2D([0], [0], color=color, lw=4, label=label)
            for label, color in color_map.items()
        ]
        legend_args["handles"] = patches
        legend_args["ncol"] = 2

        fig_embedding = plot_clusters(
            embedding=embedding.values,
            labels=labels.values,
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
        
    # Prepare arguments for parallel processing
    grouped_df = data.groupby(["ppid", "session", "trial"])
    group_args = [
        (name, group, parent_dir_bar, labels_column, label_description_column, color_map, legend_args, show, block, DPI)
        for name, group in grouped_df
    ]

    # Process groups in parallel using joblib
    n_jobs = os.cpu_count() or 1  # Use all available CPU cores
    print_color(f"\nProcessing {len(group_args)} groups using {n_jobs} CPU cores...\n", bold_color.BLUE)
    
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_single_group)(args_tuple)
        for args_tuple in tqdm(group_args, desc="Processing groups", unit="group")
    )

    print_color(f"\nSub cluster figures saved successfully.\n", bold_color.GREEN)

    return label_encoder_path, colour_map_path

def parse_arguments():
    """Parse command line arguments for syllable sequence plotting.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Plot trial clusters.")
    
    # Required arguments
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to the CSV file with the features or embedding with labels."
    )
    parser.add_argument(
        "--labels-column",
        type=str,
        required=True,
        help="Name of the column with labels."
    )
    parser.add_argument(
        "--colour-map-object",
        type=str,
        default=None,
        help="Path to the colour map object or None."
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--data-collection-rate",
        type=int,
        default=100,
        help="Data collection rate in Hz."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to the output directory."
    )
    parser.add_argument(
        "--label-description-data",
        type=str,
        default=None,
        help="Path to the csv with labels and descriptions."
    )
    parser.add_argument(
        "--label-description-column",
        type=str,
        default=None,
        help="Name of the column with label descriptions."
    )
    parser.add_argument(
        "--save-mapped-data",
        type=bool,
        default=True,
        help="Whether to save the mapped data."
    )
    
    args = parser.parse_args()

    return args


@tictoc
def main():
    """Main function to handle syllable sequence plotting."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up output directory
    if args.output_path:
        data_parent = args.output_path
    else:
        data_parent = Path(args.data_file).parent
        if "syllable_sequence" not in str(data_parent):
            data_parent = os.path.join(data_parent, "syllable_sequence")

    os.makedirs(data_parent, exist_ok=True)

    # Load and process data
    data = pd.read_csv(args.data_file)
    full_file_names = get_full_file_names(data_parent)

    plot_data(
        data=data,
        filenames=full_file_names,
        output_path=data_parent,
        args=args,
        block=False,
        labels_column=args.labels_column,
        label_description_column=args.label_description_column,
        plot_all_data_figures=False,
    )

    print_color("\nSyllable sequences completed successfully!".capitalize(), bold_color.BLUE)
    print_color(f"\nThe data can be found in: ".capitalize() + f"{data_parent}", bold_color.BLUE)


if __name__ == "__main__":
    print_welcome_message()
    main()