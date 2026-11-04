"""Module for running the full prediction pipeline."""

# ==
# author: uSerratos
# date: Nov 2024
# ==

import os
import sys
import subprocess
import shutil
import argparse
from typing import Union

from utils import *

# === GENERAL CLUSTERING PARAMETERS === #
PCA_THRESHOLD = 0.95  # 0.95

# === UMAP ===#
NEIGHBORS = 300  # 300
MIN_DIST = 0.0  # 0.001
DENSMAP = False  # True

# === HDBSCAN === #
CLUSTER_SIZE = 300  # best 300; then: 500, 700; old: 500
SAMPLES = None  # best None; then: None, None; old:30
EPSILON = 0.0  # best 0.0, then: 0.0, 0.0 ; old: 1.0

def fetch_args():
    """Parse and return command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Run the full prediction pipeline.")
    
    # Quality thresholds
    parser.add_argument(
        "--marker-quality-threshold", 
        type=int, 
        default=95, 
        help="Marker quality threshold."
    )
    parser.add_argument(
        "--marker-quality-sub-threshold", 
        type=int, 
        default=95, 
        help="Marker quality sub threshold."
    )
    parser.add_argument(
        "--segment-length-tolerance", 
        type=int, 
        default=10, 
        help="Segment length tolerance."
    )
    parser.add_argument(
        "--mean-segment-length-deviation-threshold", 
        type=int, 
        default=95, 
        help="Mean segment length deviation threshold."
    )
    parser.add_argument(
        "--individual-segment-length-deviation-threshold", 
        type=int, 
        default=95, 
        help="Individual segment length deviation threshold."
    )
    
    # Configuration parameters
    parser.add_argument(
        "--stick-figure-template", 
        type=str, 
        default="qualisys_csv", 
        help="Stick figure template."
    )
    parser.add_argument(
        "--data-collection-rate", 
        type=int, 
        required=True, 
        help="Data collection rate."
    )
    parser.add_argument(
        "--plot-sample-rate", 
        type=int, 
        default=3, 
        help="Plot sample rate."
    )
    
    # Path arguments
    parser.add_argument(
        "--classifier-path", 
        type=str, 
        help="Path to the classifier file."
    )
    parser.add_argument(
        "--colour-map-path", 
        type=str, 
        help="Path to the colour map file."
    )
    parser.add_argument(
        "--label-encoder-path", 
        type=str, 
        help="Path to the label encoder file."
    )
    parser.add_argument(
        "--vr-metadata-path", 
        type=str, 
        required=True,
        help="Path to the VR metadata file."
    )
    parser.add_argument(
        "--raw-data-path",
        required=True, 
        type=str, 
        help="Path to the raw data directory."
    )
    parser.add_argument(
        "--ffmpeg-executable-path",
        required=True, 
        type=str, 
        help="Path to the ffmpeg executable."
    )
    
    # Control flags
    parser.add_argument(
        "--remove-output-dirs", 
        action="store_true", 
        help="Remove the output directories."
    )
    parser.add_argument(
        "--skip-mocap-clean", 
        action="store_true", 
        help="Skip the mocap clean step."
    )
    parser.add_argument(
        "--skip-feature-matrix", 
        action="store_true", 
        help="Skip the feature matrix step."
    )
    parser.add_argument(
        "--skip-get-clusters", 
        action="store_true", 
        help="Skip the get clusters step."
    )
    parser.add_argument(
        "--skip-prediction", 
        action="store_true", 
        help="Skip the prediction step."
    )
    parser.add_argument(
        "--skip-syllable-sequence", 
        action="store_true", 
        help="Skip the syllable sequence step."
    )
    parser.add_argument(
        "--skip-syllable-stickfigures", 
        action="store_true", 
        help="Skip the syllable stickfigures step."
    )
    parser.add_argument(
        "--skip-syllable-counts", 
        action="store_true", 
        help="Skip the syllable counts step."
    )
    
    # get_clusters.py arguments
    parser.add_argument(
        "--rapids-cuml", 
        action='store_true', 
        help="Use Rapids cuML for PCA, UMAP, and HDBSCAN."
    )
    parser.add_argument(
        "--pca-threshold", 
        type=float, 
        default=PCA_THRESHOLD, 
        help="PCA threshold."
    )
    parser.add_argument(
        "--pca-n-components", 
        type=int, 
        default=None, 
        help="PCA n components."
    )
    parser.add_argument(
        "--umap-embedding", 
        type=str, 
        default=None, 
        help="UMAP embedding."
    )
    parser.add_argument(
        "--hdbscan-cluster-size", 
        type=int, 
        default=CLUSTER_SIZE, 
        help="HDBSCAN cluster size."
    )
    parser.add_argument(
        "--hdbscan-cluster-samples", 
        type=Union[int, None], 
        default=SAMPLES, 
        help="HDBSCAN cluster samples."
    )
    parser.add_argument(
        "--hdbscan-epsilon", 
        type=float, 
        default=EPSILON, 
        help="HDBSCAN epsilon."
    )
    parser.add_argument(
        "--hdbscan-not-single-cluster", 
        action='store_false', 
        help="HDBSCAN not single cluster."
    )
    parser.add_argument(
        "--umap-n-components", 
        type=int, 
        default=None, 
        help="UMAP n components."
    )
    parser.add_argument(
        "--umap-n-neighbors", 
        type=int, 
        default=NEIGHBORS, 
        help="UMAP n neighbors."
    )
    parser.add_argument(
        "--umap-min-dist", 
        type=float, 
        default=MIN_DIST, 
        help="UMAP min dist."
    )
    parser.add_argument(
        "--subsample-data-rate", 
        type=int, 
        default=None, 
        help="Subsample data rate."
    )

    # Curated list
    parser.add_argument(
        "--curated-list", 
        type=str, 
        default=None, 
        help="Curated list as a csv file. It expects the columns to be ppid, session, trial."
    )
    
    # Shutdown
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shutdown the computer after the pipeline is complete."
    )

    #stickfigure 
    parser.add_argument(
        "--stickfigure-group-mode",
        type=str,
        default="ppid",
        help="Group mode for the stickfigures. Options are ppid, label."
    )
    

    args = parser.parse_args()

    # Ensure that only one of the flags can be False
    if not args.skip_get_clusters and not args.skip_prediction:
        parser.error("Only one of --skip-get-clusters or --skip-prediction can be False.")

    return args


def run_script(script_name, args):
    """Run a Python script with the given arguments."""
    command = ["python", script_name] + args
    
    # Set environment variable to indicate running as subprocess
    env = os.environ.copy()
    env['RUNNING_AS_SUBPROCESS'] = 'true'
    
    process = subprocess.Popen(command, env=env)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def remove_directories(directories):
    """Remove specified directories if they exist.
    
    Args:
        directories (list): List of directory paths to remove
    """
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)


def get_default_paths(args):
    """Get default paths for the pipeline, allowing command line arguments to override them.
    
    Args:
        args: Parsed command line arguments
        raw_data_path: Optional path to raw data directory to override default
        
    Returns:
        dict: Dictionary containing all paths and derived paths for the pipeline
    """
    # Base default paths
    default_paths = {
        'classifier': r"D:\main_data\Study5\classifier_model\random_forest_classifier.pkl",
        'colour_map': r"D:\main_data\Study5\classifier_model\colour_map.pkl",
        'label_encoder': r"D:\main_data\Study5\classifier_model\label_encoder.pkl",
        'vr_metadata': r"D:\main_data\Study5\raw\study5_sub\trials_raw.csv",
        'raw_data': r"D:\main_data\Study5\raw\study5_sub",
        'ffmpeg_executable': r"D:\Software\ffmpeg\bin"
    }
    
    # Override defaults with command line arguments if provided
    if args.raw_data_path:
        default_paths['raw_data'] = args.raw_data_path
    raw_data_parent_path = os.path.dirname(default_paths['raw_data'])

    if args.classifier_path:
        default_paths['classifier'] = args.classifier_path
    else:
        default_paths['classifier'] = os.path.join(
            raw_data_parent_path, 
            "classifier_model", 
            "random_forest_classifier.pkl"
        )

    if args.colour_map_path:
        default_paths['colour_map'] = args.colour_map_path
    else:
        default_paths['colour_map'] = None

    if args.label_encoder_path:
        default_paths['label_encoder'] = args.label_encoder_path
    else:
        default_paths['label_encoder'] = None

    if args.vr_metadata_path:
        default_paths['vr_metadata'] = args.vr_metadata_path
    else:
        default_paths['vr_metadata'] = os.path.join(
            raw_data_parent_path,
            "trials_raw.csv"
        )
    if args.ffmpeg_executable_path:
        default_paths['ffmpeg_executable'] = args.ffmpeg_executable_path
    
    # Derive additional paths
    main_output_path = os.path.join(raw_data_parent_path, "syllable_analysis_output")
    
    # Add derived paths
    derived_paths = {
        'main_output': main_output_path,
        'feature_matrix': os.path.join(main_output_path, "feature_matrix"),
        'predicted_labels': os.path.join(
            main_output_path, 
            "predicted_labels" if not args.skip_prediction else "hdbscan_clusters", 
            "embedding_with_labels_light.csv"
        ),
        'syllable_sequence': os.path.join(main_output_path, "syllable_sequence"),
        'video_output': os.path.join(main_output_path, "syllable_stickfigures"),
        'coordinates': os.path.join(
            main_output_path, 
            f"mocap_clean_{args.marker_quality_threshold}_"
            f"{args.marker_quality_sub_threshold}_"
            f"{args.segment_length_tolerance}_"
            f"{args.mean_segment_length_deviation_threshold}_"
            f"{args.individual_segment_length_deviation_threshold}"
        ),
        'hdbscan_clusters': os.path.join(main_output_path, "hdbscan_clusters"),
    }
    
    return {**default_paths, **derived_paths}


def get_directories_to_remove(paths):
    """Get list of directories that should be removed before running the pipeline.
    
    Args:
        paths (dict): Dictionary containing all pipeline paths
        
    Returns:
        list: List of directory paths to remove
    """
    return [
        paths['coordinates'],
        paths['feature_matrix'],
        paths['syllable_sequence'],
        paths['video_output'],
        os.path.dirname(paths['predicted_labels']),
        os.path.join(paths['main_output'], "predicted_labels"),
        os.path.join(paths['main_output'], "syllable_counts")
    ]


def clean_output_directories(paths, remove=False):
    """Remove output directories if they exist and removal is requested.
    
    Args:
        paths (dict): Dictionary containing all pipeline paths
        remove (bool): Whether to remove the directories
    """
    if remove:
        dirs_to_remove = get_directories_to_remove(paths)
        remove_directories(dirs_to_remove)


def create_script_arguments(paths, args):
    """Create argument lists for each script in the pipeline."""
    DOWN_SAMPLED_DATA_COLLECTION_RATE = 100

    script_args = {
        'mocap_clean': [
            "-o", "-ow", "-sm", "-d", paths['raw_data'],
            "-f", paths['ffmpeg_executable'],
            "-sr", str(args.data_collection_rate), "-r",
            "--marker-quality-threshold", str(args.marker_quality_threshold),
            "--marker-quality-sub-threshold", str(args.marker_quality_sub_threshold),
            "--segment-length-tolerance", str(args.segment_length_tolerance),
            "--mean-segment-length-deviation-threshold",
            str(args.mean_segment_length_deviation_threshold),
            "--individual-segment-length-deviation-threshold",
            str(args.individual_segment_length_deviation_threshold),
            "--stick-figure-template", args.stick_figure_template
        ],
        
        'feature_matrix': [
            "-d", paths['coordinates'],
            "-sr", str(DOWN_SAMPLED_DATA_COLLECTION_RATE),
            "-od", paths['main_output'],
            "-sm", args.stick_figure_template,
            "-c", args.curated_list
        ],
        
        'predict': [
            "--feature-matrix-file", paths['feature_matrix'],
            "--classifier-file", paths['classifier'],
            "--labels-encoder-file", paths['label_encoder'],
            "--all-unknown"
        ],
        
        'syllable_sequence': [
            "--data-file", paths['predicted_labels'],
            "--labels-column", "label",
            "--data-collection-rate", str(DOWN_SAMPLED_DATA_COLLECTION_RATE) if not args.subsample_data_rate else str(DOWN_SAMPLED_DATA_COLLECTION_RATE // args.subsample_data_rate), 
            *(["--colour-map-object", paths['main_output']] if paths['colour_map'] is not None else []),
            "--output-path", paths['main_output']
        ],
        
        'stickfigures': [
            "--metadata-csv", paths['predicted_labels'],
            "--label-column", "label",
            "--coordinates-path", paths['coordinates'],
            "--ffmpeg-executable-path", paths['ffmpeg_executable'],
            "--video-output-path", paths['video_output'],
            "--data-sample-rate", str(DOWN_SAMPLED_DATA_COLLECTION_RATE) if not args.subsample_data_rate else str(DOWN_SAMPLED_DATA_COLLECTION_RATE // args.subsample_data_rate),
            "--stickfigure-template", args.stick_figure_template,
            "--save-as-video", "--provide-colour-map",
            "--colour-map-object", paths['colour_map'] if paths['colour_map'] is not None else os.path.join(paths['main_output'], "colour_map.pkl"),
            "--plot-sample-rate", str(args.plot_sample_rate),
            "--group-mode", args.stickfigure_group_mode
        ],
        
        'syllable_counts': [
            "--labels-path", paths['predicted_labels'],
            "--label-column", "label",
            "--vr-metadata-path", paths['vr_metadata'],
            "--output-dir", paths['syllable_sequence']
        ],
        'get_clusters': [
            "--path", paths['feature_matrix'],
            "--data-collection-rate", str(DOWN_SAMPLED_DATA_COLLECTION_RATE) if not args.subsample_data_rate else str(DOWN_SAMPLED_DATA_COLLECTION_RATE // args.subsample_data_rate),
            "--output-path", paths['hdbscan_clusters'],
            "--curated-list", args.curated_list,
            "--umap-n-neighbors", str(args.umap_n_neighbors) if not args.subsample_data_rate else str((args.umap_n_neighbors // args.subsample_data_rate) * 3),
            "--hdbscan-cluster-size", str(args.hdbscan_cluster_size) if not args.subsample_data_rate else str((args.hdbscan_cluster_size // args.subsample_data_rate) * 3),   
            # "--hdbscan-cluster-samples", str(args.hdbscan_cluster_samples) if args.hdbscan_cluster_samples is not None else 'None',
            "--hdbscan-epsilon", str(args.hdbscan_epsilon),
            # Only add the flag if args.hdbscan_not_single_cluster is True
            *(["--hdbscan-not-single-cluster"] if args.hdbscan_not_single_cluster else []),
        ]
    }

    # Conditionally add the --rapids-cuml flag
    if args.rapids_cuml:
        script_args['get_clusters'].append("--rapids-cuml")
    
    # if not args.hdbscan_not_single_cluster:
    #     script_args['get_clusters'].append("--hdbscan-not-single-cluster")
    
    if args.pca_n_components:
        script_args['get_clusters'].extend(["--pca-n-components", str(args.pca_n_components)])
    
    if args.umap_embedding:
        script_args['get_clusters'].extend(["--umap-embedding", args.umap_embedding])
    
    if args.pca_threshold:
        script_args['get_clusters'].extend(["--pca-threshold", str(args.pca_threshold)])
    
    if args.umap_n_components:
        script_args['get_clusters'].extend(["--umap-n-components", str(args.umap_n_components)])
    
    if args.subsample_data_rate:
        script_args['get_clusters'].extend(["--subsample-data-rate", str(args.subsample_data_rate)])
    
    return script_args

@tictoc
def main():
    """Main function to run the full prediction pipeline."""

    # Fetch and parse command line arguments
    args = fetch_args()
    
    # Get all paths
    paths = get_default_paths(args)


    if not args.skip_get_clusters:
        print_color("\nRunning get_clusters.py".capitalize(), bold_color.GREEN)
        print_color("This step may take a while depending on the size of the data.".capitalize(), bold_color.GREEN)
        paths['predicted_labels'] = os.path.join(paths['main_output'], "hdbscan_clusters", "embedding_with_labels_light.csv")
        paths['colour_map'] = os.path.join(paths['main_output'] , "hdbscan_clusters", "colour_map.pkl")
        paths['label_encoder'] = os.path.join(paths['main_output'], "hdbscan_clusters", "label_encoder.pkl")
    elif not args.skip_prediction:
        print_color("\nRunning predict_where_unknown.py".capitalize(), bold_color.GREEN)
        print_color("This step may take a while depending on the size of the data.".capitalize(), bold_color.GREEN)
    

    # Get all script arguments
    script_args = create_script_arguments(paths, args)
    
    # Handle directory removal if requested
    clean_output_directories(paths, args.remove_output_dirs)

    # Run scripts based on skip flags
    if not args.skip_mocap_clean:
        try:    
            run_script("do_mocap_clean.py", script_args['mocap_clean'])
        except Exception as e:
            print_color(f"Error running mocap_clean.py: {e}", bold_color.RED)
    if not args.skip_feature_matrix:
        try:
            run_script("get_feature_matrix.py", script_args['feature_matrix'])
        except Exception as e:
            print_color(f"Error running get_feature_matrix.py: {e}", bold_color.RED)
    if not args.skip_get_clusters:
        try:
            run_script("get_clusters.py", script_args['get_clusters'])
        except Exception as e:
            print_color(f"Error running get_clusters.py: {e}", bold_color.RED)
    if not args.skip_prediction:
        try:
            run_script("predict_where_unknown.py", script_args['predict'])
        except Exception as e:
            print_color(f"Error running predict_where_unknown.py: {e}", bold_color.RED)
    if not args.skip_syllable_sequence:
        try:
            run_script("plot_syllable_sequences.py", script_args['syllable_sequence'])
        except Exception as e:
            print_color(f"Error running plot_syllable_sequences.py: {e}", bold_color.RED)
    if not args.skip_syllable_stickfigures:
        try:
            run_script("plot_syllable_stickfigures.py", script_args['stickfigures'])
        except Exception as e:
            print_color(f"Error running plot_syllable_stickfigures.py: {e}", bold_color.RED)
    if not args.skip_syllable_counts:
        try:
            run_script("get_syllable_counts.py", script_args['syllable_counts'])
        except Exception as e:
            print_color(f"Error running get_syllable_counts.py: {e}", bold_color.RED)

    print_color("\nPipeline completed successfully!".capitalize(), bold_color.BLUE)
    print_color(f"\nThe data can be found in: ".capitalize() + f"{paths['main_output']}", bold_color.BLUE)


if __name__ == "__main__":
    os.system("cls||clear")
    print_welcome_message()
    main()
    args = fetch_args()
    if args.shutdown:
        os.system("shutdown /h")
