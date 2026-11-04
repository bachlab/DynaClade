"""Module for creating and plotting stick figure visualizations of motion data."""

# ==
# author: uSerratos
# date: Nov 2024
# ==

import os
import sys
import argparse
import glob
import pickle
import gc  # Add gc import
from pathlib import Path
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
# Import utils functions first
from utils import (
    print_welcome_message,
    StickFigure,
    skf,
    plot_stickfigure,
    ask_question_options,
    tictoc,
    create_pastel_rainbow_cmap,
    create_directory,
    print_color,
    bold_color,
    list_directory_items,
)

# Clear screen only if not running as subprocess
if os.environ.get('RUNNING_AS_SUBPROCESS') != 'true':
    os.system("cls||clear")


def load_data(args: argparse.Namespace) -> tuple:
    """Load and prepare data for stick figure plotting.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: Loaded data and configuration parameters
    """
    # Initialize variables
    colour_map_object = None
    
    # Create output directory
    video_output_path = create_directory(
        Path(os.path.join(
            os.path.normpath(args.video_output_path)
        ))
    )
    
    # Load metadata
    metadata_csv = pd.read_csv(args.metadata_csv)
    
    # Map metadata columns to standard names if needed
    metadata_csv = map_metadata_columns(metadata_csv)
    
    # Load color map if provided
    if args.provide_colour_map:
        colour_map_object = args.colour_map_object

    if args.sample_data_rate:
        data_sample_rate = args.sample_data_rate
    else:
        data_sample_rate = args.data_sample_rate

    return (
        metadata_csv,
        args.label_column,
        video_output_path,
        args.ffmpeg_executable_path,
        args.coordinates_path,
        args.file_pattern,
        data_sample_rate,
        args.plot_sample_rate,
        args.plot_sequence,
        args.save_as_video,
        args.save_frames,
        args.stickfigure_template,
        colour_map_object,
        args.group_mode,
    )


def group_and_encode(df: pd.DataFrame, label_column: str) -> tuple:
    """Group data by trial info and encode labels.
    
    Args:
        df: DataFrame containing the data
        label_column: Name of the column containing labels
        
    Returns:
        tuple: (LabelEncoder, list of group keys, grouped DataFrame)
    """

    # Check for any non-numeric values in metadata columns
    meta_columns = ["ppid", "session", "trial"]
    for col in meta_columns:
        non_numeric = df[~df[col].astype(str).str.match(r'^\d+\.?\d*$')][col]
        if not non_numeric.empty:
            print(f"\nNon-numeric values found in {col}:")
            print(non_numeric)
    
    try:
        # Only convert metadata columns to integers
        df[meta_columns] = df[meta_columns].astype(int)
    except ValueError as e:
        print("\nError during integer conversion:")
        print(str(e))
        raise
        
    grouped = df.groupby(meta_columns)
    group_keys = list(grouped.groups.keys())
    le = LabelEncoder()
    
    # Handle case when dummy labels are used or no labels are provided
    if label_column == 'dummy_label' or label_column not in df.columns:
        # For dummy labels or no labels, we don't need to fit the encoder since all values are the same
        le.classes_ = np.array(['motion_data'])
    else:
        le.fit(df[label_column])

    return le, group_keys, grouped


def create_color_map(le: LabelEncoder, colour_map_arg=None, label_description_dictionary=None) -> dict:
    """Create a color map for labels.
    
    Args:
        le: Fitted LabelEncoder object
        colour_map_arg: Optional colormap name or dictionary
        label_description_dictionary: Optional dictionary with label descriptions

    Returns:
        dict: Mapping of labels to colors
    """
    # Handle case when only dummy labels are present (no real labels provided)
    if len(le.classes_) == 1 and 'motion_data' in le.classes_:
        # Return a simple color map for motion data without labels
        return {'motion_data': 'blue'}
    
    if isinstance(colour_map_arg, dict):
        if label_description_dictionary:
            colour_map_arg = {label_description_dictionary.get(label): colour for label, colour in colour_map_arg.items()}
        return colour_map_arg

    if colour_map_arg:
        color_map = plt.colormaps.get_cmap(colour_map_arg)
    else:
        color_map = create_pastel_rainbow_cmap()

    # Normalize label indices to the range [0, 1]
    normalized_labels = le.transform(le.classes_) / (len(le.classes_) - 1)
    
    # Get colors using the colormap
    cmap = [color_map(norm_label) for norm_label in normalized_labels]
    
    # Create label to color mapping
    if label_description_dictionary:
        color_map_dict = {label_description_dictionary.key(label): cmap[i] for i, label in enumerate(le.classes_)}
    else:
        color_map_dict = {label: cmap[i] for i, label in enumerate(le.classes_)}

    color_map_dict[-1] = "grey"  # Special case for unknown labels

    return color_map_dict


def recenter_data(data, stickfigure):
    meta_columns = ["ppid", "session", "trial"]

    def recenter_group(group):
        try:
            if group.empty:
                return group
                
            MHip_cols = stickfigure._markers_xyz['MHip']
            if not all(col in group.columns for col in MHip_cols):
                print(f"Missing MHip columns in group. Available columns: {group.columns.tolist()}")
                return group
                
            subtract_column_x = MHip_cols[0]
            x_columns = group.filter(regex='_x$').columns
            if not x_columns.empty:
                group[x_columns] = group[x_columns].subtract(group[subtract_column_x].iloc[0], axis=0)
                
            subtract_column_y = MHip_cols[1]
            y_columns = group.filter(regex='_y$').columns
            if not y_columns.empty:
                group[y_columns] = group[y_columns].subtract(group[subtract_column_y].iloc[0], axis=0)
                
        except Exception as e:
            print(f"Error in recenter_group: {str(e)}")
            print(f"Group shape: {group.shape}")
            print(f"Group columns: {group.columns.tolist()}")
        return group

    # Check if all meta columns exist in the data
    if all(col in data.columns for col in meta_columns):
        grouped = data.groupby(meta_columns)
        # data = grouped.apply(recenter_group)
        data = grouped.apply(recenter_group, include_groups=False).reset_index(drop=True)
    else:
        data = recenter_group(data)

    return data


def azimuth_calculation(data, stickfigure):
    # Compute the shoulder to shoulder angle with respect to global frame
    
    shoulder_to_shoulder_angles = []
    for _, row in data.iterrows():
        LSho = stickfigure._markers_xyz['LSho']
        RSho = stickfigure._markers_xyz['RSho']
        
        RSho_coords = np.array([row[col] for col in RSho])
        LSho_coords = np.array([row[col] for col in LSho])

        vector_lr = RSho_coords - LSho_coords
        vector_lr_xy = vector_lr[:2]
        
        angle = np.arctan2(vector_lr_xy[1], vector_lr_xy[0])
        
        angle_degrees = np.degrees(angle)
        shoulder_to_shoulder_angles.append(angle_degrees)
    
    median_shoulder_angle = np.median(shoulder_to_shoulder_angles)
    
    chest_perpendicular = median_shoulder_angle + 90
    azimuth = chest_perpendicular + 45

    if azimuth > 180:
        azimuth -= 360
    elif azimuth < -180:
        azimuth += 360

    return azimuth


def plot_stick_figure(
    df_xyz_labelled: pd.DataFrame,
    video_output_path: str,
    ffmpeg_executable_path: str,
    label_column: str,
    title: str = "Stick Figure",
    color_map: dict = None,
    data_sample_rate: int = 100,
    plot_sequence: bool = False,
    plot_frame_rate: int = 1,
    markersize: float = 3,
    axis_lim: tuple = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5),
    save_as_video: bool = True,
    save_frames: bool = True,
    stick_figure_template: str = "coda",
    video_name: str = "stickfigure",
    alpha: float = 1,
    adjust_axis: bool = False,
    elevation: int = 30,
    azimuth: int = None,
    linewidth: float = 3,
    prototypes_plot: bool = False,
    should_recenter_data: bool = False,
    should_calculate_azimuth: bool = False,
) -> None:
    """Plot stick figure visualization.
    
    Args:
        df_xyz_labelled: DataFrame with labeled coordinate data
        video_output_path: Path to save video output
        ffmpeg_executable_path: Path to ffmpeg executable
        label_column: Name of column containing labels
        title: Plot title
        color_map: Mapping of labels to colors
        data_sample_rate: Data sampling rate
        plot_sequence: Whether to plot sequence
        plot_frame_rate: Frame rate for plotting
        linewidth: Width of stick figure lines
        markersize: Size of markers
        axis_lim: Axis limits for plot
        save_as_video: Whether to save as video
        save_frames: Whether to save individual frames
        stick_figure_template: Template for stick figure
        video_name: Name for output video
        alpha: Transparency value
    """
    # Create stick figure object
    stickfigure = StickFigure(
        name=skf[stick_figure_template]["name"],
        body=skf[stick_figure_template]["body"],
        data_mapping=skf[stick_figure_template]["data_mapping"],
    )
    
    # Process title and colors
    title = os.path.splitext(video_name)[0].replace("_", " ")
    colors = (
        [color_map[label] for label in df_xyz_labelled[label_column]]
        if color_map is not None else None
    )
    
    # Get plot annotations
    plot_annotation = df_xyz_labelled[label_column].to_list()

    if should_recenter_data:
        df_xyz_labelled = recenter_data(df_xyz_labelled, stickfigure)

    if should_calculate_azimuth:
        azimuth = azimuth_calculation(df_xyz_labelled, stickfigure)
    else:
        azimuth = 60

    # Create plot
    plot_stickfigure(
        data=df_xyz_labelled,
        stick_figure=stickfigure,
        axis_lim=axis_lim,
        plot_frame_rate=plot_frame_rate,
        markersize=markersize,
        color=colors,
        save_as_video=save_as_video,
        save_frames=save_frames,
        data_sample_rate=data_sample_rate,
        video_output_path=video_output_path,
        video_name=video_name,
        ffmpeg_executable_path=ffmpeg_executable_path,
        plot_sequence=plot_sequence,
        title=title,
        plot_annotation=plot_annotation,
        alpha=alpha,
        adjust_axis=adjust_axis,
        elevation=elevation,
        azimuth=azimuth,
        linewidth=linewidth,
        prototypes_plot=prototypes_plot,
    )


def find_csv_files(directory, pattern):
    return glob.glob(os.path.join(directory, "**", pattern), recursive=True)


def process_group(group_info, video_output_path, ffmpeg_executable_path, label_column, color_map, 
                 data_sample_rate, plot_sequence, plot_sample_rate, save_as_video, save_frames, 
                 stickfigure_template, group_mode, df_xyz_labelled, should_recenter_data, 
                 should_calculate_azimuth, axis_lim_tuple, linewidth=4, markersize=3):
    """Process a single group for stick figure plotting."""
    i = group_info
    if i == -1:
        return

    try:
        # DEBUG: Print the group key being searched for
        print_color(f"Looking for group: {i}", bold_color.YELLOW)
        # Recreate the group from the data
        if group_mode == "ppid":
            meta_columns = ["ppid", "session", "trial"]
            current_group = df_xyz_labelled[
                (df_xyz_labelled['ppid'] == i[0]) & 
                (df_xyz_labelled['session'] == i[1]) & 
                (df_xyz_labelled['trial'] == i[2])
            ].reset_index(drop=True)
            video_name = f"stickfigure_P{i[0]:03}_S{i[1]:03}_T{i[2]:03}"
        else:
            current_group = df_xyz_labelled[df_xyz_labelled[label_column] == i].reset_index(drop=True)
            video_name = f"stickfigure_label_{i}"

        # Check if group is empty
        if current_group.empty:
            print_color(f"Warning: Group {i} is empty, skipping", bold_color.YELLOW)
            return False

        plot_stick_figure(
            current_group,
            video_output_path,
            ffmpeg_executable_path,
            label_column=label_column,
            color_map=color_map,
            data_sample_rate=data_sample_rate,
            plot_sequence=plot_sequence,
            plot_frame_rate=plot_sample_rate,
            linewidth=linewidth,
            markersize=markersize,
            axis_lim=axis_lim_tuple,    
            save_as_video=save_as_video,
            save_frames=save_frames,
            stick_figure_template=stickfigure_template,
            video_name=video_name,
            should_recenter_data=should_recenter_data,
            should_calculate_azimuth=should_calculate_azimuth,
        )

        # Clear memory after plotting
        plt.close('all')
        gc.collect()

        return True
    except Exception as e:
        print_color(f"Error processing group {i}: {str(e)}", bold_color.RED)
        import traceback
        print_color(f"Traceback: {traceback.format_exc()}", bold_color.RED)
        # Ensure we clean up even if there's an error
        plt.close('all')
        gc.collect()
        return False


def get_groups_from_keys(dataframe: pd.DataFrame, keys: list, meta_columns: list) -> list:

    data_grouped = dataframe.groupby(meta_columns)
    
    data_to_keep = []
    for key, group in data_grouped:
        if key in keys:
            meta = int(key[0]), int(key[1]), int(key[2])
            meta_dataframe = pd.DataFrame([meta] * len(group))
            meta_dataframe.columns = meta_columns
            meta_dataframe = meta_dataframe.reset_index(drop=True)
            group = group.reset_index(drop=True)
            # group = pd.concat([meta_dataframe, group], axis=1)
            data_to_keep.append(group)
    
    output_dataframe = pd.concat(data_to_keep, axis=0)
    output_dataframe = output_dataframe.reset_index(drop=True)

    return output_dataframe


def map_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map metadata columns to standard names if they have different names."""
    
    # Define possible column name variations
    column_mapping = {}
    
    # Check for ppid variations
    ppid_variations = ['ppid', 'ppid_vr', 'participant', 'participant_id', 'subject', 'subject_id']
    for var in ppid_variations:
        if var in df.columns:
            column_mapping[var] = 'ppid'
            break
    
    # Check for session variations
    session_variations = ['session', 'session_vr', 'session_id', 'block']
    for var in session_variations:
        if var in df.columns:
            column_mapping[var] = 'session'
            break
    
    # Check for trial variations
    trial_variations = ['trial', 'trial_vr', 'trial_id', 'trial_num']
    for var in trial_variations:
        if var in df.columns:
            column_mapping[var] = 'trial'
            break
    
    # Rename columns if mapping was found
    if column_mapping:
        print_color(f"Mapping columns: {column_mapping}", bold_color.BLUE)
        df = df.rename(columns=column_mapping)
    
    return df


@tictoc
def main(args):
    
    (
        df,
        label_column,
        video_output_path,
        ffmpeg_executable_path,
        coordinates_path,
        file_pattern,
        data_sample_rate,
        plot_sample_rate,
        plot_sequence,
        save_as_video,
        save_frames,
        stickfigure_template,
        colour_map_object,
        group_mode,
    ) = load_data(args)

    stickfigure = StickFigure(name=skf[stickfigure_template]["name"],body=skf[stickfigure_template]["body"],data_mapping=skf[stickfigure_template]["data_mapping"])
    xyz_target_columns = natsorted([i for key, value in stickfigure._markers_xyz.items() for i in value])
    xyz_target_columns = ["time"] + xyz_target_columns

    # Handle case when no label column is provided
    if label_column is None:
        print_color("\nNo label column provided. Using metadata columns (ppid, session, trial) to find mocap data.", bold_color.BLUE)
        print_color(f"Available columns in metadata CSV: {df.columns.tolist()}", bold_color.BLUE)
        # Create a dummy label column for consistency
        df['dummy_label'] = 'motion_data'
        label_column = 'dummy_label'
        # Force group mode to ppid when no labels are provided
        if args.group_mode != "ppid":
            print_color("Warning: Group mode changed to 'ppid' since no label column was provided.", bold_color.YELLOW)
            args.group_mode = "ppid"

    if args.label_description_data:
        df_copy = df.copy()
        
        if args.label_description_data:
            label_description = pd.read_csv(args.label_description_data)
            label_description_dict = dict(zip(label_description["label"], label_description[args.label_description_column]))
            df_copy[args.label_description_column] = df_copy[args.label_description_column].map(label_description_dict)
        
        copy_target_columns = ['ppid', 'session', 'trial', 'time', args.label_description_column]
        df_copy = df_copy[copy_target_columns]
        df_copy[args.label_description_column] = df_copy[args.label_description_column].astype(str)
        df_copy['ppid'] = df_copy['ppid'].astype(int)
        df_copy['session'] = df_copy['session'].astype(int)
        df_copy['trial'] = df_copy['trial'].astype(int)

    # Handle target columns based on whether label column is provided
    if label_column == 'dummy_label':
        # When no labels are provided, we only need metadata columns
        target_columns = ['ppid', 'session', 'trial']
        print_color(f"Using target columns for no-label mode: {target_columns}", bold_color.BLUE)
    else:
        # When labels are provided, include time and label columns
        target_columns = ['ppid', 'session', 'trial', 'time', str(label_column)]
        print_color(f"Using target columns for label mode: {target_columns}", bold_color.BLUE)
    
    print_color(f"Attempting to select columns: {target_columns}", bold_color.BLUE)
    print_color(f"Available columns: {df.columns.tolist()}", bold_color.BLUE)
    
    df = df[target_columns]
    df['ppid'] = df['ppid'].astype(int)
    df['session'] = df['session'].astype(int)
    df['trial'] = df['trial'].astype(int)
    if label_column != 'dummy_label':
        df[str(label_column)] = df[str(label_column)].astype(str)  # Ensure label_column is treated as string

    le, group_keys, grouped = group_and_encode(df, label_column=label_column)
    if colour_map_object is not None:
        with open(colour_map_object, "rb") as f:
            colour_map_object = pickle.load(f)

    if args.label_description_data:
        print_color(f"\nLoading label description data from {args.label_description_data}...\n", bold_color.BLUE)
        label_description = pd.read_csv(args.label_description_data)
        label_description_dictionary = dict(zip(label_description['label'], label_description[args.label_description_column]))
    else:
        label_description_dictionary = None

    color_map = create_color_map(le, colour_map_arg=colour_map_object, label_description_dictionary=label_description_dictionary)
    xyz_files = list_directory_items(coordinates_path, recursive=True, search_pattern=["T", ".csv"]).get(
        "items"
    )
    
    matching_files = {}
    for i in group_keys:
        ppid = i[0]
        session = f"S{i[1]:03}"
        trial = f"T{i[2]:03}"
        
        # Find files that match the directory structure pattern anywhere in the path
        matching_files_ = [
            file for file in xyz_files
            if os.path.sep + str(ppid) + os.path.sep + session + os.path.sep in file and 
               trial in os.path.basename(file) and 
               file.endswith('.csv')
        ]
        
        if len(matching_files_) > 1:
            raise ValueError(f"Multiple matching files found for {ppid} {session} {trial}")
        
        # Use the tuple key directly like in label mode
        matching_files[i] = matching_files_
        

    
    print_color(f"Found {len(matching_files)} matching coordinate files", bold_color.BLUE)
    
    if not matching_files:
        raise ValueError(f"No matching coordinate files found. Please check:\n"
                        f"1. The coordinates_path exists: {coordinates_path}\n"
                        f"2. The directory structure matches expected pattern: participant/S###/T###.csv\n"
                        f"3. The group_keys match the directory structure: {group_keys[:5]}")

    meta_columns = ["ppid", "session", "trial"]
    dataframes = []
    print_color("Reading coordinate files...", bold_color.BLUE)
    
    for i, (key, file) in enumerate(matching_files.items()):
        if i % 100 == 0:  # Show progress every 100 files
            print_color(f"Processing file {i+1}/{len(matching_files)}", bold_color.BLUE)
            
        try:
            if len(file) != 1:
                continue
            file = os.path.normpath(file[0])
            df_matching = pd.read_csv(file)
            df_matching = df_matching[xyz_target_columns]

            if args.sample_data_rate:
                original_length = len(df_matching)
                sampled_indices = list(range(0, original_length, data_sample_rate))
                df_matching = df_matching.iloc[sampled_indices]
                df_matching = df_matching.reset_index(drop=True)
         
            # Parse the tuple key to get ppid, session, trial
            ppid, session, trial = key
            
            df_meta = pd.DataFrame([[ppid, session, trial]] * len(df_matching))
            df_meta.columns = meta_columns
            df_meta = df_meta.reset_index(drop=True)
            df_matching = pd.concat([df_meta, df_matching], axis=1)
            dataframes.append(df_matching)

        except Exception as e:
            print_color(f"Error reading file {file}: {str(e)}", bold_color.RED)

    print_color(f"Created {len(dataframes)} dataframes from coordinate files", bold_color.BLUE)
    
    if not dataframes:
        raise ValueError("No valid dataframes could be created from the coordinate files.")

    print_color("Concatenating dataframes...", bold_color.BLUE)
    df_xyz = pd.concat(dataframes)

    print_color("Processing column names...", bold_color.BLUE)
    df_xyz.columns = [col.split('_')[0] + '_' + col.split('_')[1].lower() if '_' in col else col for col in df_xyz.columns]

    df_xyz = df_xyz.reset_index(drop=True)

    # Handle the case when no label column was originally provided
    if label_column == 'dummy_label':
        # When no labels are provided, we don't need to merge with label data
        # Just add the dummy label column to the mocap data
        df_xyz[label_column] = 'motion_data'
        
        # Ensure consistent data types for metadata columns (same as label mode)
        for col in meta_columns:
            df_xyz.loc[:, col] = df_xyz[col].astype(int)
        
        df_xyz_labelled = df_xyz
        
        # Create groups directly from the mocap data
        if args.group_mode == "ppid":
            grouped = df_xyz_labelled.groupby(meta_columns)
        else:
            grouped = df_xyz_labelled.groupby([label_column])
    else:
        # Original logic for when labels are provided
        if args.label_description_data:
            df = df_copy.reset_index(drop=True)
            label_column = args.label_description_column
        else:
            df = df.reset_index(drop=True)
        
        labels_data = df[meta_columns + ["time"] + [label_column]].copy()  # Create a copy to avoid chained assignment
        # Ensure consistent data types for merge columns using .loc

        for col in meta_columns:
            df_xyz.loc[:, col] = df_xyz[col].astype(int)
            labels_data.loc[:, col] = labels_data[col].astype(int)
        

        # check groups consistency
        xyz_groups = [k for k, i in df_xyz.groupby(meta_columns)]
        labels_groups = [k for k, i in labels_data.groupby(meta_columns)]
        both_groups = natsorted(set(xyz_groups).intersection(set(labels_groups)))

        if len(xyz_groups) != len(both_groups) or len(labels_groups) != len(both_groups):
            print_color(f"The length of the groups are not consistent. Length of both: {len(both_groups)}, xyz_groups: {len(xyz_groups)}, labels_groups: {len(labels_groups)}", bold_color.RED)
            print_color(f"Only the groups that exist in both dataframes will be used.", bold_color.YELLOW)

        
        xyz_groups = get_groups_from_keys(df_xyz, both_groups, meta_columns)
        labels_groups = get_groups_from_keys(labels_data, both_groups, meta_columns)

        # Merge the two dataframes on meta_columns and time
        merged_df = pd.merge(df_xyz, labels_data, on=meta_columns + ["time"], how="inner")

        # Ensure the merged dataframe is not empty
        if merged_df.empty:
            raise ValueError("The merged dataframe is empty. Please check the data and the merge columns.")

        df_xyz_labelled = merged_df

        if args.group_mode == "ppid":
            grouped = df_xyz_labelled.groupby(meta_columns)
        elif args.group_mode == "label":
            grouped = df_xyz_labelled.groupby([label_column])

    import math
    
    if args.group_mode == "ppid":
        # Function to check if a tuple contains NaN
        def contains_nan(tup):
            for x in tup:
                if isinstance(x, (int, float)):
                    if math.isnan(x):
                        return True
                elif pd.isna(x):  # Handle pandas NA values
                    return True
            return False
        
        # Filter out tuples with NaN elements - keep as tuples like in label mode
        group_keys = [tuple(int(x) for x in key) for key in grouped.groups.keys() if not contains_nan(key)]
        print_color(f"Generated {len(group_keys)} groups for processing", bold_color.BLUE)
    else:
        group_keys = [key for key in grouped.groups.keys()]
        print_color(f"Generated {len(group_keys)} groups for processing", bold_color.BLUE)

    # Set up parallel processing
    if label_column == 'dummy_label':
        # For no-label mode, limit cores to prevent memory/drive issues
        num_cores = min(5, mp.cpu_count() - 1)
        print_color(f"\nUsing {num_cores} CPU cores for parallel processing (limited for no-label mode)", bold_color.BLUE)
    else:
        # For label mode, use more cores
        num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free for system
        print_color(f"\nUsing {num_cores} CPU cores for parallel processing", bold_color.BLUE)

    print_color(f"Total data rows: {len(df_xyz_labelled)}", bold_color.BLUE)

    # Create partial function with fixed arguments
    process_group_partial = partial(
        process_group,
        video_output_path=video_output_path,
        ffmpeg_executable_path=ffmpeg_executable_path,
        label_column=label_column,
        color_map=color_map,
        data_sample_rate=data_sample_rate,
        plot_sequence=plot_sequence,
        plot_sample_rate=plot_sample_rate,
        save_as_video=save_as_video,
        save_frames=save_frames,
        stickfigure_template=stickfigure_template,
        group_mode=group_mode,
        df_xyz_labelled=df_xyz_labelled,
        should_recenter_data=args.recenter_data,
        should_calculate_azimuth=args.azimuth_calculation,
        axis_lim_tuple=tuple(args.axis_lim_tuple),
        linewidth=args.linewidth,
        markersize=args.markersize,
    )

    # Process groups in parallel with careful memory management
    print_color(f"\nProcessing {len(group_keys)} groups in parallel...", bold_color.BLUE)
    results = []
    
    # Process all groups in parallel
    print_color(f"Starting parallel processing of {len(group_keys)} groups...", bold_color.BLUE)
    with mp.Pool(num_cores) as pool:
        for result in tqdm(
            pool.imap_unordered(process_group_partial, group_keys, chunksize=1),
            total=len(group_keys),
            desc="Processing groups",
            unit="group"
        ):
            results.append(result)
            # Explicitly clear memory after each result is processed
            gc.collect()

    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    
    print_color(f"\nProcessed {len(results)} groups: {successful} successful, {failed} failed", bold_color.BLUE)
    print_color("\nStick figure videos completed successfully!", bold_color.BLUE)
    print_color(f"\nThe data can be found in: {video_output_path}", bold_color.BLUE)

def parse_arguments():
    """Parse command line arguments for stick figure plotting."""
    parser = argparse.ArgumentParser(description="Process stick figure videos.")
    
    # Required arguments
    parser.add_argument(
        "--metadata-csv",
        type=str,
        required=True,
        help="Path to the CSV file with metadata and labels"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        required=False,
        default=None,
        help="Name of the column with the labels. If not provided, will use metadata columns (ppid, session, trial) to find mocap data."
    )
    parser.add_argument(
        "--coordinates-path",
        type=str,
        required=True,
        help="Parent path to study data with marker coordinates CSV files"
    )
    parser.add_argument(
        "--ffmpeg-executable-path",
        type=str,
        default="./ffmpeg.exe",
        help="Path to the ffmpeg executable"
    )
    parser.add_argument(
        "--video-output-path",
        type=str,
        required=True,
        help="Path to the video output directory"
    )
    parser.add_argument(
        "--data-sample-rate",
        type=int,
        required=True,
        help="Data sample rate"
    )
    parser.add_argument(
        "--stickfigure-template",
        type=str,
        choices=[
            "coda",
            "openpose",
            "qualisys_csv",
            "qualisys_tsv",
            "template"
        ],
        required=True,
        help="Which stickfigure should be used"
    )
    
    # Optional arguments
    parser.add_argument(
        "--plot-sample-rate",
        type=int,
        default=10,
        help="Plot sample rate"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        required=False,
        default=None,
        help="File pattern to search for"
    )
    parser.add_argument(
        "--group-mode",
        type=str,
        choices=["ppid", "label"],
        required=False,
        default="ppid",
        help="How do you want to group the data"
    )
    parser.add_argument(
        "--sample-data-rate",
        type=int,
        required=False,
        default=None,
        help="Data sample rate"
    )
    parser.add_argument(
        "--axis-lim-tuple",
        type=float,
        nargs=6,
        required=False,
        default=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        help="Axis limits (six floats as positive numbers: x_min x_max y_min y_max z_min z_max)"
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        required=False,
        default=4,
        help="Line width"
    )
    parser.add_argument(
        "--markersize",
        type=float,
        required=False,
        default=3,
        help="Marker size"
    )
    
    # Boolean flags
    parser.add_argument(
        "--plot-sequence",
        action='store_true',
        help="Plot the sequence"
    )
    parser.add_argument(
        "--save-as-video",
        action='store_true',
        help="Save as video"
    )
    parser.add_argument(
        "--save-frames",
        action='store_true',
        help="Save individual frames"
    )
    parser.add_argument(
        "--provide-colour-map",
        action='store_true',
        help="Use a colour map object"
    )
    parser.add_argument(
        "--colour-map-object",
        type=str,
        help="Path to the colour map object (if applicable)"
    )
    parser.add_argument(
        "--label-description-data",
        type=str,
        help="Path to the csv with labels and descriptions."
    )
    parser.add_argument(
        "--label-description-column",
        type=str,
        help="Name of the column with label descriptions."
    )
    parser.add_argument(
        "--recenter-data",
        action='store_true',
        help="Recenter the data"
    )
    parser.add_argument(
        "--azimuth-calculation",
        action='store_true',
        help="Calculate the azimuth"
    )   
    
    args = parser.parse_args()
    
    # Validate dependencies
    if args.provide_colour_map and not args.colour_map_object:
        parser.error(
            "--colour_map_object is required when --provide_colour_map is set"
        )

    if args.save_as_video and not args.ffmpeg_executable_path:
        parser.error(
            "--ffmpeg_executable_path is required when --save_as_video is set"
        )
    
    # Validate label description data requirements
    if args.label_description_data and not args.label_column:
        parser.error(
            "--label_column is required when --label_description_data is provided"
        )
    
    if args.label_description_column and not args.label_description_data:
        parser.error(
            "--label_description_data is required when --label_description_column is provided"
        )
    
    return args


if __name__ == "__main__":
    print_welcome_message()
    args = parse_arguments()
    main(args)