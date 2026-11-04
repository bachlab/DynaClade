import re
import joblib
import json
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from utils import tictoc, print_welcome_message, ffmpeg_process, list_directory_items, StickFigure, skf, print_color, bold_color
from plot_syllable_stickfigures import plot_stick_figure
import pdb


@tictoc
def get_most_representative_prototypes(clusterer, data_file=None):
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    unique_labels = np.unique(labels)

    most_representative_prototypes = {}
    for label in unique_labels:
        if label == -1:
            continue  # Skip the noise label
        indices = np.where(labels == label)[0]
        probabilities_for_label = probabilities[indices]
        
        # Find all indices that have the maximum probability
        max_prob = np.max(probabilities_for_label)
        max_prob_indices = indices[probabilities_for_label == max_prob]
        
        if len(max_prob_indices) == 1:
            most_representative_index = max_prob_indices[0]
        else:
            print_color(f"\nFound {len(max_prob_indices)} samples with same probability {max_prob} for label {label}", color=bold_color.YELLOW)
            
            if data_file is not None:
                # Load the data and use additional features to break the tie
                data = pd.read_csv(data_file)
                
                # Add diagnostic information
                print_color(f"Data file length: {len(data)}", color=bold_color.BLUE)
                print_color(f"Max index in clusterer: {np.max(max_prob_indices)}", color=bold_color.BLUE)
                print_color(f"Min index in clusterer: {np.min(max_prob_indices)}", color=bold_color.BLUE)
                
                # Validate indices against data length
                valid_indices = max_prob_indices[max_prob_indices < len(data)]
                if len(valid_indices) == 0:
                    print_color(f"Warning: No valid indices found for label {label}. Skipping.", color=bold_color.RED)
                    print_color(f"All indices were out of bounds. Data length: {len(data)}, Max index: {np.max(max_prob_indices)}", color=bold_color.RED)
                    continue
                    
                candidate_samples = data.iloc[valid_indices]
                
                # Get numerical columns only
                numeric_cols = candidate_samples.select_dtypes(include=[np.number]).columns
                # Exclude non-feature columns if they exist
                cols_to_exclude = ['ppid', 'session', 'trial', 'time', 'label']
                feature_cols = [col for col in numeric_cols if col not in cols_to_exclude]
                
                if len(feature_cols) > 0:
                    # Calculate the mean of all samples in this label
                    label_mean = data.iloc[indices[indices < len(data)]][feature_cols].mean()
                    
                    # Calculate Euclidean distance to the mean for each candidate
                    distances = []
                    for idx in valid_indices:
                        sample = data.iloc[idx][feature_cols]
                        distance = np.sqrt(((sample - label_mean) ** 2).sum())
                        distances.append(distance)
                    
                    # Select the sample closest to the mean
                    closest_to_mean_idx = np.argmin(distances)
                    most_representative_index = valid_indices[closest_to_mean_idx]
                    
                    print_color(f"Selected sample closest to cluster mean (distance: {distances[closest_to_mean_idx]:.4f})", color=bold_color.GREEN)
                else:
                    # If no suitable features found, just take the first one
                    most_representative_index = valid_indices[0]
                    print_color("No suitable features found to break the tie. Using first sample.", color=bold_color.YELLOW)
            else:
                # If no data file provided, just take the first one
                most_representative_index = max_prob_indices[0]
                print_color("No data file provided to break the tie. Using first sample.", color=bold_color.YELLOW)
        
        most_representative_prototypes[label] = most_representative_index
        
    return most_representative_prototypes


@tictoc
def get_data_prototypes(prototype_index: dict, data: str):
    data = pd.read_csv(data)
    index = [i for i in prototype_index.values()]
    data_prototypes = data.iloc[index]
    return data_prototypes


def get_prototypes_from_videos(prototype_videos_path: str, data_prototypes: pd.DataFrame, ffmpeg_path: str, out_dir: str):
    
    out_dir = os.path.join(out_dir, 'prototype_label_stickfigures')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    list_videos = [i for i in os.listdir(prototype_videos_path) if os.path.isfile(os.path.join(prototype_videos_path, i))]

    for _, row in data_prototypes.iterrows():
        # Create the filename pattern
        pattern = "P{:03d}_S{:03d}_T{:03d}".format(int(row['ppid']), int(row['session']), int(row['trial']))

        matching_videos = [video for video in list_videos if pattern in video]

        input_file = os.path.join(prototype_videos_path, matching_videos[0])

        if matching_videos:
            ffmpeg_process(
                    command='trim',
                    input_file_or_directory=input_file,
                    ffmpeg_executable_path=ffmpeg_path,
                    output_file_or_directory_name=out_dir,
                    video_frame_rate=100,
                    trim_start=row['time'],
                    trim_end=row['time'] + 1,
                    video_name=f'{int(row["label"])}_{pattern}',
                    )


@tictoc
def main():
    parser = argparse.ArgumentParser(description='Process motion data and generate prototypes')

    # General arguments
    parser.add_argument('--ffmpeg-path', type=str, default=".", help='Path to the ffmpeg executable')
    parser.add_argument('--out-dir', type=str, default=".", help='Path to the output directory')
    
    # Clusterer analysis arguments
    parser.add_argument('--clusterer-file', type=str, default=None, help='Path to the clusterer file')
    parser.add_argument('--clusterer-data-file', type=str, default=None, help='Path to the clusterer data file, expects a csv file with the following columns: "ppid", "session", "trial", "time", "label"')
    
    # Visuals arguments
    parser.add_argument('--prototype-videos-path', type=str, default=None, help='Path to the prototype videos directory')
    
    # Label prototypes arguments
    parser.add_argument('--xyz-coordinates-path', type=str, default=None, help='Path to the xyz coordinates directory')
    parser.add_argument('--prototype-labels-file', type=str, default=None, help='Path to the prototype labels file')
    parser.add_argument('--label-column', type=str, default='label', help='Name of the column with labels for prototypes data')
    parser.add_argument('--stick-figure-template', type=str, default="qualisys_csv", choices=['coda', 'openpose', 'qualisys_csv'], help='Stickfigure template to use')



    args = parser.parse_args()

    # Set default output directory based on clusterer file if provided
    out_dir = os.path.join(args.out_dir, 'label_prototypes')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.clusterer_file:
        print_color(f'EXTRACTING LABEL PROTOTYPES FROM: {args.clusterer_file}', color=bold_color.GREEN)
        # Load the clusterer
        clusterer = joblib.load(args.clusterer_file)
        output_file = os.path.join(out_dir, 'label_prototypes.json')
    
        label_prototypes = get_most_representative_prototypes(clusterer, args.clusterer_data_file)
        label_prototypes = {str(k): v.tolist() for k, v in label_prototypes.items()}

        # Save the label_prototypes dictionary as a JSON file
        with open(output_file, 'w') as f:
            json.dump(label_prototypes, f)
        
        data_prototypes = get_data_prototypes(label_prototypes, args.clusterer_data_file)
        data_prototypes.to_csv(os.path.join(out_dir, 'label_prototypes.csv'), index=False)

        args.prototype_labels_file = os.path.join(out_dir, 'label_prototypes.csv')

        print_color(f'OUTPUT DATA: \n Label Prototypes: {os.path.join(out_dir, "label_prototypes.csv")} \n Label Prototypes JSON: {os.path.join(out_dir, "label_prototypes.json")} ', color=bold_color.BLUE)

    if args.xyz_coordinates_path and args.prototype_labels_file and args.label_column and args.stick_figure_template:
        print_color(f'PLOTTING LABEL PROTOTYPES FROM: {args.prototype_labels_file}\n', color=bold_color.GREEN)
        # PLUS_MINUS = 100
        MINUS = 100
        PLUS = 0
        ALPHA = 0.1
        PLOT_FRAMERATE = 8
        DATA_COLLECTION_RATE = 100
        AXIS_LIM = (1,1,1,1,1,1)
        xyz_files = list_directory_items(args.xyz_coordinates_path, recursive=True, search_pattern=["T", ".csv"]).get(
            "items"
        )

        stickfigure = StickFigure(
            name=skf[args.stick_figure_template]["name"],
            body=skf[args.stick_figure_template]["body"],
            data_mapping=skf[args.stick_figure_template]["data_mapping"],
            )

        list_xyz_files = []
        for current_file in xyz_files:
            session = int(re.search(r'S(\d{3})', current_file).group(1))
            trial = int(re.search(r'T(\d{3})', current_file).group(1))
            current_file_parent = os.path.dirname(current_file)
            current_file_grandparent = int(os.path.dirname(current_file_parent).split("\\")[-1])
            list_xyz_files.append((current_file_grandparent, session, trial, current_file))
        
        xyz_files_df = pd.DataFrame(list_xyz_files, columns=["ppid", "session", "trial", "file"])

        label_prototypes_df = pd.read_csv(args.prototype_labels_file)

        data_prototypes = pd.merge(label_prototypes_df, xyz_files_df, on=["ppid", "session", "trial"], how="inner")

        processed_label_prototypes = [] 
        for _, row in tqdm(data_prototypes.iterrows(), desc="Processing label prototypes", total=len(data_prototypes), leave=False):
            data = pd.read_csv(row['file'])
            match_idx = data[data['time'] == row['time']].index[0]    
            if (match_idx -MINUS > 0) and (match_idx + PLUS < len(data)):
                data_plus_minus = data.loc[match_idx - MINUS: match_idx]
            elif (match_idx -MINUS < 0):
                data_plus_minus = data.loc[match_idx: match_idx + 10]
            elif (match_idx + PLUS > len(data)):
                data_plus_minus = data.loc[match_idx - MINUS: match_idx]
            else:
                None
            
            if row[args.label_column] in processed_label_prototypes:
                continue
            
            def recenter_data(data_plus_minus, stickfigure):
                data_plus_minus = data_plus_minus.copy()
                data_plus_minus.loc[:, args.label_column] = row[args.label_column]
                data_plus_minus = data_plus_minus.reset_index(drop=True)
                MHip_cols = stickfigure._markers_xyz['MHip']
                subtract_column_x = MHip_cols[0]
                x_columns = data_plus_minus.filter(regex='_x$').columns
                data_plus_minus[x_columns] = data_plus_minus[x_columns].subtract(data_plus_minus[subtract_column_x][0], axis=0)
                subtract_column_y = MHip_cols[1]
                y_columns = data_plus_minus.filter(regex='_y$').columns
                data_plus_minus[y_columns] = data_plus_minus[y_columns].subtract(data_plus_minus[subtract_column_y][0], axis=0)
                out_dir_plot = os.path.join(out_dir, 'stickfigures')

                return data_plus_minus, out_dir_plot
            
            data_plus_minus, out_dir_plot = recenter_data(data_plus_minus, stickfigure)

            if not os.path.exists(out_dir_plot):
                os.makedirs(out_dir_plot)
            
            def azimuth_calculation(data_plus_minus, stickfigure):
                # Compute the shoulder to shoulder angle with respect to global frame
                shoulder_to_shoulder_angles = []
                for _, row in data_plus_minus.iterrows():
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
            
            azimuth = azimuth_calculation(data_plus_minus, stickfigure)

            label_description = int(row[args.label_column]) if not isinstance(row[args.label_column], str) else row[args.label_column]
            plot_stick_figure(data_plus_minus, data_sample_rate=DATA_COLLECTION_RATE, plot_sequence=True, 
                            save_as_video=True, video_name=f"Label_{label_description}_Prototype", 
                            stick_figure_template=args.stick_figure_template, ffmpeg_executable_path=args.ffmpeg_path, 
                            linewidth=4, prototypes_plot=True,
                            label_column=args.label_column, video_output_path=out_dir_plot, 
                            title=f"Label {label_description} Prototype", plot_frame_rate=PLOT_FRAMERATE, 
                            alpha=ALPHA, axis_lim=AXIS_LIM, adjust_axis=False, elevation=20, azimuth=azimuth)
            
            processed_label_prototypes.append(label_description)

        print_color(f'PLOTTING COMPLETED\n', color=bold_color.GREEN)
        print_color(f'OUTPUT DATA: \n Label Prototypes Stickfigures: {out_dir_plot} ', color=bold_color.BLUE)

    if args.prototype_videos_path:
        get_prototypes_from_videos(
            prototype_videos_path=args.prototype_videos_path,
            data_prototypes=data_prototypes,
            ffmpeg_path=args.ffmpeg_path,
            out_dir=out_dir
        )

if __name__ == "__main__":
    os.system("cls||clear")
    print_welcome_message()
    main()