"""Module for cleaning motion capture data."""

# ==
# author: uSerratos
# date: Nov 2024
# ==

import os
import sys
from argparse import ArgumentParser
import json
import pdb

if not sys.modules.get('utils'):
    if os.environ.get('RUNNING_AS_SUBPROCESS') != 'true':
        os.system("cls||clear")
    from utils import *


# Get the filename without extension for messages
CURRENT_FILE = os.path.splitext(os.path.basename(__file__))[0]

class Args:
    def __init__(self, data, ffmpeg, recursive, output, overwrite, stickfigure, save_figure, sample_rate, smoothing,
                 stick_figure_template, plot_frame_rate, stick_figure_plot_pause, plot_smoothing_figures,
                 marker_quality_threshold, marker_quality_sub_threshold, segment_length_tolerance,
                 mean_segment_length_deviation_threshold, individual_segment_length_deviation_threshold,
                 lengths_data_subset, estimate_loops, save_output):
        self.data = data
        self.ffmpeg = ffmpeg
        self.recursive = recursive
        self.output = output  # Now stores the output directory path
        self.overwrite = overwrite
        self.stickfigure = stickfigure
        self.save_figure = save_figure
        self.sample_rate = sample_rate
        self.smoothing = smoothing
        self.stick_figure_template = stick_figure_template
        self.plot_frame_rate = plot_frame_rate
        self.stick_figure_plot_pause = stick_figure_plot_pause
        self.plot_smoothing_figures = plot_smoothing_figures
        self.marker_quality_threshold = marker_quality_threshold
        self.marker_quality_sub_threshold = marker_quality_sub_threshold
        self.segment_length_tolerance = segment_length_tolerance
        self.mean_segment_length_deviation_threshold = mean_segment_length_deviation_threshold
        self.individual_segment_length_deviation_threshold = individual_segment_length_deviation_threshold
        self.lengths_data_subset = lengths_data_subset
        self.estimate_loops = estimate_loops
        self.save_output = save_output


def remove_files(file_list):
    for f in file_list:
        try:
            os.remove(f)
        except Exception as e:
            logging.error(f"Failed to remove {f}. Reason: {e}")


def remove_previous_data(directory: int, recursive: bool) -> None:
    """Removes previously saved data"""
    mocap_clean = {
        x
        for x in glob.glob(
            os.path.join(f"{directory}", "**/*.csv"), recursive=recursive
        )
        if (
            os.path.basename(os.path.abspath(os.path.join(x, os.pardir)))[0]
            != "_"
        )
    }
    mocap_clean_stickfigures = {
        x
        for x in glob.glob(
            os.path.join(f"{directory}", "**/*.mp4"), recursive=recursive
        )
        if (
            os.path.basename(os.path.abspath(os.path.join(x, os.pardir)))[0]
            != "_"
        )
    }
    mocap_clean = [
        x
        for x in mocap_clean
        if ("mocap_clean_T" in x)
        or ("mocap_clean_metadata" in x)
        or ("files_successfully_processed" in x)
    ]
    mocap_clean_stickfigures = [
        x for x in mocap_clean_stickfigures if ("stickfigure_T" in x)
    ]
    found_data = (len(mocap_clean) > 0) or (len(mocap_clean_stickfigures) > 0)
    remove_files(mocap_clean)
    remove_files(mocap_clean_stickfigures)
    if found_data:
        logging.info("Successfully removed previously saved data")


def ask_question_and_update(ad, key, question):
    ad.update(
        {
            key: ask_question_options(question=question, options=["y", "n"])
            == "y"
        }
    )


def parse_arguments() -> Args:
    """Parse command line arguments for mocap cleaning.
    
    Returns:
        Args: Parsed command line arguments
    """
    parser = ArgumentParser(
        description="Process mocap data.",
        usage="This code does mocap_clean batch processing"
    )
    
    # Create argument groups
    output_group = parser.add_argument_group(title="Output arguments")
    stickfigure_group = parser.add_argument_group(title="Stickfigure args")
    
    # Required arguments
    parser.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help="Provide directory where study is located"
    )
    parser.add_argument(
        "-f", "--ffmpeg",
        type=str,
        required=True,
        help="Provide directory where ffmpeg.exe is located"
    )
    parser.add_argument(
        "-sr", "--sample-rate",
        type=int,
        required=True,
        help="Data collection sample rate in Hz (e.g. 100, 200)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Find data recursively"
    )
    parser.add_argument(
        "--stick-figure-template",
        type=str,
        default="coda",
        help="Template for stick figure"
    )
    parser.add_argument(
        "--plot-frame-rate",
        type=int,
        default=100,
        help="Frame rate for plotting"
    )
    parser.add_argument(
        "--stick-figure-plot-pause",
        type=bool,
        default=False,
        help="Pause stick figure plot"
    )
    parser.add_argument(
        "--plot-smoothing-figures",
        type=bool,
        default=False,
        help="Plot smoothing figures"
    )
    
    # Quality threshold arguments
    parser.add_argument(
        "--marker-quality-threshold",
        type=int,
        default=85,
        help="Marker quality threshold"
    )
    parser.add_argument(
        "--marker-quality-sub-threshold",
        type=int,
        default=90,
        help="Marker quality sub-threshold"
    )
    parser.add_argument(
        "--segment-length-tolerance",
        type=int,
        default=15,
        help="Segment length tolerance"
    )
    parser.add_argument(
        "--mean-segment-length-deviation-threshold",
        type=int,
        default=80,
        help="Mean segment length deviation threshold"
    )
    parser.add_argument(
        "--individual-segment-length-deviation-threshold",
        type=int,
        default=80,
        help="Individual segment length deviation threshold"
    )
    parser.add_argument(
        "--lengths-data-subset",
        type=int,
        default=100,
        help="Lengths data subset"
    )
    parser.add_argument(
        "--estimate-loops",
        type=int,
        default=1,
        help="Estimate loops"
    )
    parser.add_argument(
        "--save-output",
        type=bool,
        default=True,
        help="Save output"
    )
    
    # Output group arguments
    output_group.add_argument(
        "-o", "--output",
        type=str,
        help="Save data in separate directory as independent data structure (provide directory path)"
    )
    output_group.add_argument(
        "-ow", "--overwrite",
        action="store_true",
        help="Overwrite data"
    )
    
    # Stickfigure group arguments
    stickfigure_group.add_argument(
        "-p", "--stickfigure",
        action="store_true",
        help="Plot stick figure"
    )
    stickfigure_group.add_argument( 
        "-s", "--save-figure",
        action="store_true",
        help="Save stickfigure frames as a video"
    )
    
    # Smoothing argument
    parser.add_argument(
        "-sm", "--smoothing",
        action="store_true",
        help="Apply smoothing filters to data"
    )
    
    ap = parser.parse_args()
    
    # Create Args object from parsed arguments
    args = Args(
        data=ap.data,
        ffmpeg=ap.ffmpeg,
        recursive=ap.recursive,
        output=ap.output,
        overwrite=ap.overwrite,
        stickfigure=ap.stickfigure,
        save_figure=ap.save_figure,
        sample_rate=ap.sample_rate,
        smoothing=ap.smoothing,
        stick_figure_template=ap.stick_figure_template,
        plot_frame_rate=ap.plot_frame_rate,
        stick_figure_plot_pause=ap.stick_figure_plot_pause,
        plot_smoothing_figures=ap.plot_smoothing_figures,
        marker_quality_threshold=ap.marker_quality_threshold,
        marker_quality_sub_threshold=ap.marker_quality_sub_threshold,
        segment_length_tolerance=ap.segment_length_tolerance,
        mean_segment_length_deviation_threshold=
            ap.mean_segment_length_deviation_threshold,
        individual_segment_length_deviation_threshold=
            ap.individual_segment_length_deviation_threshold,
        lengths_data_subset=ap.lengths_data_subset,
        estimate_loops=ap.estimate_loops,
        save_output=ap.save_output
    )
    
    # Print configuration message
    _print_configuration_message(args)
    
    return args


def _print_configuration_message(args: Args) -> None:
    """Print the configuration message with all argument values.
    
    Args:
        args: Parsed command line arguments
    """
    message = (
        "\n MOCAP CLEAN BATCH PROCESSING PARAMETERS: \n"
        f"\n  [study directory] => {args.data}"
        f"\n  [ffmpeg directory] => {args.ffmpeg}"
        f"\n  [recursive] => {args.recursive}"
        f"\n  [output directory] => {os.path.join(args.output, 'mocap_processed_data') if args.output else 'default (syllable_analysis_output)'}"
        f"\n  [overwrite] => {args.overwrite}"
        f"\n  [plot stick figure] => {args.stickfigure}"
        f"\n  [save stick figure as video] => {args.save_figure}"
        f"\n  [data sample rate] => {args.sample_rate}"
        f"\n  [apply smoothing filters] => {args.smoothing}"
        f"\n  [stick figure template] => {args.stick_figure_template}"
        f"\n  [plot frame rate] => {args.plot_frame_rate}"
        f"\n  [stick figure plot pause] => {args.stick_figure_plot_pause}"
        f"\n  [plot smoothing figures] => {args.plot_smoothing_figures}"
        f"\n  [marker quality threshold] => {args.marker_quality_threshold}"
        f"\n  [marker quality sub-threshold] => "
        f"{args.marker_quality_sub_threshold}"
        f"\n  [segment length tolerance] => {args.segment_length_tolerance}"
        f"\n  [mean segment length deviation threshold] => "
        f"{args.mean_segment_length_deviation_threshold}"
        f"\n  [individual segment length deviation threshold] => "
        f"{args.individual_segment_length_deviation_threshold}"
        f"\n  [lengths data subset] => {args.lengths_data_subset}"
        f"\n  [estimate loops] => {args.estimate_loops}"
        f"\n  [save output] => {args.save_output}\n"
    )

    print_color(message, color=bold_color.GREEN)
    logger.info(
        f"|mocap_clean_batch_processing.py| Study directory: {args.data}"
    )


def process_files(files, args, output_directory, new_parent, directories_processed):

    total_files = len(files)
    idx = 1
    for idx2, f in enumerate(tqdm(files, desc=f"Processing file {idx}/{total_files}", leave=False), start=1):
        # tqdm.write(f"PROCESSING:  {idx}/{total_files} | {f}")
        idx += 1
        file_path_split = f.split("\\")
        session_pattern = r'^S\d{3}'
        if not any(re.search(session_pattern, e) for e in file_path_split):
            logger.warning(f"Skipping file {f} as it does not contain a session pattern.")
            continue

        session_id = [e for e in file_path_split if re.search(session_pattern, e)][0]
        trialid_pattern = r'T\d{3}'

        trial_id = [re.search(trialid_pattern, element).group() for element in file_path_split if re.search(trialid_pattern, element)][0]
        participant_id = file_path_split[file_path_split.index(session_id) - 1]
        session_id = int(session_id[1:])
        trial_id = int(trial_id[1:])

        # Ensure that file_path_split has at least two elements
        if not session_id or not trial_id or not participant_id:
            logger.error(f"No session and trial found in file: {f}")
            continue

        logger.info(f"|{__file__}| Processing: {f}")
        # print_color(f"\nPROCESSING: Participant {participant_id} | Session: {session_id} | Trial: {trial_id}\n", color=bold_color.GREEN)
        # file_path_split = os.path.relpath(f, args.data).split("\\")


        try:
            mocap_clean(
                data_path=f,
                data_sample_rate=args.sample_rate,
                stick_figure_template=args.stick_figure_template,
                plot_frame_rate=args.plot_frame_rate,
                stick_figure_plot_pause=args.stick_figure_plot_pause,
                plot_smoothing_figures=args.plot_smoothing_figures,
                marker_quality_threshold=args.marker_quality_threshold,
                marker_quality_sub_threshold=args.marker_quality_sub_threshold,
                segment_length_tolerance=args.segment_length_tolerance,
                mean_segment_length_deviation_threshold = args.mean_segment_length_deviation_threshold,
                individual_segment_length_deviation_threshold = args.individual_segment_length_deviation_threshold,
                lengths_data_subset=args.lengths_data_subset,
                ffmpeg_executable_path=args.ffmpeg,
                estimate_loops=args.estimate_loops,
                apply_smoothing_filters=args.smoothing,
                plot_stick_figure=args.stickfigure,
                output_directory=output_directory,
                save_stickfigure_as_video=args.save_figure,
                save_output=args.save_output,
                output_metadata=new_parent,
                participant_id=participant_id,
                session_id=session_id,
                trial_id=trial_id,
            )
        except ValueError as e:
            message = (
                f"|{__file__}| [mocap_clean]"
                + f"function for {f} failed, reason {e}"
            )
            logger.critical(message)


import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt

def downsample_signal(signal, original_fs, target_fs: int = 100):
    """
    Downsamples a signal using decimation with anti-aliasing.

    Args:
      signal: The input signal as a NumPy array.
      original_fs: The original sampling frequency in Hz.
      target_fs: The target sampling frequency in Hz.

    Returns:
      A tuple containing:
        - The downsampled signal as a NumPy array.
        - The time points for the original signal.
        - The time points for the downsampled signal.
    """
 
    # Calculate the downsampling factor
    downsampling_factor = int(original_fs / target_fs)

    # Downsample the signal using decimate
    downsampled_signal = decimate(signal, downsampling_factor, ftype='fir')

    # Generate time points for the original and downsampled signals
    time_original = np.arange(0, len(signal) / original_fs, 1 / original_fs)
    time_downsampled = np.arange(0, len(downsampled_signal) / target_fs, 1 / target_fs)

    # Visualize the original and downsampled signals
    plt.figure(figsize=(10, 6))
    plt.plot(time_original, signal, label=f'Original Signal ({original_fs}Hz)')
    plt.plot(time_downsampled, downsampled_signal, label=f'Downsampled Signal ({target_fs}Hz)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Original vs. Downsampled Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

    return downsampled_signal, time_downsampled


@tictoc
def main():
    """Main function to handle mocap data cleaning and processing."""
    args = parse_arguments()

    # Set up output paths
    if args.output:
        # Use the provided output directory with mocap_processed_data subdirectory
        new_parent = os.path.join(os.path.abspath(args.output), "mocap_processed_data")
    else:
        # Use default output directory structure
        main_path_parent = os.path.normpath(os.path.join(args.data, os.pardir))
        new_parent = os.path.join(
            main_path_parent, 
            "syllable_analysis_output",
            "_".join([
                "mocap_clean",
                str(args.marker_quality_threshold),
                str(args.marker_quality_sub_threshold),
                str(args.segment_length_tolerance),
                str(args.mean_segment_length_deviation_threshold),
                str(args.individual_segment_length_deviation_threshold)
            ])
        )

    # Handle existing data
    # Always overwrite when using custom output directory, or when overwrite flag is set
    should_overwrite = args.output or args.overwrite
    if os.path.exists(new_parent) and should_overwrite:
        try:
            remove_previous_data(
                directory=new_parent, 
                recursive=args.recursive
            )
        except Exception as e:
            logger.error(f"Failed to delete {new_parent}. Reason: {e}")

    # Create output directory
    new_parent = create_directory(new_parent)

    # Find target directories
    directories_list = {
        os.path.normpath(os.path.join(x, os.pardir))
        for x in glob.glob(
            os.path.join(f"{args.data}", "**/*.csv"), 
            recursive=args.recursive
        )
        if (os.path.basename(os.path.abspath(os.path.join(x, os.pardir)))[0]
            != "_")
    }

    # Filter directories
    directories_list = {
        x for x in directories_list
        if os.path.relpath(x, args.data)[:1] != "_"
    }
    directories_list = natsorted(directories_list)

    # Process each directory
    idx = 1
    for idx2, d in enumerate(tqdm(directories_list, desc=f"Processing directory {idx}/{len(directories_list)}", leave=False), start=1):
        idx += 1
        # tqdm.write(f"Directory {idx}/{len(directories_list)} | {d}")
        directories_processed = f"{idx}/{len(directories_list)}"
        # Set up output directory
        rel_path = os.path.relpath(d, args.data)
        output_directory = os.path.join(new_parent, rel_path)
        if not os.path.exists(output_directory):
            create_directory(output_directory)

        # Get list of files to process
        files_list = [
            x for x in glob.glob(
                os.path.join(f"{d}", "*.csv"), 
                recursive=False
            )
        ]

        # Process files in current directory
        process_files(files_list, args, output_directory, new_parent, directories_processed)

    try:
        # Process metadata
        metadata_path = os.path.join(new_parent, "checks_output.csv")
        mocap_metadata = pd.read_csv(metadata_path)
        
        # Filter data that passed checks
        check_columns = ['check_1', 'check_2', 'check_3']
        data_curated = mocap_metadata[
            mocap_metadata[check_columns].all(axis=1)
        ]
        
        # Save curated data
        curated_path = os.path.join(new_parent, "data_passed_checks.csv")
        data_curated.to_csv(curated_path, index=False)
        data_passed_checks = len(data_curated)

        # Print results
        print_color(
            f"\n{data_passed_checks} out of {len(mocap_metadata)} files "
            f"passed all checks", 
            bold_color.GREEN
        )
        print_color(
            f"\nMoCap data that passed checks is saved in {curated_path}",
            bold_color.GREEN
        )

        # Save arguments and metadata
        args_dict = vars(args)
        args_dict['output_directory'] = new_parent
        args_dict['data_passed_checks'] = data_passed_checks
        
        args_file_name = (
            "mocap_clean_arguments_and_output_metadata.json"
        )
        args_json_path = os.path.join(args.data, args_file_name)
        
        with open(args_json_path, 'w') as json_file:
            json.dump(args_dict, json_file, indent=4)
        
        # print(f"\nArguments and output metadata saved to {args_json_path}")

        # Print final summary
        message = (
            "\n MOCAP CLEAN BATCH PROCESSING FINISHED: \n"
            f"\n [CHECKS] => {data_passed_checks} / "
            f"{len(mocap_metadata)} files passed all checks"
            f"\n [DATA PASSED CHECKS] => {curated_path}"
            f"\n [ARGS AND OUTPUT METADATA] => {args_json_path}\n"
        )
        print_color(message, color=bold_color.BLUE)

    except Exception as e:
        logger.error(f"Failed to process metadata. Reason: {e}")


if __name__ == "__main__":
    print_welcome_message()
    main()
