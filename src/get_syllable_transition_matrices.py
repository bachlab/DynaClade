"""Module for analyzing syllable transition matrices using new metadata approach."""

# ==
# author: uSerratos (based on transition matrices module)
# date: Nov 2024
# ==

import pandas as pd
# from pomegranate.markov_chain import MarkovChain  # Not available in current version
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt
import time
import getpass
import sys
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

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
        # Get just the filename without path and extension
        current_file = os.path.splitext(os.path.basename(func.__code__.co_filename))[0].replace("_", " ").title()
        
        T1: time = time.time()
        func_out = func(*args, **kwargs)
        T2: time = time.time()
        T: time = time.strftime("%H:%M:%S", time.gmtime(T2 - T1))
        print_color(f"* |{current_file}| run time {T} *\n", bold_color.BOLD)
        return func_out

    return inner


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


def calculate_transition_matrix(sequences, le):
    """Calculate transition matrix directly from sequences."""
    # Get all unique states
    all_states = le.classes_
    n_states = len(all_states)
    
    # Initialize transition count matrix
    transition_counts = np.zeros((n_states, n_states))
    
    # Count transitions in all sequences
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            transition_counts[current_state, next_state] += 1
    
    # Convert counts to probabilities
    row_sums = transition_counts.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_counts / row_sums[:, np.newaxis]
    
    # Create DataFrame with proper labels
    matrix = pd.DataFrame(transition_matrix, 
                         index=all_states, 
                         columns=all_states)
    
    return matrix


def restructure_matrix(model):
    """Restructure the Markov chain model into a readable transition matrix."""
    # For compatibility with the old pomegranate model structure
    if hasattr(model, 'distributions') and hasattr(model.distributions[1], 'parameters'):
        transition_matrix = model.distributions[1].parameters[0]
        df = pd.DataFrame(transition_matrix, columns=['Initial State', 'Final State', 'Probability'])
        matrix = df.pivot(index='Initial State', columns='Final State', values='Probability')
        return matrix
    else:
        # If it's already a DataFrame, return as is
        return model


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


def plot_matrix(matrix, labels, output_dir, title="Transition matrix for state", show: bool = False, max_matrix_size: int = 2000):
    """Plot the transition matrix as a heatmap with memory-efficient settings for large matrices."""
    # Check matrix size to determine appropriate plotting parameters
    matrix_size = matrix.shape[0]
    
    # Calculate average label length for proportional font sizing
    avg_label_length = np.mean([len(str(label)) for label in labels]) if len(labels) > 0 else 1
    
    # Base font size calculation considering both matrix size and label length
    base_fontsize = max(4, 20 - (matrix_size * 0.01) - (avg_label_length * 0.5))
    
    # Debug information for font size calculation
    if matrix_size <= 100:  # Only print for smaller matrices to avoid spam
        print_color(f"Matrix size: {matrix_size}, Avg label length: {avg_label_length:.1f}, Calculated font size: {base_fontsize:.1f}", bold_color.CYAN)
    
    # For very large matrices, use more conservative settings
    if matrix_size > 1000:
        print_color(f"Large matrix detected ({matrix_size}x{matrix_size}). Using memory-efficient plotting settings.", bold_color.YELLOW)
        figsize = (8, 8)  # Smaller figure size
        dpi = 100  # Lower DPI to reduce memory usage
        fontsize = min(6, base_fontsize)  # Use calculated font size or 6, whichever is smaller
        # Skip tick labels for very large matrices to save memory
        show_ticks = False
    elif matrix_size > 500:
        figsize = (12, 12)
        dpi = 150
        fontsize = min(8, base_fontsize)  # Use calculated font size or 8, whichever is smaller
        show_ticks = True
    else:
        figsize = (15, 15)
        dpi = 300
        fontsize = min(10, base_fontsize)  # Use calculated font size or 10, whichever is smaller
        show_ticks = True
    
    # For extremely large matrices, skip plotting entirely
    if matrix_size > max_matrix_size:
        print_color(f"Matrix too large ({matrix_size}x{matrix_size}) to plot efficiently. Skipping plot generation.", bold_color.RED)
        return
    
    try:
        # Plot the transition matrix
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(matrix, cmap='plasma')

        fig.colorbar(cax)
        
        if show_ticks and matrix_size <= 100:
            # Only show tick labels for smaller matrices
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=fontsize)
            # Move x-axis labels to bottom
            ax.xaxis.set_ticks_position('bottom')
        else:
            # For larger matrices, just show grid without labels
            ax.set_xticks(range(0, len(labels), max(1, len(labels)//20)))
            ax.set_yticks(range(0, len(labels), max(1, len(labels)//20)))
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=min(20, 20 * 500 / matrix_size))  # Scale title size
        fig.subplots_adjust(left=0.1, top=0.90, right=0.95, bottom=0.1)
        fig.tight_layout()    

        # Save the figure
        fig_name = os.path.join(output_dir, title.replace(" ", "_") + '.png')
        plt.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        
        if show: 
            plt.show()

        plt.close(fig)
        
    except MemoryError as e:
        print_color(f"Memory error when plotting matrix {title}. Skipping plot generation. Error: {str(e)}", bold_color.RED)
        plt.close('all')  # Clean up any existing figures
        return
    except Exception as e:
        print_color(f"Error plotting matrix {title}: {str(e)}. Skipping plot generation.", bold_color.RED)
        plt.close('all')  # Clean up any existing figures
        return


def process_single_trial(group_data, args, matrix2, output_dir, le):
    """Process a single trial to create its transition matrix."""
    name = group_data[0]  # tuple of (ppid, session, trial)
    group = group_data[1]  # dataframe for this group
    
    name_current = f'transition_matrix_P{name[0]:03}_S{name[1]:03}_T{name[2]:03}'

    # Create sequence without self-transitions using original labels
    original_labels = group[args.label_column].tolist()
    new_sequence = [original_labels[0]]
    for i in range(1, len(original_labels)):
        if original_labels[i] != original_labels[i - 1]:
            new_sequence.append(original_labels[i])

    current_data = pd.DataFrame(new_sequence, columns=[args.label_column])
    sequences3, _ = encode_and_get_sequences(current_data, args.label_column, le)
    matrix3 = calculate_transition_matrix(sequences3, le)

    # The matrix3 already has the correct labels from calculate_transition_matrix
    # Create a new matrix with the same size and structure as matrix2, but initialized with zeros
    new_matrix = pd.DataFrame(np.zeros(matrix2.shape), index=matrix2.index, columns=matrix2.columns)

    # Iterate over the columns of matrix3
    for col in matrix3.columns:
        # Check if the column exists in new_matrix
        if col in new_matrix.columns:
            # Assign the values from matrix3 to the corresponding rows in new_matrix
            for i, value in enumerate(matrix3[col]):
                new_matrix.at[matrix3.index[i], col] = value

    plot_matrix(new_matrix, labels=new_matrix.columns, output_dir=output_dir, title=name_current.replace('_', ' '), max_matrix_size=args.max_matrix_size)
    new_matrix.to_csv(os.path.join(output_dir, f'{name_current}.csv'))


def get_transition_matrix(data, label_column, output_dir, save_model: bool = True, save_matrix: bool = True, 
                         file_name_prefix: str = "transition_matrix", save_matrix_figure: bool = True, le=None, max_matrix_size: int = 2000):
    """Generate transition matrices with and without self-transitions."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequences, le = encode_and_get_sequences(data, label_column, le)

    # Calculate transition matrix directly
    matrix_with_self_transitions = calculate_transition_matrix(sequences, le)
    if save_model:
        # Save the matrix as a simple model
        model_filename = os.path.join(output_dir, f'{file_name_prefix}_model_with_self_transitions.pkl')
        dump(matrix_with_self_transitions, model_filename)

    # The matrix already has the correct labels from calculate_transition_matrix
    # No need to inverse transform since they're already decoded
    pass
    if save_matrix_figure:
        plot_matrix(matrix_with_self_transitions, labels=matrix_with_self_transitions.columns, 
                   output_dir=output_dir, title=f"{file_name_prefix} including self-transitions", max_matrix_size=max_matrix_size)
    if save_matrix:
        matrix_with_self_transitions.to_csv(os.path.join(output_dir, f'{file_name_prefix}_with_self_transitions.csv'))

    # Create sequence without self-transitions using original labels
    original_labels = le.inverse_transform(sequences[0])
    new_sequence = [original_labels[0]]
    for i in range(1, len(original_labels)):
        if original_labels[i] != original_labels[i - 1]:
            new_sequence.append(original_labels[i])

    data2 = pd.DataFrame(new_sequence, columns=[label_column])

    sequences2, le2 = encode_and_get_sequences(data2, label_column, le)
    # Calculate transition matrix directly
    matrix_without_self_transitions = calculate_transition_matrix(sequences2, le)
    if save_model:
        # Save the matrix as a simple model
        model_filename2 = os.path.join(output_dir, f'{file_name_prefix}_model_without_self_transitions.pkl')
        dump(matrix_without_self_transitions, model_filename2)
    # The matrix already has the correct labels from calculate_transition_matrix
    # No need to set labels since they're already correct
    pass
    if save_matrix_figure:
        plot_matrix(matrix_without_self_transitions, labels=matrix_without_self_transitions.columns, 
                   output_dir=output_dir, title=f"{file_name_prefix} excluding self-transitions", max_matrix_size=max_matrix_size)
    if save_matrix:
        matrix_without_self_transitions.to_csv(os.path.join(output_dir, f'{file_name_prefix}_excluding_self_transitions.csv'))

    return matrix_with_self_transitions, matrix_without_self_transitions, le


@tictoc
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate syllable transition matrices from labeled data using new metadata approach.')
    parser.add_argument('--labels-path', type=str, required=True,
                      help='Path to the labels dataset CSV file')
    parser.add_argument('--label-column', type=str, required=True,
                      help='Name of the column containing labels')
    parser.add_argument('--metadata-path', type=str, required=True,
                      help='Path to the new metadata dataset CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Path to the output directory')
    parser.add_argument('--n-jobs', type=int, default=None,
                      help='Number of parallel jobs to run. Defaults to number of CPU cores - 1')
    parser.add_argument('--exclude-block', type=str, default=None,
                      help='Name of the block to exclude when creating "all_but_excluded" group (optional)')
    parser.add_argument('--no-plots', action='store_true',
                      help='Disable plot generation')
    parser.add_argument('--max-matrix-size', type=int, default=2000,
                      help='Maximum matrix size for plotting (default: 2000). Matrices larger than this will skip plotting.')
    parser.add_argument('--skip-trial-matrices', action='store_true',
                      help='Skip per-trial transition matrix computations and figures. Only process block-level matrices.')
    parser.add_argument('--failed-trials-path', type=str, required=False, default=None,
                      help='Path to CSV file containing metadata of failed trials (must have ppid, session, trial columns)')

    args = parser.parse_args()
    
    # Initialize row tracking system
    dropped_rows_tracker = []
    
    def log_dropped_row(ppid, session, trial, reason, details, stage, block_name=None):
        """Log a dropped row with detailed information."""
        dropped_rows_tracker.append({
            'ppid': ppid,
            'session': session,
            'trial': trial,
            'reason': reason,
            'details': details,
            'stage': stage,
            'block_name': block_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    if args.n_jobs is None:
        args.n_jobs = max(1, cpu_count() - 1)  # Leave one core free by default
    
    if args.output_dir is None:
        output_dir = os.path.dirname(args.metadata_path)
    else:
        output_dir = args.output_dir
    output_dir = os.path.join(output_dir, 'markov_transition_matrices_output')
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
    threat_column = 'threat2'
    threat_action_column = 'threat_action2'
    episode_set_name_column = 'episode_set_name2'
    
    print_color(f"Using columns: {threat_column}, {threat_action_column}, {episode_set_name_column}", bold_color.GREEN)
    
    # Filter out failed trials from dataset1 (labels) if failed trials data is provided
    if failed_trials_data is not None:
        print_color(f"\nFiltering out failed trials from labels dataset...", bold_color.BLUE)
        before_count = len(dataset1)
        
        # Create a mask to exclude failed trials
        failed_mask = dataset1.set_index(['ppid', 'session', 'trial']).index.isin(
            failed_trials_data.set_index(['ppid', 'session', 'trial']).index
        )
        
        if failed_mask.sum() > 0:
            # Track the failed trials being removed
            failed_trials_in_labels = dataset1[failed_mask]
            for _, row in failed_trials_in_labels.iterrows():
                log_dropped_row(
                    row['ppid'], row['session'], row['trial'],
                    "Failed trial exclusion",
                    f"Trial marked as failed in failed trials file. Excluded from transition matrix analysis.",
                    "Failed trial filtering",
                    None
                )
            
            # Remove failed trials from dataset1
            dataset1 = dataset1[~failed_mask]
            after_count = len(dataset1)
            print_color(f"Removed {before_count - after_count} failed trials from labels dataset", bold_color.YELLOW)
        else:
            print_color("No failed trials found in labels dataset", bold_color.YELLOW)
    
    # Filter out invalid labels
    dataset1 = dataset1[~dataset1[args.label_column].isin([-1, '-1'])]
    
    # First, create a label encoder that knows all possible labels
    le = LabelEncoder()
    le.fit(dataset1[args.label_column].unique())
    
    # Count number of unique observations (ppid-session-trial combinations)
    n_observations = len(dataset1.groupby(['ppid', 'session', 'trial']))

    # Initialize unique_blocks to avoid UnboundLocalError
    unique_blocks = []
    
    # For "all_trials", exclude "instructed" and "failed" blocks by default
    all_trials_data = dataset1.copy()
    if episode_set_name_column in dataset2.columns:
        # Get unique blocks first
        unique_blocks = dataset2[episode_set_name_column].unique()
        
        # Find blocks to exclude by default
        blocks_to_exclude = []
        for block_name in ['instructed', 'failed']:
            if block_name in unique_blocks:
                blocks_to_exclude.append(block_name)
                print_color(f"Excluding '{block_name}' block from all_trials analysis", bold_color.YELLOW)
        
        if blocks_to_exclude:
            # Create mask to exclude these blocks
            excluded_trials = set()
            for block_name in blocks_to_exclude:
                block_trials = set(
                    tuple(x) for x in dataset2[dataset2[episode_set_name_column] == block_name][['ppid', 'session', 'trial']].values
                )
                excluded_trials.update(block_trials)
            
            mask = ~dataset1[['ppid', 'session', 'trial']].apply(tuple, axis=1).isin(excluded_trials)
            all_trials_data = dataset1[mask].copy()
            print_color(f"All trials analysis now contains {len(all_trials_data.groupby(['ppid', 'session', 'trial']))} trials (excluded {len(excluded_trials)} trials from {blocks_to_exclude} blocks)", bold_color.GREEN)
    
    # Process all trials together
    print_color("\nProcessing all trials together:", bold_color.CYAN)
    
    # Get the transition matrices and label encoder for all trials
    matrix_with_self_transitions, matrix_without_self_transitions, _ = get_transition_matrix(
        all_trials_data, args.label_column, output_dir, 
        file_name_prefix=f"transition_matrix_n{n_observations}_all_trials",
        save_matrix_figure=not args.no_plots, le=le, max_matrix_size=args.max_matrix_size
    )

    # Process individual trials in parallel (only if not skipping)
    if not args.skip_trial_matrices:
        data3 = all_trials_data.copy().groupby(['ppid', 'session', 'trial'])
        groups_list = list(data3)
        
        # Create a partial function with fixed arguments
        process_trial = partial(process_single_trial, args=args, matrix2=matrix_without_self_transitions, 
                               output_dir=output_dir, le=le)
        
        # Process trials in parallel
        print_color(f"\nProcessing {len(groups_list)} trials using {args.n_jobs} parallel jobs...", bold_color.PURPLE)
        with Pool(processes=args.n_jobs) as pool:
            list(tqdm(pool.imap(process_trial, groups_list), total=len(groups_list), desc="Processing trials"))
    else:
        print_color("\nSkipping per-trial transition matrix computations as requested.", bold_color.YELLOW)

    # Process each block separately based on episode_set_name2
    analysis_groups = {'all_trials': all_trials_data}
    
    if episode_set_name_column in dataset2.columns:
        print_color(f"\nFound {len(unique_blocks)} unique blocks in {episode_set_name_column}: {list(unique_blocks)}", bold_color.GREEN)
        
        for block_name in unique_blocks:
            print_color(f"\nProcessing block: {block_name}", bold_color.CYAN)
            block_trials = dataset2[dataset2[episode_set_name_column] == block_name][['ppid', 'session', 'trial']]
            block_data = dataset1.merge(block_trials, on=['ppid', 'session', 'trial'], how='inner')
            analysis_groups[block_name] = block_data
        
        # Create 'all_but_excluded' group if a block is specified to exclude
        if args.exclude_block:
            if args.exclude_block in unique_blocks:
                excluded_trials = set(
                    tuple(x) for x in dataset2[dataset2[episode_set_name_column] == args.exclude_block][['ppid', 'session', 'trial']].values
                )
                mask = ~dataset1[['ppid', 'session', 'trial']].apply(tuple, axis=1).isin(excluded_trials)
                analysis_groups[f'all_but_{args.exclude_block}'] = dataset1[mask].copy()
                print_color(f"Created 'all_but_{args.exclude_block}' group", bold_color.GREEN)
            else:
                print_color(f"Warning: Specified exclude block '{args.exclude_block}' not found in available blocks: {list(unique_blocks)}", bold_color.YELLOW)
    else:
        print_color(f"Warning: Block column '{episode_set_name_column}' not found in metadata. Only 'all_trials' will be analyzed.", bold_color.YELLOW)

    # Now run transition matrix analysis for each group
    for group_name, group_data in analysis_groups.items():
        if group_data.empty:
            print_color(f"Warning: No data found for block/group {group_name}", bold_color.YELLOW)
            continue
            
        n_observations_group = len(group_data.groupby(['ppid', 'session', 'trial']))
        print_color(f"\nProcessing group '{group_name}' with {n_observations_group} trials", bold_color.CYAN)
        
        # Get the transition matrices for this group
        matrix_with_self_transitions_group, matrix_without_self_transitions_group, _ = get_transition_matrix(
            group_data, args.label_column, output_dir, 
            file_name_prefix=f"transition_matrix_n{n_observations_group}_{group_name}",
            save_matrix_figure=not args.no_plots, le=le, max_matrix_size=args.max_matrix_size
        )

    # Save the dropped rows tracking data
    if dropped_rows_tracker:
        dropped_rows_df = pd.DataFrame(dropped_rows_tracker)
        dropped_rows_path = os.path.join(output_dir, 'dropped_rows_tracking.csv')
        dropped_rows_df.to_csv(dropped_rows_path, index=False)
        print_color(f"\nSaved dropped rows tracking data to: {dropped_rows_path}", bold_color.GREEN)
        print_color(f"Total rows dropped: {len(dropped_rows_tracker)}", bold_color.YELLOW)
        
        # Print summary by reason
        reason_counts = dropped_rows_df['reason'].value_counts()
        print_color("\nDropped rows summary by reason:", bold_color.BLUE)
        for reason, count in reason_counts.items():
            print_color(f"  {reason}: {count} rows", bold_color.BLUE)
    else:
        print_color("\nNo rows were dropped during processing", bold_color.GREEN)

    print_color("\nTransition matrices analysis completed successfully!", bold_color.BLUE)
    print_color(f"\nThe data can be found in: ".capitalize() + f"{output_dir}", bold_color.BLUE)


if __name__ == "__main__":
    print_welcome_message()
    main() 