"""Module for analyzing syllable bigrams and trigrams (n-grams) using new metadata approach."""

# ==
# author: uSerratos (based on counts/durations modules)
# date: Nov 2024
# ==

import os
import sys
import time
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import pandas as pd
from collections import Counter, defaultdict

# Only import utils if it hasn't been imported already
if 'utils' not in sys.modules:
    if os.environ.get('RUNNING_AS_SUBPROCESS') != 'true':
        os.system("cls||clear")
    from utils import *

warnings.filterwarnings("ignore")

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

def get_ngram_type_name(n):
    """Get the human-readable name for an n-gram size."""
    if n == 2:
        return "bigram"
    elif n == 3:
        return "trigram"
    elif n == 4:
        return "4-gram"
    elif n == 5:
        return "5-gram"
    else:
        return f"{n}-gram"

def extract_ngrams(labels, n):
    """Extract n-grams (as tuples) from a list/array of labels, excluding self-transitions."""
    ngrams = []
    for i in range(len(labels)-n+1):
        ngram = tuple(labels[i:i+n])
        # Skip if any consecutive elements in the n-gram are the same
        if all(ngram[j] != ngram[j+1] for j in range(len(ngram)-1)):
            ngrams.append(ngram)
    return ngrams

def extract_ngrams_with_frames(labels, n):
    """Extract n-grams and their start/end frame indices, excluding self-transitions."""
    ngrams_with_frames = []
    for i in range(len(labels)-n+1):
        ngram = tuple(labels[i:i+n])
        # Skip if any consecutive elements in the n-gram are the same
        if all(ngram[j] != ngram[j+1] for j in range(len(ngram)-1)):
            ngrams_with_frames.append((ngram, i, i+n-1))
    return ngrams_with_frames

def count_ngrams_in_trials(dataset, label_column, n):
    """Count n-grams and collect their durations across all trials."""
    ngram_counter = Counter()
    ngram_durations = defaultdict(list)
    for (ppid, session, trial), trial_data in tqdm(dataset.groupby(['ppid', 'session', 'trial']), desc=f"Extracting {n}-grams"):
        labels = trial_data[label_column].values
        ngrams_with_frames = extract_ngrams_with_frames(labels, n)
        for ngram, start, end in ngrams_with_frames:
            ngram_counter[ngram] += 1
            ngram_durations[ngram].append((start, end))
    return ngram_counter, ngram_durations

def compute_ngram_frequencies(ngram_counter, total_ngrams):
    """Compute normalized frequencies for each n-gram."""
    return {ngram: count/total_ngrams for ngram, count in ngram_counter.items()}

def compute_ngram_duration_stats(ngram_durations, fps=10):
    """Compute mean, std, and distribution of durations for each n-gram."""
    stats = {}
    for ngram, spans in ngram_durations.items():
        durations = [end-start+1 for (start, end) in spans]
        durations_sec = np.array(durations) / fps
        stats[ngram] = {
            'mean': np.mean(durations_sec),
            'std': np.std(durations_sec),
            'count': len(durations_sec)
        }
    return stats

def ngram_matrix_plot(ngram_freq, ngram_pvals, syllable_list, n, title, output_path, color_map=None, pval_thresh=0.01, save=True, n_observations=None, block_name=None):
    """Plot n-gram frequency bar plots with color-coded bars."""
    import matplotlib.patches as mpatches
    from matplotlib.font_manager import FontProperties
    
    # Normalize frequencies to probabilities
    total_freq = sum(ngram_freq.values())
    ngram_probs = {k: v/total_freq for k, v in ngram_freq.items()}
    
    # Sort n-grams by probability for bar plot
    sorted_ngrams = sorted(ngram_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top 20 n-grams
    top_n = min(20, len(sorted_ngrams))
    num_observed_ngrams = len(ngram_probs)
    
    # Generate n-gram labels and extract syllables for any n-gram size
    ngrams = []
    all_syllables = []
    for ngram_tuple, _ in sorted_ngrams[:top_n]:
        # Create label with arrows between elements
        ngram_label = "→".join(str(syl) for syl in ngram_tuple)
        ngrams.append(ngram_label)
        all_syllables.extend(ngram_tuple)
    
    # Calculate dynamic figure size based on label length
    # This ensures the plot area is always at least 50% of the final figure size
    # even when x-tick labels are very long (like in the reference figure)
    max_label_length = max(len(label) for label in ngrams) if ngrams else 0
    avg_label_length = np.mean([len(label) for label in ngrams]) if ngrams else 0
    
    # Handle edge case where no n-grams exist
    if not ngrams:
        max_label_length = 0
        avg_label_length = 0
    
    # Base figure dimensions
    base_width = 15
    base_height = 8
    
    # Determine optimal rotation angle and font size based on label length
    if max_label_length > 50:  # Very long labels
        rotation_angle = 60
        font_size = 8
        char_width_factor = 0.06
        char_height_factor = 0.10
    elif max_label_length > 30:  # Long labels
        rotation_angle = 50
        font_size = 9
        char_width_factor = 0.07
        char_height_factor = 0.11
    else:  # Normal labels
        rotation_angle = 45
        font_size = 10
        char_width_factor = 0.08
        char_height_factor = 0.12
    
    # Calculate required space for rotated labels
    label_width_estimate = max_label_length * char_width_factor
    label_height_estimate = avg_label_length * char_height_factor
    
    # Calculate minimum height needed for labels
    min_height_for_labels = label_height_estimate + 1.2  # 1.2 inch buffer
    
    # Ensure plot area is at least 50% of total figure height
    min_plot_height = 4.0  # minimum plot area height
    total_min_height = max(min_height_for_labels + min_plot_height, min_plot_height * 2)  # 50% rule
    
    # Adjust figure height if needed
    adjusted_height = max(base_height, total_min_height)
    
    # Adjust width if labels are very long
    adjusted_width = max(base_width, label_width_estimate + 8)  # 8 inches for plot area
    
    # Create figure with dynamic size
    fig_freq = plt.figure(figsize=(adjusted_width, adjusted_height))
    ax_freq = fig_freq.add_subplot(111)
    
    bar_width = 0.8
    x_pos = np.arange(top_n)
    probs = [prob for _, prob in sorted_ngrams[:top_n]]
    cum_percent = sum(probs) * 100
    
    # Handle case where there are no n-grams to plot
    if not probs:
        print(f"No n-grams of size {n} found to plot.")
        return
    
    # Draw invisible bars to set up the axis
    ax_freq.bar(x_pos, probs, bar_width, color='none')
    
    # Draw colored segments for each n-gram
    for i, (ngram_tuple, prob) in enumerate(sorted_ngrams[:top_n]):
        segment_width = bar_width / n
        colors = [color_map.get(str(syl), '#D3D3D3') for syl in ngram_tuple]
        
        # Draw each segment
        for j, (syl, color) in enumerate(zip(ngram_tuple, colors)):
            segment_start = x_pos[i] - bar_width/2 + j * segment_width
            rect = mpatches.Rectangle((segment_start, 0), segment_width, prob, 
                                     facecolor=color, edgecolor='none', zorder=2)
            ax_freq.add_patch(rect)
        
        # Draw outline
        rect_outline = mpatches.Rectangle((x_pos[i] - bar_width/2, 0), bar_width, prob, 
                                         fill=False, edgecolor='#B0B0B0', linewidth=2, zorder=3)
        ax_freq.add_patch(rect_outline)
        
        # Add value label
        ax_freq.text(x_pos[i], prob, f'{prob:.3f}', ha='center', va='bottom', zorder=4)
    
    # Add cumulative curve
    cum_probs = np.cumsum(probs)
    ax_cum = ax_freq.twinx()
    ax_cum.plot(x_pos, cum_probs, color='gray', marker='o', linestyle='-', linewidth=2, alpha=0.7, label='Cumulative')
    for i, cp in enumerate(cum_probs):
        ax_cum.text(x_pos[i], cp, f'{cp*100:.0f}%', color='gray', ha='center', va='bottom', fontsize=9)
    ax_cum.set_ylabel('Cumulative Probability', color='gray')
    ax_cum.tick_params(axis='y', labelcolor='gray')
    ax_cum.set_ylim(0, 1.05)
    
    # Set axis properties
    ax_freq.set_xlim(-0.5, top_n - 0.5)
    ax_freq.set_ylim(0, max(probs) * 1.1)
    ax_freq.set_xticks(x_pos)
    ax_freq.set_xticklabels(ngrams, rotation=rotation_angle, ha='right', fontsize=font_size)
    ax_freq.set_ylabel('Probability')
    
    # Create title with n-gram type and number of observed n-grams
    ngram_type = get_ngram_type_name(n)
    obs_text = f" (n={n_observations})" if n_observations is not None else ""
    block_text = f" | Block: {block_name}" if block_name else ""
    ngram_count_text = f" | #Observed n-grams: {num_observed_ngrams}"
    ax_freq.set_title(f'Top {top_n} {ngram_type.capitalize()} Frequencies (Cum {cum_percent:.1f}%){obs_text}{block_text}{ngram_count_text}')
    
    # Create legend with unique syllables
    unique_syllables = sorted(set(all_syllables))
    legend_elements = [mpatches.Patch(color=color_map.get(str(syl), '#D3D3D3'), label=str(syl)) 
                       for syl in unique_syllables]
    ax_freq.legend(handles=legend_elements, title='Syllables', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Use subplots_adjust to ensure plot area is at least 50% of figure height
    # Calculate the required space for labels and ensure plot area gets adequate space
    # Adjust space calculation based on rotation angle
    if rotation_angle >= 60:
        label_space_multiplier = 1.3
    elif rotation_angle >= 50:
        label_space_multiplier = 1.2
    else:
        label_space_multiplier = 1.1
    
    adjusted_label_height = label_height_estimate * label_space_multiplier
    label_space_ratio = min(0.4, adjusted_label_height / adjusted_height)  # Max 40% for labels
    plot_space_ratio = max(0.5, 1.0 - label_space_ratio)  # At least 50% for plot
    
    # Adjust subplot parameters to maintain the 50% rule
    plt.subplots_adjust(
        bottom=label_space_ratio,  # Space for rotated labels
        top=0.9,  # Space for title
        left=0.1,  # Space for y-axis label
        right=0.75  # Space for legend
    )
    if save:
        # Calculate maximum allowed figure size based on DPI
        max_size = 2**16 - 1  # Maximum allowed size in pixels
        dpi = 300  # Default DPI
        max_inches = max_size / dpi
        
        # Get current figure size
        current_size = fig_freq.get_size_inches()
        
        # Scale down if necessary
        if current_size[0] > max_inches or current_size[1] > max_inches:
            scale = min(max_inches / current_size[0], max_inches / current_size[1])
            new_size = (current_size[0] * scale, current_size[1] * scale)
            fig_freq.set_size_inches(new_size)
        
        # Create filename with the new format
        ngram_type = get_ngram_type_name(n)
        block_suffix = f"_{block_name}" if block_name else ""
        # Use consistent filename format
        filename = f"{ngram_type.replace('-', '')}_frequency_n{n_observations}{block_suffix}.png"
        
        plt.savefig(os.path.join(output_path, filename),
                   dpi=dpi, bbox_inches='tight')
    plt.close()

def permutation_pvals(ngram_counter, total_ngrams, label_sequences, n, n_perm=1000, seed=42):
    """Compute permutation p-values for n-gram frequencies."""
    np.random.seed(seed)
    observed = compute_ngram_frequencies(ngram_counter, total_ngrams)
    null_distributions = defaultdict(list)
    for _ in tqdm(range(n_perm), desc=f"Permuting for {n}-gram p-values"):
        for labels in label_sequences:
            shuffled = np.random.permutation(labels)
            ngrams = extract_ngrams(shuffled, n)
            for ngram in ngrams:
                null_distributions[ngram].append(1)
    pvals = {}
    for ngram, obs_freq in observed.items():
        null_freqs = np.array(null_distributions.get(ngram, [0]*n_perm)) / total_ngrams
        pvals[ngram] = (np.sum(null_freqs >= obs_freq) + 1) / (n_perm + 1)
    return pvals

@tictoc
def main():
    parser = argparse.ArgumentParser(description='Analyze syllable n-grams of any size from datasets using new metadata approach.')
    parser.add_argument('--labels-path', type=str, required=True, help='Path to the labels dataset CSV file')
    parser.add_argument('--label-column', type=str, required=True, help='Name of the column containing the labels')
    parser.add_argument('--metadata-path', type=str, required=True, help='Path to the new metadata dataset CSV file')
    parser.add_argument('--output-dir', type=str, required=False, default=None, help='Path to the output directory')
    parser.add_argument('--colour-map-object', type=str, required=False, default=None, help='Path to the colour map object')
    parser.add_argument('--fps', type=float, required=False, default=10.0, help='Frames per second for durations (default: 10.0)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--n-perm', type=int, required=False, default=0, help='Number of permutations for p-value estimation (0=skip)')
    parser.add_argument('--do-permutation-test', action='store_true', help='If set, perform permutation test for n-gram frequencies')
    parser.add_argument('--exclude-block', type=str, required=False, default=None, help='Name of the block to exclude when creating "all_but_excluded" group (optional)')
    parser.add_argument('--failed-trials-path', type=str, required=False, default=None, help='Path to CSV file containing metadata of failed trials (must have ppid, session, trial columns)')
    parser.add_argument('--ngram-sizes', type=int, nargs='+', required=False, default=[2, 3], help='N-gram sizes to analyze (default: 2 3 for bigrams and trigrams)')
    args = parser.parse_args()

    # Validate n-gram sizes
    for n in args.ngram_sizes:
        if n < 1:
            raise ValueError(f"Invalid n-gram size: {n}. Must be >= 1.")
    
    print_color(f"Analyzing n-gram sizes: {args.ngram_sizes}", "GREEN")

    if args.output_dir is None:
        output_dir = os.path.dirname(args.metadata_path)
    else:
        output_dir = args.output_dir
    output_dir = os.path.join(output_dir, 'syllable_ngrams_output')
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
    
    # Filter out failed trials from dataset1 if failed trials data is provided
    if failed_trials_data is not None:
        print_color(f"\nFiltering out failed trials from labels dataset...", bold_color.BLUE)
        before_count = len(dataset1)
        
        # Create a mask to exclude failed trials
        failed_mask = dataset1.set_index(['ppid', 'session', 'trial']).index.isin(
            failed_trials_data.set_index(['ppid', 'session', 'trial']).index
        )
        
        if failed_mask.sum() > 0:
            dataset1 = dataset1[~failed_mask]
            after_count = len(dataset1)
            print_color(f"Removed {before_count - after_count} failed trials from labels dataset", bold_color.YELLOW)
            print_color(f"Labels dataset now contains {after_count} rows (was {before_count})", bold_color.GREEN)
        else:
            print_color("No failed trials found in the labels dataset", bold_color.YELLOW)
    
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
    
    print_color(f"Using columns: {threat_column}, {threat_action_column}, {episode_set_name_column}", "GREEN")
    
    unique_syllables = sorted(dataset1[args.label_column].unique())
    le = LabelEncoder()
    le.fit(list(unique_syllables))
    color_map = create_color_map(le, colour_map_arg=args.colour_map_object)

    # Prepare label sequences for permutation
    label_sequences = [trial[args.label_column].values for _, trial in dataset1.groupby(['ppid', 'session', 'trial'])]
    
    # Count number of unique observations (ppid-session-trial combinations)
    n_observations = len(dataset1.groupby(['ppid', 'session', 'trial']))

    pastel_grey = '#B0B0B0'

    # Process all trials together
    print_color("\nProcessing all trials together:", "CYAN")
    
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
                print_color(f"Excluding '{block_name}' block from all_trials analysis", "YELLOW")
        
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
            print_color(f"All trials analysis now contains {len(all_trials_data.groupby(['ppid', 'session', 'trial']))} trials (excluded {len(excluded_trials)} trials from {blocks_to_exclude} blocks)", "GREEN")
    
    for n in args.ngram_sizes:
        ngram_counter, ngram_durations = count_ngrams_in_trials(all_trials_data, args.label_column, n)
        total_ngrams = sum(ngram_counter.values())
        ngram_freq = compute_ngram_frequencies(ngram_counter, total_ngrams)
        ngram_stats = compute_ngram_duration_stats(ngram_durations, fps=args.fps)
        
        # Create filename with the new format
        ngram_type = get_ngram_type_name(n)
        base_filename = f"{ngram_type.replace('-', '')}_frequency_n{n_observations}"
        
        # Save CSVs with the new naming convention
        freq_df = pd.DataFrame([
            {'ngram': ngram, 'freq': freq, 'count': ngram_counter[ngram], **ngram_stats.get(ngram, {})}
            for ngram, freq in ngram_freq.items()
        ])
        freq_df.to_csv(os.path.join(output_dir, f'{base_filename}.csv'), index=False)
        
        # Permutation p-values (optional)
        ngram_pvals = None
        if args.do_permutation_test and args.n_perm > 0:
            ngram_pvals = permutation_pvals(ngram_counter, total_ngrams, label_sequences, n, n_perm=args.n_perm)
        
        # Plot
        if not args.no_plots:
            ngram_matrix_plot(ngram_freq, ngram_pvals, unique_syllables, n,
                             title=f'Syllable {n}-gram Matrix',
                             output_path=output_dir, color_map=color_map, save=True,
                             n_observations=n_observations)

    # Process each block separately based on episode_set_name2
    analysis_groups = {'all_trials': all_trials_data}
    
    if episode_set_name_column in dataset2.columns:
        print_color(f"\nFound {len(unique_blocks)} unique blocks in {episode_set_name_column}: {list(unique_blocks)}", "GREEN")
        
        for block_name in unique_blocks:
            print_color(f"\nProcessing block: {block_name}", "CYAN")
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
                print_color(f"Created 'all_but_{args.exclude_block}' group", "GREEN")
            else:
                print_color(f"Warning: Specified exclude block '{args.exclude_block}' not found in available blocks: {list(unique_blocks)}", "YELLOW")
    else:
        print_color(f"Warning: Block column '{episode_set_name_column}' not found in metadata. Only 'all_trials' will be analyzed.", "YELLOW")

    # Now run n-gram analysis for each group
    for group_name, group_data in analysis_groups.items():
        if group_data.empty:
            print_color(f"Warning: No data found for block/group {group_name}", "YELLOW")
            continue
        n_observations_group = len(group_data.groupby(['ppid', 'session', 'trial']))
        group_label_sequences = [trial[args.label_column].values for _, trial in group_data.groupby(['ppid', 'session', 'trial'])]
        for n in args.ngram_sizes:
            ngram_counter, ngram_durations = count_ngrams_in_trials(group_data, args.label_column, n)
            total_ngrams = sum(ngram_counter.values())
            ngram_freq = compute_ngram_frequencies(ngram_counter, total_ngrams)
            ngram_stats = compute_ngram_duration_stats(ngram_durations, fps=args.fps)
            ngram_type = get_ngram_type_name(n)
            base_filename = f"{ngram_type.replace('-', '')}_frequency_n{n_observations_group}_{group_name}"
            freq_df = pd.DataFrame([
                {'ngram': ngram, 'freq': freq, 'count': ngram_counter[ngram], **ngram_stats.get(ngram, {})}
                for ngram, freq in ngram_freq.items()
            ])
            freq_df.to_csv(os.path.join(output_dir, f'{base_filename}.csv'), index=False)
            ngram_pvals = None
            if args.do_permutation_test and args.n_perm > 0:
                ngram_pvals = permutation_pvals(ngram_counter, total_ngrams, group_label_sequences, n, n_perm=args.n_perm)
            if not args.no_plots:
                ngram_matrix_plot(ngram_freq, ngram_pvals, unique_syllables, n,
                                 title=f'Syllable {n}-gram Matrix',
                                 output_path=output_dir, color_map=color_map, save=True,
                                 n_observations=n_observations_group, block_name=group_name)

    print_color("\nSyllable n-gram analysis completed successfully!", bold_color.BLUE)
    print_color(f"\nThe data can be found in: ".capitalize() + f"{output_dir}", bold_color.BLUE)

if __name__ == "__main__":
    print_welcome_message()
    main() 