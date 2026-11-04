"""
Module for analyzing syllable durations from time-series data.

This script calculates the duration of consecutive syllables from a CSV file,
groups the data by experimental blocks, and generates a comprehensive set of
statistical summaries and plots for each group.

Author: uSerratos (Reviewed and enhanced by Gemini)
Date: June 2024
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import scipy.signal
import pickle
from natsort import natsorted
import scipy.stats
from scipy.stats import gaussian_kde

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Helper function to print colored text to the console for better readability
def print_color(text, color="WHITE"):
    """Prints text in a specified color."""
    colors = {
        "BLUE": "\033[94m", "CYAN": "\033[96m", "GREEN": "\033[92m",
        "YELLOW": "\033[93m", "RED": "\033[91m", "ENDC": "\033[0m",
        "BOLD": "\033[1m", "UNDERLINE": "\033[4m", "WHITE": "\033[97m"
    }
    color_code = colors.get(color.upper(), colors["WHITE"])
    print(f"{color_code}{text}{colors['ENDC']}")

def tictoc(func):
    """Decorator to time function execution."""
    def inner(*args, **kwargs):
        t1 = time.time()
        print_color(f"Function: {func.__name__} started at {time.strftime('%H:%M:%S', time.gmtime(t1))}", "GREEN")
        func_out = func(*args, **kwargs)
        t2 = time.time()
        exec_time = time.strftime("%H:%M:%S", time.gmtime(t2 - t1))
        print_color(f"Function: {func.__name__} finished. Total runtime: {exec_time}", "GREEN")
        return func_out
    return inner

def create_color_map(unique_labels, colour_map_arg=None):
    """Creates a dictionary mapping syllable labels to colors."""
    if isinstance(colour_map_arg, str) and colour_map_arg.endswith('.pkl'):
        try:
            with open(colour_map_arg, 'rb') as f: return pickle.load(f)
        except Exception as e:
            print_color(f"Warning: Could not load color map from {colour_map_arg}: {e}", "YELLOW")

    if isinstance(colour_map_arg, dict): return colour_map_arg

    try:
        color_map = plt.get_cmap(colour_map_arg or 'tab20')
    except (ValueError, TypeError):
        print_color(f"Warning: Colormap '{colour_map_arg}' not found. Falling back to 'tab20'.", "YELLOW")
        color_map = plt.get_cmap('tab20')

    colors = color_map(np.linspace(0, 1, len(unique_labels)))
    color_map_dict = {str(label): color for label, color in zip(unique_labels, colors)}
    color_map_dict['other'] = '#D3D3D3'; color_map_dict['-1'] = '#D3D3D3';
    return color_map_dict

def find_consecutive_syllables(labels):
    """Finds consecutive sequences of the same syllable using a vectorized approach."""
    if not isinstance(labels, (list, np.ndarray)) or len(labels) == 0: return []
    labels = np.asarray(labels)
    change_indices = np.where(labels[:-1] != labels[1:])[0] + 1
    start_frames = np.concatenate(([0], change_indices))
    sequence_labels = labels[start_frames]
    durations = np.diff(np.concatenate((start_frames, [len(labels)])))
    return list(zip(sequence_labels, start_frames, durations))

def calculate_duration_statistics(durations):
    """Calculates summary statistics for a list of durations."""
    if len(durations) == 0:
        return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan,
                'max': np.nan, 'q25': np.nan, 'q75': np.nan, 'iqr': np.nan}
    return {'mean': np.mean(durations), 'median': np.median(durations), 'std': np.std(durations),
            'min': np.min(durations), 'max': np.max(durations), 'q25': np.percentile(durations, 25),
            'q75': np.percentile(durations, 75), 'iqr': np.percentile(durations, 75) - np.percentile(durations, 25)}

# === PLOTTING FUNCTIONS ===

def plot_duration_distributions(duration_data, output_dir, color_map, experimental_block, n_obs):
    """Create violin and box plots for syllable durations."""
    plot_dir = os.path.join(output_dir, 'duration_distributions'); os.makedirs(plot_dir, exist_ok=True)
    unique_syllables = natsorted(duration_data['syllable'].unique())
    
    # Calculate dynamic figure size based on number of syllables and label length
    n_syllables = len(unique_syllables)
    max_label_length = max(len(str(s)) for s in unique_syllables)
    
    # Base figure size
    base_width = max(15, n_syllables * 0.8)  # At least 0.8 units per syllable
    base_height = 8
    
    # Adjust height based on label length to prevent cutting
    label_height_factor = max(1.0, max_label_length / 15)  # Scale based on max label length
    fig_height = base_height * label_height_factor
    
    # Dynamic font size for x-axis labels based on label length
    if max_label_length > 20:
        x_label_fontsize = 8
    elif max_label_length > 15:
        x_label_fontsize = 9
    else:
        x_label_fontsize = 10

    # Violin plot
    plt.figure(figsize=(base_width, fig_height))
    sns.violinplot(data=duration_data, x='syllable', y='duration_seconds', palette=color_map, order=unique_syllables, cut=0)
    plt.title(f'Syllable Duration Distribution (Violin) - {experimental_block} (n={n_obs})', fontsize=12, pad=20)
    plt.ylabel('Duration (seconds)', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=x_label_fontsize)
    plt.yticks(fontsize=10)
    
    # Calculate bottom margin based on label length, but ensure it's reasonable
    bottom_margin = max(0.15, min(0.4, 0.1 + (max_label_length - 10) * 0.01))
    plt.subplots_adjust(bottom=bottom_margin)
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(plot_dir, f'duration_violin_{experimental_block}_n{n_obs}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Box plot
    plt.figure(figsize=(base_width, fig_height))
    sns.boxplot(data=duration_data, x='syllable', y='duration_seconds', palette=color_map, order=unique_syllables)
    plt.title(f'Syllable Duration Distribution (Box) - {experimental_block} (n={n_obs})', fontsize=12, pad=20)
    plt.ylabel('Duration (seconds)', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=x_label_fontsize)
    plt.yticks(fontsize=10)
    
    # Calculate bottom margin based on label length, but ensure it's reasonable
    bottom_margin = max(0.15, min(0.4, 0.1 + (max_label_length - 10) * 0.01))
    plt.subplots_adjust(bottom=bottom_margin)
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(plot_dir, f'duration_box_{experimental_block}_n{n_obs}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_duration_heatmap(duration_stats, output_dir, experimental_block):
    """Create a heatmap of duration statistics."""
    plot_dir = os.path.join(output_dir, 'duration_statistics'); os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(duration_stats, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title(f'Syllable Duration Statistics Heatmap - {experimental_block}'); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'duration_heatmap_{experimental_block}.png'), dpi=300); plt.close()

def plot_mean_durations(duration_data, output_dir, color_map, experimental_block, n_obs, global_max_mean=None):
    """Creates a bar plot of mean syllable durations with a consistent Y-axis."""
    plot_dir = os.path.join(output_dir, 'mean_duration_plots'); os.makedirs(plot_dir, exist_ok=True)
    mean_durations = duration_data.groupby('syllable')['duration_seconds'].mean()
    mean_durations.to_csv(os.path.join(plot_dir, f'mean_durations_{experimental_block}_n{n_obs}.csv'))
    
    unique_syllables = natsorted(mean_durations.index)
    mean_durations = mean_durations.reindex(unique_syllables)
    
    # Calculate dynamic figure size based on number of syllables and label length
    n_syllables = len(unique_syllables)
    max_label_length = max(len(str(s)) for s in unique_syllables)
    
    # Base figure size with minimum 50% for bars
    base_width = max(15, n_syllables * 0.8)  # At least 0.8 units per syllable
    base_height = 8
    
    # Adjust height based on label length to prevent cutting
    label_height_factor = max(1.0, max_label_length / 15)  # Scale based on max label length
    fig_height = base_height * label_height_factor
    
    # Ensure bars take up at least 50% of figure height
    min_bar_height = fig_height * 0.5
    if base_height < min_bar_height:
        fig_height = max(fig_height, min_bar_height * 2)
    
    plt.figure(figsize=(base_width, fig_height))
    
    # Create the bar plot
    bars = plt.bar(mean_durations.index, mean_durations.values, color=[color_map.get(str(s), '#D3D3D3') for s in mean_durations.index])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # Dynamic font size for x-axis labels based on label length
    if max_label_length > 20:
        x_label_fontsize = 8
    elif max_label_length > 15:
        x_label_fontsize = 9
    else:
        x_label_fontsize = 10
    
    plt.title(f'Mean Syllable Durations - {experimental_block} (n={n_obs})', fontsize=12, pad=20)
    plt.xlabel('Syllable', fontsize=11)
    plt.ylabel('Mean Duration (seconds)', fontsize=11)
    
    # Rotate x-axis labels and adjust positioning
    plt.xticks(rotation=45, ha='right', fontsize=x_label_fontsize)
    plt.yticks(fontsize=10)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits with padding to prevent bar cutting
    # Always ensure at least 5% padding above the maximum value in this specific plot
    actual_max = mean_durations.max()
    y_max = max(actual_max * 1.05, actual_max + 0.1)  # At least 5% padding or 0.1 seconds
    
    # If global_max_mean is provided, use the larger of the two for consistency
    if global_max_mean: 
        y_max = max(y_max, global_max_mean)
    
    plt.ylim(0, y_max)
    
    # Adjust layout with extra padding for rotated labels
    # Calculate bottom margin based on label length, but ensure it's reasonable
    bottom_margin = max(0.15, min(0.4, 0.1 + (max_label_length - 10) * 0.01))
    
    # Apply layout adjustments before tight_layout
    plt.subplots_adjust(bottom=bottom_margin)
    plt.tight_layout(pad=3.0)
    
    plt.savefig(os.path.join(plot_dir, f'mean_durations_{experimental_block}_n{n_obs}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_syllable_duration_distribution_with_cumulative(duration_df, output_dir, experimental_block, global_xmax=None):
    """Produces three plots per block: histogram only, KDE only, and histogram with KDE overlay, all with CDF."""
    plot_dir = os.path.join(output_dir, 'syllable_distribution_plots'); os.makedirs(plot_dir, exist_ok=True)
    syllable_durations = duration_df['duration_seconds'].dropna().values
    if len(syllable_durations) == 0: return

    pd.DataFrame({'duration_seconds': syllable_durations}).to_csv(os.path.join(plot_dir, f'distribution_data_{experimental_block}.csv'), index=False)
    
    max_duration = syllable_durations.max()
    n_bins = int(np.ceil(max_duration * 10))
    mean_duration = np.mean(syllable_durations)

    # Prepare for CDF
    sorted_durations = np.sort(syllable_durations)
    cumulative_prob = np.arange(1, len(sorted_durations)+1) / float(len(sorted_durations))
    shape, loc, scale = scipy.stats.lognorm.fit(syllable_durations, floc=0)
    x_fit = np.linspace(sorted_durations.min(), sorted_durations.max(), 500)
    cdf_fit = scipy.stats.lognorm.cdf(x_fit, shape, loc, scale)

    # Find the 95% point on the empirical cumulative
    idx_95 = np.searchsorted(cumulative_prob, 0.95)
    if idx_95 < len(sorted_durations):
        dur_95 = sorted_durations[idx_95]
        prob_95 = cumulative_prob[idx_95]
    else:
        dur_95 = sorted_durations[-1]
        prob_95 = cumulative_prob[-1]

    # 1. Histogram only
    fig, ax1 = plt.subplots(figsize=(12, 7))
    n, bins, patches = ax1.hist(syllable_durations, bins=n_bins, color='skyblue', edgecolor='navy', alpha=0.7, density=True, label='Probability Density')
    ax1.set_xlabel('Syllable Duration (seconds)'); ax1.set_ylabel('Probability Density', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue'); ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axvline(mean_duration, color='red', linestyle='--', label=f'Mean: {mean_duration:.2f}s')
    if global_xmax is not None:
        ax1.set_xlim([0, global_xmax])
    ax2 = ax1.twinx()
    ax2.plot(sorted_durations, cumulative_prob, color='gray', lw=2, linestyle='--', label='Empirical Cumulative')
    ax2.plot(x_fit, cdf_fit, color='orange', lw=2, label='Log-normal CDF (fit)')
    # Add green vertical line and marker at 95%
    ax2.axvline(dur_95, color='green', linestyle='-', lw=2, label='95% Cumulative')
    ax2.plot(dur_95, prob_95, 'o', color='green')
    ax2.set_ylabel('Cumulative Probability', color='orange'); ax2.tick_params(axis='y', labelcolor='orange'); ax2.set_ylim(0, 1)
    if global_xmax is not None:
        ax2.set_xlim([0, global_xmax])
    plt.title(f'Histogram: Syllable Duration Distribution - {experimental_block}'); fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    plt.savefig(os.path.join(plot_dir, f'distribution_hist_{experimental_block}.png'), dpi=300); plt.close()

    # 3. Histogram with KDE overlay
    fig, ax1 = plt.subplots(figsize=(12, 7))
    n, bins, patches = ax1.hist(syllable_durations, bins=n_bins, color='skyblue', edgecolor='navy', alpha=0.5, density=True, label='Probability Density')
    try:
        import seaborn as sns
        sns.kdeplot(syllable_durations, ax=ax1, color='blue', fill=True, alpha=0.3, label='KDE')
    except ImportError:
        density = scipy.stats.gaussian_kde(syllable_durations)
        x_kde = np.linspace(0, max_duration, 500)
        ax1.plot(x_kde, density(x_kde), color='blue', label='KDE')
    ax1.set_xlabel('Syllable Duration (seconds)'); ax1.set_ylabel('Probability Density', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue'); ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axvline(mean_duration, color='red', linestyle='--', label=f'Mean: {mean_duration:.2f}s')
    if global_xmax is not None:
        ax1.set_xlim([0, global_xmax])
    ax2 = ax1.twinx()
    ax2.plot(sorted_durations, cumulative_prob, color='gray', lw=2, linestyle='--', label='Empirical Cumulative')
    ax2.plot(x_fit, cdf_fit, color='orange', lw=2, label='Log-normal CDF (fit)')
    # Add green vertical line and marker at 95%
    ax2.axvline(dur_95, color='green', linestyle='-', lw=2, label='95% Cumulative')
    ax2.plot(dur_95, prob_95, 'o', color='green')
    ax2.set_ylabel('Cumulative Probability', color='orange'); ax2.tick_params(axis='y', labelcolor='orange'); ax2.set_ylim(0, 1)
    if global_xmax is not None:
        ax2.set_xlim([0, global_xmax])
    plt.title(f'Histogram + KDE: Syllable Duration Distribution - {experimental_block}'); fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    plt.savefig(os.path.join(plot_dir, f'distribution_hist_kde_{experimental_block}.png'), dpi=300); plt.close()

def plot_block_duration_distribution(duration_df, output_dir, experimental_block, bin_edges=None):
    """Plots the distribution of TOTAL TRIAL DURATIONS within a block."""
    plot_dir = os.path.join(output_dir, 'trial_duration_plots'); os.makedirs(plot_dir, exist_ok=True)
    trial_durations = duration_df.groupby(['ppid', 'session', 'trial'])['duration_seconds'].sum().values
    if len(trial_durations) == 0: return

    plt.figure(figsize=(10, 6))
    plt.hist(trial_durations, bins=bin_edges if bin_edges is not None else 50, color='mediumpurple', edgecolor='black', alpha=0.7)
    plt.xlabel('Total Trial Duration (seconds)'); plt.ylabel('Number of Trials')
    plt.title(f'Distribution of Trial Durations - {experimental_block}')
    plt.grid(True, alpha=0.3, linestyle='--'); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'trial_duration_dist_{experimental_block}.png'), dpi=300); plt.close()

def plot_block_duration_spectrum(duration_df, output_dir, experimental_block, fs=10):
    """Plots the power spectrum of trial durations within a block."""
    plot_dir = os.path.join(output_dir, 'trial_duration_plots'); os.makedirs(plot_dir, exist_ok=True)
    trial_durations = duration_df.groupby(['ppid', 'session', 'trial'])['duration_seconds'].sum().values
    if len(trial_durations) < 4: return
    f, Pxx = scipy.signal.welch(trial_durations, fs=fs, nperseg=min(256, len(trial_durations)))
    plt.figure(figsize=(8, 4))
    plt.plot(f, Pxx, color='k', lw=2); plt.fill_between(f, Pxx, color='k', alpha=0.1)
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Power Spectral Density')
    plt.title(f'Trial Duration Spectrum - {experimental_block}'); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'trial_duration_spectrum_{experimental_block}.png'), dpi=300); plt.close()

def plot_block_duration_spaghetti(duration_df, output_dir, experimental_block, time_unit='s'):
    """
    Plots epoch duration distribution for each subject (gray), and the mean (black).
    """
    plot_dir = os.path.join(output_dir, 'epoch_duration_spaghetti_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Always use seconds
    duration_df = duration_df.copy()
    duration_df['duration'] = duration_df['duration_seconds']
    x_label = 'Epoch Duration (s)'

    # Group by subject
    subjects = duration_df['ppid'].unique()
    mean_val = duration_df['duration'].mean()
    std_val = duration_df['duration'].std()
    min_val = duration_df['duration'].min()
    max_val = duration_df['duration'].max()
    x_max = mean_val + 2 * std_val
    x_grid = np.linspace(0, x_max, 500)
    all_densities = []
    first_grey = True

    plt.figure(figsize=(8, 6))
    for subj in subjects:
        subj_durations = duration_df[duration_df['ppid'] == subj]['duration'].dropna().values
        if len(subj_durations) > 1:
            kde = gaussian_kde(subj_durations)
            density = kde(x_grid)
            all_densities.append(density)
            if first_grey:
                plt.plot(x_grid, density, color='gray', alpha=0.3, lw=1, label='Individual participants')
                first_grey = False
            else:
                plt.plot(x_grid, density, color='gray', alpha=0.3, lw=1)

    # Mean density
    if all_densities:
        mean_density = np.mean(all_densities, axis=0)
        plt.plot(x_grid, mean_density, color='black', lw=2, label='Mean')

    # Mean, SD, min, max annotation
    plt.axvline(mean_val, color='black', linestyle='--', lw=1)
    plt.xlabel(x_label)
    plt.ylabel('Proportion of Epochs')
    plt.title(f'Epoch Duration Distribution - {experimental_block}')
    plt.xlim([0, x_max])
    plt.legend(title=f"Mean = {mean_val:.2f} s\nSD = {std_val:.2f} s\nMin = {min_val:.2f} s\nMax = {max_val:.2f} s")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'epoch_duration_spaghetti_{experimental_block}.png'), dpi=300)
    plt.close()

def plot_syllable_duration_spaghetti(duration_df, output_dir, experimental_block, time_unit='s'):
    """
    Plots syllable duration distribution for each participant (gray), and the mean (black).
    """
    plot_dir = os.path.join(output_dir, 'syllable_duration_spaghetti_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Convert to ms if needed
    if time_unit == 'ms':
        duration_df = duration_df.copy()
        duration_df['duration'] = duration_df['duration_seconds'] * 1000
        x_label = 'Syllable Duration (ms)'
    else:
        duration_df['duration'] = duration_df['duration_seconds']
        x_label = 'Syllable Duration (s)'

    # Group by participant
    participants = duration_df['ppid'].unique()
    mean_val = duration_df['duration'].mean()
    std_val = duration_df['duration'].std()
    min_val = duration_df['duration'].min()
    max_val = duration_df['duration'].max()
    x_max = mean_val + 2 * std_val
    x_grid = np.linspace(0, x_max, 500)
    all_densities = []
    first_grey = True

    plt.figure(figsize=(8, 6))
    for ppid in participants:
        ppid_durations = duration_df[duration_df['ppid'] == ppid]['duration'].dropna().values
        if len(ppid_durations) > 1:
            kde = gaussian_kde(ppid_durations)
            density = kde(x_grid)
            all_densities.append(density)
            if first_grey:
                plt.plot(x_grid, density, color='gray', alpha=0.3, lw=1, label='Individual participants')
                first_grey = False
            else:
                plt.plot(x_grid, density, color='gray', alpha=0.3, lw=1)

    # Mean density
    if all_densities:
        mean_density = np.mean(all_densities, axis=0)
        plt.plot(x_grid, mean_density, color='black', lw=2, label='Mean')

    # Mean, SD, min, max annotation
    plt.axvline(mean_val, color='black', linestyle='--', lw=1)
    plt.xlabel(x_label)
    plt.ylabel('Proportion of Syllables')
    plt.title(f'Syllable Duration Distribution - {experimental_block}')
    plt.xlim([0, x_max])
    plt.legend(title=f"Mean = {mean_val:.2f} {time_unit}\nSD = {std_val:.2f} {time_unit}\nMin = {min_val:.2f} {time_unit}\nMax = {max_val:.2f} {time_unit}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'syllable_duration_spaghetti_{experimental_block}.png'), dpi=300)
    plt.close()

@tictoc
def main():
    parser = argparse.ArgumentParser(description='Analyze syllable durations from time-series data using new metadata approach.')
    parser.add_argument('--labels-path', type=str, help='Path to the labels dataset CSV file.')
    parser.add_argument('--metadata-path', type=str, help='Path to the metadata CSV file with block info (must include episode_set_name2, threat2, etc).')
    parser.add_argument('--label-column', type=str, default='syllable', help='Name of the column with syllable labels.')
    parser.add_argument('--output-dir', type=str, default='syllable_durations_output_light', help='Directory to save output files.')
    parser.add_argument('--fps', type=float, default=10.0, help='Data collection rate in frames per second.')
    parser.add_argument('--colour-map-object', type=str, help='Path to a pickled color map dictionary.')
    parser.add_argument('--failed-trials-path', type=str, required=False, default=None,
                      help='Path to CSV file containing metadata of failed trials (must have ppid, session, trial columns)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation.')
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.join(args.output_dir, 'syllable_duration_output_light')
    os.makedirs(output_dir, exist_ok=True)
    raw_data_dir = os.path.join(output_dir, 'raw_duration_data'); os.makedirs(raw_data_dir, exist_ok=True)
    stats_dir = os.path.join(output_dir, 'duration_statistics'); os.makedirs(stats_dir, exist_ok=True)

    print_color("Loading datasets...", "CYAN")
    labels_df = pd.read_csv(args.labels_path)
    metadata_df = pd.read_csv(args.metadata_path)

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
                print_color(f"Loaded failed trials data with {len(failed_trials_data)} trials", "GREEN")
                print_color(f"Using columns: {ppid_col}->ppid, {session_col}->session, {trial_col}->trial", "BLUE")
            else:
                missing_cols = []
                if not ppid_col:
                    missing_cols.append("ppid (or ppid_vr, ppid_coda)")
                if not session_col:
                    missing_cols.append("session (or session_vr, session_coda)")
                if not trial_col:
                    missing_cols.append("trial (or trial_vr, trial_coda)")
                print_color(f"Warning: Failed trials CSV missing required columns: {missing_cols}. Failed trials will be ignored.", "YELLOW")
                failed_trials_data = None
        except Exception as e:
            print_color(f"Warning: Could not load failed trials CSV: {str(e)}. Failed trials will be ignored.", "YELLOW")
            failed_trials_data = None

    # --- Required columns check ---
    required_columns = ['end_state', 'reward', 'instructed_action', 'day_time', 'scenario', 'scenario_id', 'threat', 'visibility', 'threat_distance', 'threat_action', 'weapons', 'safehouse', 'threat2', 'threat_action2', 'fixed_order', 'episode_set_name2']
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in metadata file: {missing_columns}")

    block_column = 'episode_set_name2'
    all_unique_syllables = natsorted(list(labels_df[args.label_column].unique()))
    color_map = create_color_map(all_unique_syllables, colour_map_arg=args.colour_map_object)

    # --- Exclude 'instructed' and 'failed' blocks from all_trials ---
    all_trials_data = labels_df.copy()
    if block_column in metadata_df.columns:
        unique_blocks = metadata_df[block_column].unique()
        blocks_to_exclude = []
        for block_name in ['instructed', 'failed']:
            if block_name in unique_blocks:
                blocks_to_exclude.append(block_name)
                print_color(f"Excluding '{block_name}' block from all_trials analysis", "YELLOW")
        if blocks_to_exclude:
            excluded_trials = set()
            for block_name in blocks_to_exclude:
                block_trials = set(tuple(x) for x in metadata_df[metadata_df[block_column] == block_name][['ppid', 'session', 'trial']].values)
                excluded_trials.update(block_trials)
            mask = ~labels_df[['ppid', 'session', 'trial']].apply(tuple, axis=1).isin(excluded_trials)
            all_trials_data = labels_df[mask].copy()
            print_color(f"All trials analysis now contains {len(all_trials_data.groupby(['ppid', 'session', 'trial']))} trials (excluded {len(excluded_trials)} trials from {blocks_to_exclude} blocks)", "GREEN")

    # --- Prepare Data Groups for Analysis ---
    analysis_groups = {'all_trials': all_trials_data}
    if block_column in metadata_df.columns:
        for block_name in unique_blocks:
            print_color(f"Found block: {block_name}", "CYAN")
            block_trials = metadata_df[metadata_df[block_column] == block_name][['ppid', 'session', 'trial']]
            analysis_groups[block_name] = pd.merge(labels_df, block_trials, on=['ppid', 'session', 'trial'], how='inner')
    else:
        print_color(f"Warning: Block column '{block_column}' not found in metadata. Only 'all_trials' will be analyzed.", "YELLOW")

    # --- Filter out failed trials if provided ---
    if failed_trials_data is not None:
        print_color("\nFiltering out failed trials from analysis...", "BLUE")
        
        # Create a set of failed trial identifiers for efficient lookup using pandas index
        failed_trial_index = failed_trials_data.set_index(['ppid', 'session', 'trial']).index
        
        # Filter out failed trials from all analysis groups
        for group_name, group_data in analysis_groups.items():
            if not group_data.empty:
                # Create mask to exclude failed trials using pandas index operations
                group_index = group_data.set_index(['ppid', 'session', 'trial']).index
                failed_mask = group_index.isin(failed_trial_index)
                
                if failed_mask.sum() > 0:
                    print_color(f"Removing {failed_mask.sum()} failed trials from {group_name} group", "YELLOW")
                    analysis_groups[group_name] = group_data[~failed_mask].copy()
                else:
                    print_color(f"No failed trials found in {group_name} group", "GREEN")
        
        # Also filter the original labels_df for global calculations
        labels_index = labels_df.set_index(['ppid', 'session', 'trial']).index
        labels_failed_mask = labels_index.isin(failed_trial_index)
        if labels_failed_mask.sum() > 0:
            print_color(f"Removing {labels_failed_mask.sum()} failed trials from global calculations", "YELLOW")
            labels_df = labels_df[~labels_failed_mask].copy()
        
        print_color(f"Failed trials filtering completed. Total failed trials excluded: {len(failed_trial_index)}", "GREEN")
    else:
        print_color("No failed trials file provided. Using all available data.", "BLUE")

    # --- Analysis Setup ---
    # Compute all syllable durations in seconds for global max mean
    all_durations_for_global = []
    for (ppid, session, trial), group in labels_df.groupby(['ppid', 'session', 'trial']):
        labels = group[args.label_column].values
        for syllable, start_frame, duration in find_consecutive_syllables(labels):
            all_durations_for_global.append({'syllable': str(syllable), 'duration_frames': duration})
    all_durations_df = pd.DataFrame(all_durations_for_global)
    all_durations_df['duration_seconds'] = all_durations_df['duration_frames'] / args.fps
    global_max_mean = all_durations_df.groupby('syllable')['duration_seconds'].mean().max() * 1.15
    all_trial_durations = labels_df.groupby(['ppid', 'session', 'trial']).size().values
    global_mean = labels_df.groupby(args.label_column).size().mean()
    global_std = labels_df.groupby(args.label_column).size().std()
    global_xmax = global_mean + 3 * global_std
    global_bin_edges = np.histogram_bin_edges(all_trial_durations, bins='auto')

    # --- Run Analysis on Each Group ---
    for block, data in analysis_groups.items():
        print_color(f"\n--- Analyzing block: {block} ({len(data)} syllable events) ---", "BOLD")
        if data.empty: print_color("No data in this block. Skipping.", "YELLOW"); continue
        all_durations = []
        for (ppid, session, trial), group in data.groupby(['ppid', 'session', 'trial']):
            labels = group[args.label_column].values
            for syllable, start_frame, duration in find_consecutive_syllables(labels):
                all_durations.append({'ppid': ppid, 'session': session, 'trial': trial, 'syllable': str(syllable), 
                                      'start_frame': start_frame, 'duration_frames': duration})
        if not all_durations: print_color("No syllables found. Exiting.", "RED"); continue
        duration_df = pd.DataFrame(all_durations)
        duration_df['duration_seconds'] = duration_df['duration_frames'] / args.fps
        n_obs = duration_df.drop_duplicates(['ppid', 'session', 'trial']).shape[0]
        duration_df.to_csv(os.path.join(raw_data_dir, f'durations_{block}_n{n_obs}.csv'), index=False)
        stats_by_syllable = {s: calculate_duration_statistics(duration_df[duration_df['syllable'] == s]['duration_seconds'].values) for s in all_unique_syllables}
        stats_df = pd.DataFrame(stats_by_syllable).T.reindex(all_unique_syllables)
        stats_df.to_csv(os.path.join(stats_dir, f'duration_stats_{block}_n{n_obs}.csv'))
        if not args.no_plots:
            plot_duration_distributions(duration_df, output_dir, color_map, block, n_obs)
            plot_duration_heatmap(stats_df, output_dir, block)
            plot_mean_durations(duration_df, output_dir, color_map, block, n_obs, global_max_mean)
            plot_syllable_duration_distribution_with_cumulative(duration_df, output_dir, block, global_xmax=global_xmax)
            plot_block_duration_distribution(duration_df, output_dir, block, global_bin_edges)
            plot_block_duration_spectrum(duration_df, output_dir, block, args.fps)
            trial_durations = duration_df.groupby(['ppid', 'session', 'trial'])['duration_seconds'].sum().reset_index()
            plot_block_duration_spaghetti(trial_durations, output_dir, block, time_unit='s')
            plot_syllable_duration_spaghetti(duration_df, output_dir, block, time_unit='s')

    print_color("\nSyllable duration analysis completed successfully!", "BLUE")
    print_color(f"All outputs are saved in the '{output_dir}' directory.", "BLUE")

if __name__ == "__main__":
    main()








