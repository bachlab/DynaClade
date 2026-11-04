"""Module for getting syllable counts (light version)."""

# ==
# author: uSerratos
# date: Nov 2024
# ==

import os
import sys
import time
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import pandas as pd

# Set matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require a display

from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from natsort import index_natsorted, order_by_index

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
    """Create a color map for labels."""
    if isinstance(colour_map_arg, str) and colour_map_arg.endswith('.pkl'):
        try:
            import pickle
            with open(colour_map_arg, 'rb') as f:
                colour_map_arg = pickle.load(f)
            # print(f"Successfully loaded color map from {colour_map_arg}")
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
            print("Falling back to pastel rainbow colormap")
            color_map = plt.cm.Pastel1
    else:
        print("Creating pastel rainbow cmap")
        color_map = plt.cm.Pastel1

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

def plot_heatmap(data: pd.DataFrame, title: str = "heatmap figure", xtick_labels: list = None, 
                 yaxis_label: str = None, xaxis_label: str = None, show: bool = False, 
                 output_path: str = None, save:bool = False, yticks_on:bool = False, ytick_obs_counts: dict = None, total_obs: int = None) -> None:
    if total_obs is None:
        total_obs = data.shape[0]
    title_with_obs = f"{title} (n={total_obs})"
    
    # Validate data and labels
    if xtick_labels is not None and len(xtick_labels) != len(data.columns):
        raise ValueError(f"Number of xtick_labels ({len(xtick_labels)}) does not match number of columns ({len(data.columns)})")
    
    # Save original column order
    original_columns = data.columns.tolist()
    
    if save and output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        # Include observation count in filename
        output_path_file = os.path.join(output_path, f"heatmap_{title.lower().replace(' ', '_')}_n_{total_obs}.png")
        csv_path = os.path.join(output_path, f"heatmap_{title.lower().replace(' ', '_')}_n_{total_obs}.csv")
        data.to_csv(csv_path)
    else:
        output_path_file = os.path.join(os.curdir, f"heatmap_{title.lower().replace(' ', '_')}_n_{total_obs}.png")
        if save and not os.path.exists(os.path.join(os.curdir)):
            os.makedirs(os.path.join(os.curdir), exist_ok=True)
            csv_path = os.path.join(os.curdir, f"heatmap_{title.lower().replace(' ', '_')}_n_{total_obs}.csv")
            data.to_csv(csv_path)

    # Use original column order instead of sorting
    data = data[original_columns]
    
    try:
        plt.figure(num=1, clear=True)
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(data, cmap='plasma', ax=ax)
        ax.set_title(title_with_obs, fontsize=20)
        
        if xtick_labels is not None:
            ax.set_xticks(np.arange(len(xtick_labels)) + 0.5)
            ax.set_xticklabels(xtick_labels, rotation='vertical', fontsize=8)
        else:
            ax.set_xticks([])
            
        if yticks_on:
            ax.tick_params(axis='y', length=0)
            ax.set_yticks(np.arange(len(data.index)) + 0.5)
            y_labels_text = []
            for label in data.index:
                if ytick_obs_counts is not None and label in ytick_obs_counts:
                    obs_count = ytick_obs_counts[label]
                else:
                    obs_count = ''
                y_labels_text.append(f"{label} (n={obs_count})" if obs_count != '' else str(label))
            ax.set_yticklabels(y_labels_text, rotation=0, fontsize=10)
            for label in ax.get_yticklabels():
                label.set_visible(True)
        else:
            ax.set_yticks([])
            
        ax.set_xlim([0, data.shape[1]])
        ax.set_ylabel(yaxis_label, fontsize=18)
        ax.set_xlabel(xaxis_label, fontsize=18)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
        
        if save:
            os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
            plt.savefig(output_path_file, bbox_inches='tight')
        if show:
            plt.show()
    finally:
        plt.close('all')
        import gc
        gc.collect()
    return None

def plot_cumulative_frequency(data, title, output_path, total_obs=None, color_map=None):
    """Create a cumulative frequency plot for syllable usage."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Sort data by mean frequency in descending order
        sorted_data = data.sort_values(by='mean', ascending=False)
        
        # Calculate cumulative sum and percentage
        sorted_data['cumulative_sum'] = sorted_data['mean'].cumsum()
        sorted_data['cumulative_percentage'] = (sorted_data['cumulative_sum'] / sorted_data['mean'].sum()) * 100
        
        # Filter to show either top 22 syllables or syllables accounting for 80% cumulative frequency
        if len(sorted_data) > 22:
            # Find the index where cumulative percentage reaches 80%
            eighty_percent_idx = sorted_data[sorted_data['cumulative_percentage'] <= 80].index
            if len(eighty_percent_idx) > 0:
                # Use the last index that's <= 80% + 1 to include the syllable that pushes us over 80%
                cutoff_idx = eighty_percent_idx[-1] if len(eighty_percent_idx) < len(sorted_data) else eighty_percent_idx[-1]
                # Take the smaller of: syllables up to 80% + 1, or top 22
                max_syllables = min(22, len(eighty_percent_idx) + 1)
            else:
                # If no syllables are <= 80%, just take top 22
                max_syllables = 22
            
            # Filter the data
            sorted_data = sorted_data.head(max_syllables)
            print_color(f"Filtered to top {len(sorted_data)} syllables (max 22 or 80% cumulative frequency)", bold_color.BLUE)
        
        # Calculate dynamic figure size based on number of syllables and label lengths
        base_width = 12
        base_height = 6
        
        # Calculate average label length
        avg_label_length = np.mean([len(str(label)) for label in sorted_data.index])
        max_label_length = max([len(str(label)) for label in sorted_data.index])
        
        # Adjust figure size based on number of syllables and label lengths
        num_syllables = len(sorted_data)
        width_multiplier = max(1.0, min(2.0, num_syllables / 15))  # Scale width based on syllable count
        height_multiplier = max(1.0, min(1.5, max_label_length / 20))  # Scale height based on max label length
        
        fig_width = base_width * width_multiplier
        fig_height = base_height * height_multiplier
        
        # Create figure with dynamic sizing
        fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        
        # Calculate dynamic font sizes based on figure dimensions and label characteristics
        # Base font sizes
        base_title_font = 14
        base_axis_font = 12
        base_tick_font = 10
        base_label_font = 8
        
        # Calculate scaling factors
        figure_area = fig_width * fig_height
        area_scale = min(1.2, max(0.7, figure_area / (base_width * base_height)))
        
        # Scale font sizes based on figure area and label complexity
        title_font = max(10, int(base_title_font * area_scale))
        axis_font = max(8, int(base_axis_font * area_scale))
        tick_font = max(6, int(base_tick_font * area_scale * (1 - avg_label_length / 100)))
        label_font = max(5, int(base_label_font * area_scale * (1 - max_label_length / 150)))
        
        # Ensure minimum readability
        tick_font = max(5, min(tick_font, 12))
        label_font = max(4, min(label_font, 10))
        
        print_color(f"Dynamic sizing: fig=({fig_width:.1f}x{fig_height:.1f}), fonts: title={title_font}, axis={axis_font}, tick={tick_font}, label={label_font}", bold_color.BLUE)
        
        # Plot bars for individual frequencies with color map if provided
        if color_map is not None:
            bar_colors = [color_map.get(str(idx), 'skyblue') for idx in sorted_data.index]
        else:
            bar_colors = 'skyblue'
            
        bars = ax1.bar(range(len(sorted_data)), sorted_data['mean'], 
                      color=bar_colors, alpha=0.6, label='Mean Syllable Proportion')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0, fontsize=label_font)
        
        # Set up second y-axis for cumulative percentage
        ax2 = ax1.twinx()
        
        # Plot cumulative percentage line in gray
        line = ax2.plot(range(len(sorted_data)), sorted_data['cumulative_percentage'],
                       color='gray', linewidth=2, label='Cumulative %')
        
        # Add percentage labels on the line
        for i, pct in enumerate(sorted_data['cumulative_percentage']):
            if i % 5 == 0:  # Label every 5th point to avoid overcrowding
                ax2.text(i, pct, f'{pct:.1f}%',
                        ha='center', va='bottom', color='gray', fontsize=label_font)
        
        # Customize the plot
        ax1.set_xlabel('Syllables (sorted by frequency)', fontsize=axis_font)
        ax1.set_ylabel('Mean Syllable Proportion', color='black', fontsize=axis_font)
        ax2.set_ylabel('Cumulative Percentage', color='gray', fontsize=axis_font)
        
        # Set title with observation count if provided
        if total_obs is not None:
            title = f"{title} (n={total_obs})"
        
        # Add filtering info to title if data was filtered
        if len(data) > len(sorted_data):
            title = f"{title} - Top {len(sorted_data)} syllables shown"
        
        ax1.set_title(title, fontsize=title_font)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=tick_font)
        
        # Set and rotate x-axis labels using set_xticklabels for proper rotation
        ax1.set_xticks(range(len(sorted_data)))
        ax1.set_xticklabels(sorted_data.index, rotation=90, ha='center', va='top', fontsize=tick_font)
        
        # Adjust layout to prevent label cutoff with dynamic space for rotated labels
        plt.tight_layout()
        # Calculate dynamic bottom margin based on label length and font size
        dynamic_bottom = max(0.15, min(0.4, 0.15 + (max_label_length * tick_font) / 2000))
        plt.subplots_adjust(bottom=dynamic_bottom)
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print_color(f"Error creating cumulative frequency plot: {str(e)}", bold_color.YELLOW)
        return False

def create_overall_syllable_plot(dataset, syllable_columns, experimental_block, output_dir, color_map, save_plots: bool = True, save_data: bool = True):
    """Create a plot of overall syllable counts for the dataset."""
    # Validate input data
    if not isinstance(syllable_columns, (list, np.ndarray)):
        raise ValueError("syllable_columns must be a list or numpy array")
    if not all(col in dataset.columns for col in syllable_columns):
        missing_cols = [col for col in syllable_columns if col not in dataset.columns]
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    # Calculate means while preserving original column order
    overall_means = dataset[syllable_columns].mean()
    
    # Create cumulative frequency plot
    try:
        # The dataset should already be normalized when passed to this function
        # Calculate mean of the normalized proportions
        overall_freq_df = pd.DataFrame({'mean': overall_means})
        plot_cumulative_frequency(
            overall_freq_df,
            f'Overall Syllable Frequency Distribution - {experimental_block}',
            os.path.join(output_dir, 'overall', f'cumulative_frequency_{experimental_block}.png'),
            len(dataset),
            color_map
        )
    except Exception as e:
        print_color(f"Could not create overall cumulative frequency plot: {str(e)}", bold_color.YELLOW)

    # Create plot with original column order
    plt.figure(figsize=(8, 5))
    
    # Plot data in original order
    plot_data = dataset[syllable_columns]
    
    sns.violinplot(data=plot_data, 
                  orient='h',
                  inner="point",
                  palette=color_map,
                  cut=0, 
                  zorder=0, 
                  scale="width", 
                  scale_hue=True,
                  inner_kws={"color": "grey", "s": 1, "edgecolor": None})
    
    mean_point = plt.scatter([], [], color='whitesmoke', marker="o", s=50, 
                           edgecolor='#555555', linewidth=1.5, label='Mean of observation means')
    obs_point = plt.scatter([], [], color='grey', marker="o", s=1, 
                          edgecolor=None, label='Observation means')
    
    # Add mean points in original order
    for j, col_name in enumerate(syllable_columns):
        if not np.isnan(overall_means[col_name]):
            plt.scatter(overall_means[col_name], j, color='whitesmoke',
                       marker="o", s=50, edgecolor='#555555', linewidth=1.5, zorder=10)
    
    plt.xlabel('Normalized Syllable Frequency per Trial')
    plt.ylabel('Syllables')
    plt.title(f'{experimental_block} | Overall | n={len(dataset)}')
    
    plt.legend(handles=[mean_point, obs_point], 
              loc='lower right',
              bbox_to_anchor=(1.0, 0.0),
              frameon=True,
              framealpha=0.8)
    
    plt.tight_layout()
    
    if save_plots:
        output_dir_overall = os.path.join(output_dir, 'overall')
        os.makedirs(output_dir_overall, exist_ok=True)
        fig_full_name = os.path.join(output_dir_overall,
                                    f'overall_syllable_usage_{experimental_block}.png')
        plt.savefig(fig_full_name, dpi=300, bbox_inches='tight')
    
    if save_data:
        # Save means in original order
        means_df = pd.DataFrame({
            'syllable': syllable_columns,
            'mean': overall_means[syllable_columns].values
        })
        means_df.to_csv(os.path.join(output_dir_overall,
                                    f'overall_syllable_usage_{experimental_block}.csv'),
                       index=False)
    
    plt.close('all')
    return overall_means

# =============================================================================
# == MODIFIED FUNCTION START ==
# =============================================================================
def perform_chi_square_tests(grouped_data, condition_name, output_dir, raw_data, syllable_cols, log_dropped_row=None):
    """
    Perform a single, overall chi-square test on the raw syllable counts 
    for all levels of a given condition.
    """
    # Create a directory for statistical tests
    stats_dir = os.path.join(output_dir, 'statistical_tests')
    os.makedirs(stats_dir, exist_ok=True)
    
    try:
        # Create the overall contingency table (MxN) where M is the number of condition levels
        # and N is the number of syllables. The values are the sum of raw counts.
        contingency_table = raw_data.groupby(condition_name)[syllable_cols].sum()

        # --- Data Cleaning and Validation ---
        
        # Remove levels (rows) with no syllable usage
        original_level_count = len(contingency_table)
        levels_with_zero_counts = contingency_table[contingency_table.sum(axis=1) == 0].index.tolist()
        contingency_table = contingency_table[contingency_table.sum(axis=1) > 0]
        if len(contingency_table) < original_level_count:
            print_color(f"Warning: Removed {original_level_count - len(contingency_table)} levels with zero syllable counts for condition '{condition_name}'.", bold_color.YELLOW)
            
            # Track the levels being removed
            if log_dropped_row:
                for level in levels_with_zero_counts:
                    # Find all trials for this level
                    level_trials = raw_data[raw_data[condition_name] == level]
                    for _, row in level_trials.iterrows():
                        log_dropped_row(
                            row['ppid'], row['session'], row['trial'],
                            f"Chi-square test: Zero syllable counts for level '{level}'",
                            f"Condition level '{level}' has zero syllable counts across all trials, removed from chi-square test for condition '{condition_name}'",
                            "Chi-square data cleaning",
                            None
                        )

        # Check if there are enough levels to perform the test
        if len(contingency_table) < 2:
            print_color(f"Error: Not enough levels with data to perform chi-square test for condition '{condition_name}'. Requires at least 2. Skipping.", "red")
            return None

        # Remove syllables (columns) with no usage across all levels
        original_syllable_count = len(contingency_table.columns)
        syllables_with_zero_counts = contingency_table.columns[contingency_table.sum(axis=0) == 0].tolist()
        contingency_table = contingency_table.loc[:, contingency_table.sum(axis=0) > 0]
        if len(contingency_table.columns) < original_syllable_count:
            print_color(f"Warning: Removed {original_syllable_count - len(contingency_table.columns)} syllables with zero counts across all levels for condition '{condition_name}'.", bold_color.YELLOW)
            
            # Track the syllables being removed
            if log_dropped_row:
                for syllable in syllables_with_zero_counts:
                    # Find all trials where this syllable has zero counts
                    zero_syllable_trials = raw_data[raw_data[syllable] == 0]
                    for _, row in zero_syllable_trials.iterrows():
                        log_dropped_row(
                            row['ppid'], row['session'], row['trial'],
                            f"Chi-square test: Zero counts for syllable '{syllable}'",
                            f"Syllable '{syllable}' has zero counts across all condition levels, removed from chi-square test for condition '{condition_name}'",
                            "Chi-square data cleaning",
                            None
                        )

        # Check if there are enough syllables to perform the test
        if len(contingency_table.columns) < 2:
            print_color(f"Error: Not enough syllables with data to perform chi-square test for condition '{condition_name}'. Requires at least 2. Skipping.", "red")
            return None

        # --- Save Artifacts and Perform Test ---

        # Save the final contingency table being used for the test
        contingency_output_path = os.path.join(stats_dir, f'overall_contingency_table_{condition_name}.csv')
        contingency_table.to_csv(contingency_output_path)
        print_color(f"Saved final contingency table to {contingency_output_path}", bold_color.GREEN)

        # Perform the single, overall Chi-Square test
        chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)

        # Create a DataFrame for the results
        results_df = pd.DataFrame([{
            'condition': condition_name,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant (p < 0.05)': p_value < 0.05,
            'levels_compared': len(contingency_table),
            'syllables_compared': len(contingency_table.columns)
        }])

        # Save results to CSV
        output_file = os.path.join(stats_dir, f'overall_chi_square_test_{condition_name}.csv')
        results_df.to_csv(output_file, index=False)
        print_color(f"Saved overall chi-square test results to {output_file}", bold_color.GREEN)

        # Save the expected frequencies table
        expected_df = pd.DataFrame(expected_freq, index=contingency_table.index, columns=contingency_table.columns)
        expected_output_path = os.path.join(stats_dir, f'overall_expected_frequencies_{condition_name}.csv')
        expected_df.to_csv(expected_output_path)
        print_color(f"Saved expected frequencies table to {expected_output_path}", bold_color.GREEN)
        
        return results_df

    except Exception as e:
        print_color(f"Error performing overall chi-square test for {condition_name}: {str(e)}", "red")
        return None

# ===========================================================================
# == MODIFIED FUNCTION END ==
# ===========================================================================

def create_combined_condition_matrix(valid_data, condition_columns, syllable_cols_list, output_dir, block_name):
    """Create a combined matrix of syllable counts for all condition levels."""
    try:
        # Initialize empty list to store all condition data
        all_condition_data = []
        
        # Process each condition
        for condition in condition_columns:
            if condition in valid_data.columns:
                # Get unique levels for this condition
                levels = valid_data[condition].dropna().unique()
                
                # Calculate mean syllable usage for each level
                for level in levels:
                    level_data = valid_data[valid_data[condition] == level]
                    if not level_data.empty:
                        mean_counts = level_data[syllable_cols_list].mean()
                        row_data = {
                            'condition': condition,
                            'level': level,
                            'n_observations': level_data.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0]
                        }
                        # Add mean counts for each syllable
                        for syllable in syllable_cols_list:
                            row_data[f'mean_{syllable}'] = mean_counts[syllable]
                        all_condition_data.append(row_data)
        
        # Convert to DataFrame
        combined_matrix = pd.DataFrame(all_condition_data)
        
        # Save the combined matrix
        output_file = os.path.join(output_dir, f'combined_condition_matrix_{block_name}.csv')
        combined_matrix.to_csv(output_file, index=False)
        
        # Create heatmap
        plt.figure(figsize=(20, len(combined_matrix) * 0.5))
        
        # Prepare data for heatmap
        heatmap_data = combined_matrix[[col for col in combined_matrix.columns if col.startswith('mean_')]]
        heatmap_data.columns = [col.replace('mean_', '') for col in heatmap_data.columns]
        
        # Create row labels with condition and level information
        row_labels = [f"{row['condition']}: {row['level']} (n={row['n_observations']})" 
                     for _, row in combined_matrix.iterrows()]
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   cmap='plasma',
                   xticklabels=True,
                   yticklabels=row_labels,
                   cbar_kws={'label': 'Mean Syllable Count'})
        
        plt.title(f'Combined Syllable Counts Across All Conditions - {block_name}')
        plt.xlabel('Syllables')
        plt.ylabel('Condition Levels')
        plt.tight_layout()
        
        # Save heatmap
        plt.savefig(os.path.join(output_dir, f'combined_condition_heatmap_{block_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return combined_matrix
        
    except Exception as e:
        print_color(f"Error creating combined condition matrix: {str(e)}", bold_color.YELLOW)
        return None

def process_combined_data(combined_data, unique_child_values, output_dir, color_map, block_name, save_plots=True, perform_stats=False, log_dropped_row=None):
    """Process combined data for all blocks or all blocks excluding instructed."""
    # Initialize list to store chi-square results for this block
    block_chi_square_results = []
    
    if not combined_data.empty and save_plots:
        syllable_cols_list = list(unique_child_values)
        
        # Track rows being dropped due to NaN values in syllable columns
        before_count = len(combined_data)
        valid_data = combined_data.dropna(subset=syllable_cols_list)
        after_count = len(valid_data)
        
        if log_dropped_row and before_count > after_count:
            # Find rows with NaN values in syllable columns
            nan_mask = combined_data[syllable_cols_list].isna().any(axis=1)
            nan_rows = combined_data[nan_mask]
            for _, row in nan_rows.iterrows():
                # Find which syllable columns have NaN values
                nan_syllables = [col for col in syllable_cols_list if pd.isna(row[col])]
                log_dropped_row(
                    row['ppid'], row['session'], row['trial'],
                    "Missing syllable data",
                    f"NaN values found in syllable columns: {nan_syllables}. This may indicate failed syllable counting or merge issues.",
                    "Syllable data filtering",
                    block_name
                )
        
        if valid_data.empty:
            print_color(f"Warning: No valid data after filtering for {block_name}", bold_color.YELLOW)
            return block_chi_square_results
            
        total_unique_obs = valid_data.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0]

        # Create overall cumulative frequency plot
        try:
            # Normalize syllable counts per trial before calculating means
            syllable_data_raw = valid_data[syllable_cols_list]
            # Calculate row sums (total syllables per trial)
            trial_sums = syllable_data_raw.sum(axis=1)
            # Perform normalization. Trials with 0 syllables will result in NaNs which are handled by fillna.
            normalized_syllable_data = syllable_data_raw.divide(trial_sums, axis=0).fillna(0)
            
            # Calculate mean of normalized proportions
            overall_freq = normalized_syllable_data.mean()
            overall_freq_df = pd.DataFrame({'mean': overall_freq})
            plot_cumulative_frequency(
                overall_freq_df,
                f'Overall Syllable Frequency Distribution - {block_name}',
                os.path.join(output_dir, f'cumulative_frequency_{block_name}.png'),
                total_unique_obs,
                color_map
            )
        except Exception as e:
            print_color(f"Could not create overall cumulative frequency plot: {str(e)}", bold_color.YELLOW)

        # Create combined condition matrix and heatmap
        condition_columns = [
            'scenario', 'scenario_id', 'threat', 'visibility', 'threat_distance',
            'threat_action', 'weapons', 'safehouse', 'day_time', 'episode_set_name',
            'threat2', 'threat_action2', 'end_state', 'reward'
        ]
        
        combined_matrix = create_combined_condition_matrix(
            valid_data, 
            condition_columns, 
            syllable_cols_list, 
            output_dir, 
            block_name
        )
        
        if combined_matrix is not None:
            print_color("Successfully created combined condition matrix and heatmap", bold_color.GREEN)

        # 1. Overall Syllable Frequencies (CSV)
        try:
            overall_stats_df = pd.DataFrame({
                'syllable': syllable_cols_list,
                'mean': valid_data[syllable_cols_list].mean().values,
                'median': valid_data[syllable_cols_list].median().values,
                'std': valid_data[syllable_cols_list].std().values,
                'q25': valid_data[syllable_cols_list].quantile(0.25).values,
                'q75': valid_data[syllable_cols_list].quantile(0.75).values
            })
            overall_stats_df.sort_values(by='mean', ascending=False, inplace=True)
            overall_stats_csv_path = os.path.join(output_dir, f'overall_syllable_frequencies_{block_name}_n_{total_unique_obs}.csv')
            overall_stats_df.to_csv(overall_stats_csv_path, index=False)
            print_color(f"Saved overall syllable frequencies to {overall_stats_csv_path}", bold_color.GREEN)
        except Exception as e_stats_csv:
            print_color(f"Could not save overall syllable frequencies CSV: {str(e_stats_csv)}", bold_color.YELLOW)

        # 2. Overall Trial-Level Heatmap
        try:
            heatmap_data_overall_trials = valid_data[syllable_cols_list]
            if not heatmap_data_overall_trials.empty:
                title_overall_trials_heatmap = f'Overall Syllable Counts - All Trials, {block_name}'
                plot_heatmap(data=heatmap_data_overall_trials, 
                           title=title_overall_trials_heatmap,
                           xtick_labels=syllable_cols_list,
                           yaxis_label=f'Trials ({block_name})',
                           xaxis_label='Syllables',
                           output_path=output_dir,
                           save=True, 
                           yticks_on=False,
                           total_obs=total_unique_obs)
        except Exception as e_overall_heatmap:
            print_color(f"Could not generate overall trial-level heatmap: {str(e_overall_heatmap)}", bold_color.YELLOW)

        # 3. Generate overall syllable usage plot
        try:
            # Normalize syllable counts per trial for violin plots
            syllable_data_raw = valid_data[syllable_cols_list]
            # Calculate row sums (total syllables per trial)
            trial_sums = syllable_data_raw.sum(axis=1)
            # Perform normalization. Trials with 0 syllables will result in NaNs which are handled by fillna.
            normalized_syllable_data = syllable_data_raw.divide(trial_sums, axis=0).fillna(0)
            
            # Create a new dataframe with normalized data for plotting
            plot_data_normalized = valid_data.copy()
            plot_data_normalized[syllable_cols_list] = normalized_syllable_data
            
            # Call the plotting function with the normalized data
            create_overall_syllable_plot(plot_data_normalized,
                                       unique_child_values,
                                       block_name,
                                       output_dir,
                                       color_map)
        except Exception as e_overall_plots:
            print_color(f"Could not generate overall syllable usage plots: {str(e_overall_plots)}", bold_color.YELLOW)

        # 4. Process each condition and level
        existing_conditions = [col for col in condition_columns if col in combined_data.columns]
        print_color(f"Processing conditions: {existing_conditions}", bold_color.BLUE)

        # Add combined threat2_threat_action2 condition
        if 'threat2' in valid_data.columns and 'threat_action2' in valid_data.columns:
            try:
                # Create combined condition
                valid_data['threat2_threat_action2'] = valid_data['threat2'] + '_' + valid_data['threat_action2']
                
                # Create directory for the combined condition
                combined_condition_dir = os.path.join(output_dir, 'threat2_threat_action2')
                os.makedirs(combined_condition_dir, exist_ok=True)
                
                # Track rows being dropped due to missing threat2 or threat_action2
                before_count = len(valid_data)
                condition_data = valid_data.dropna(subset=['threat2', 'threat_action2'])
                after_count = len(condition_data)
                
                if log_dropped_row and before_count > after_count:
                    missing_threat_rows = valid_data[valid_data[['threat2', 'threat_action2']].isna().any(axis=1)]
                    for _, row in missing_threat_rows.iterrows():
                        missing_cols = []
                        if pd.isna(row['threat2']):
                            missing_cols.append('threat2')
                        if pd.isna(row['threat_action2']):
                            missing_cols.append('threat_action2')
                        log_dropped_row(
                            row['ppid'], row['session'], row['trial'],
                            f"Missing combined condition values: {', '.join(missing_cols)}",
                            f"Required columns for threat2_threat_action2 analysis are missing: {missing_cols}",
                            "Combined condition filtering",
                            block_name
                        )
                
                if not condition_data.empty:
                    # Group data by combined condition
                    grouped_data = condition_data.groupby('threat2_threat_action2')[syllable_cols_list].mean()
                    
                    if not grouped_data.empty:
                        # Calculate observation counts for each level
                        group_sizes = (
                            condition_data
                            .groupby('threat2_threat_action2')
                            .apply(lambda df: df.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0])
                            .to_dict()
                        )
                        total_obs = condition_data.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0]

                        # Generate heatmap for the combined condition
                        plot_heatmap(data=grouped_data,
                                   title=f'Mean Syllable Counts per threat2_threat_action2 - {block_name}',
                                   xtick_labels=syllable_cols_list,
                                   yaxis_label='threat2_threat_action2',
                                   xaxis_label='Syllables',
                                   output_path=combined_condition_dir,
                                   save=True,
                                   yticks_on=True,
                                   ytick_obs_counts=group_sizes,
                                   total_obs=total_obs)

                        # Save frequency data for the combined condition
                        freq_data = pd.DataFrame({
                            'level': grouped_data.index,
                            'n_observations': [group_sizes.get(level, 0) for level in grouped_data.index]
                        })
                        
                        # Add mean frequency for each syllable
                        for syllable in syllable_cols_list:
                            freq_data[f'mean_{syllable}'] = grouped_data[syllable].values
                        
                        freq_data.to_csv(os.path.join(combined_condition_dir, f'frequency_data_threat2_threat_action2_n_{total_obs}.csv'), index=False)

                        # Create cumulative frequency plot for the combined condition
                        try:
                            # Normalize the grouped data to proportions
                            grouped_data_normalized = grouped_data.div(grouped_data.sum(axis=1), axis=0).fillna(0)
                            condition_freq = pd.DataFrame({
                                'mean': grouped_data_normalized.mean()
                            })
                            plot_cumulative_frequency(
                                condition_freq,
                                f'Syllable Frequency Distribution - threat2_threat_action2 ({block_name})',
                                os.path.join(combined_condition_dir, f'cumulative_frequency_threat2_threat_action2_n_{total_obs}.png'),
                                total_obs,
                                color_map
                            )
                        except Exception as e:
                            print_color(f"Could not create cumulative frequency plot for threat2_threat_action2: {str(e)}", bold_color.YELLOW)

                        # Perform chi-square tests if requested
                        if perform_stats:
                            results = perform_chi_square_tests(grouped_data, 'threat2_threat_action2', combined_condition_dir, condition_data, syllable_cols_list, log_dropped_row)
                            if results is not None:
                                results['block'] = block_name
                                block_chi_square_results.append(results)
                                print_color("Chi-square tests completed successfully", bold_color.GREEN)

            except Exception as e:
                print_color(f"Could not process combined threat2_threat_action2 condition: {str(e)}", bold_color.YELLOW)

        # Process other conditions
        for condition in tqdm(existing_conditions, desc=f"Processing conditions for {block_name}", leave=False):
            try:
                condition_dir = os.path.join(output_dir, condition)
                os.makedirs(condition_dir, exist_ok=True)

                # Track rows being dropped due to missing condition values
                before_count = len(valid_data)
                condition_data = valid_data.dropna(subset=[condition])
                after_count = len(condition_data)
                
                if log_dropped_row and before_count > after_count:
                    missing_condition_rows = valid_data[valid_data[condition].isna()]
                    for _, row in missing_condition_rows.iterrows():
                        log_dropped_row(
                            row['ppid'], row['session'], row['trial'],
                            f"Missing condition value: {condition}",
                            f"Condition column '{condition}' has NaN/None value. Required for condition-specific analysis.",
                            "Condition filtering",
                            block_name
                        )
                
                if condition_data.empty:
                    print_color(f"Warning: No valid data for condition {condition}", bold_color.YELLOW)
                    continue
                
                # Group data by condition
                grouped_data = condition_data.groupby(condition)[syllable_cols_list].mean()
                
                if not grouped_data.empty:
                    # Calculate observation counts for each level (only counting valid observations)
                    group_sizes = (
                        condition_data
                        .groupby(condition)
                        .apply(lambda df: df.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0])
                        .to_dict()
                    )
                    total_obs = condition_data.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0]

                    # Generate heatmap for this condition
                    plot_heatmap(data=grouped_data,
                               title=f'Mean Syllable Counts per {condition} - {block_name}',
                               xtick_labels=syllable_cols_list,
                               yaxis_label=str(condition),
                               xaxis_label='Syllables',
                               output_path=condition_dir,
                               save=True,
                               yticks_on=True,
                               ytick_obs_counts=group_sizes,
                               total_obs=total_obs)

                    # Save frequency data for this condition
                    freq_data = pd.DataFrame({
                        'level': grouped_data.index,
                        'n_observations': [group_sizes.get(level, 0) for level in grouped_data.index]
                    })
                    
                    # Add mean frequency for each syllable
                    for syllable in syllable_cols_list:
                        freq_data[f'mean_{syllable}'] = grouped_data[syllable].values
                    
                    freq_data.to_csv(os.path.join(condition_dir, f'frequency_data_{condition}_n_{total_obs}.csv'), index=False)

                    # Create cumulative frequency plot for this condition
                    try:
                        # Normalize the grouped data to proportions
                        grouped_data_normalized = grouped_data.div(grouped_data.sum(axis=1), axis=0).fillna(0)
                        condition_freq = pd.DataFrame({
                            'mean': grouped_data_normalized.mean()
                        })
                        plot_cumulative_frequency(
                            condition_freq,
                            f'Syllable Frequency Distribution - {condition} ({block_name})',
                            os.path.join(condition_dir, f'cumulative_frequency_{condition}_n_{total_obs}.png'),
                            total_obs,
                            color_map
                        )
                    except Exception as e:
                        print_color(f"Could not create cumulative frequency plot for {condition}: {str(e)}", bold_color.YELLOW)

                    # Perform chi-square tests if requested
                    if perform_stats:
                        print_color(f"\nPerforming chi-square tests for {condition}...", bold_color.BLUE)
                        results = perform_chi_square_tests(grouped_data, condition, condition_dir, condition_data, syllable_cols_list, log_dropped_row)
                        if results is not None:
                            results['block'] = block_name
                            block_chi_square_results.append(results)
                            print_color("Chi-square tests completed successfully", bold_color.GREEN)

            except Exception as e:
                print_color(f"Could not process condition {condition}: {str(e)}", bold_color.YELLOW)

        # 5. Generate pairwise difference heatmaps for each condition
        for condition in tqdm(existing_conditions, desc=f"Generating pairwise differences for {block_name}", leave=False):
            try:
                condition_dir = os.path.join(output_dir, condition)
                os.makedirs(condition_dir, exist_ok=True)

                # Drop NaN values for this specific condition
                condition_data = valid_data.dropna(subset=[condition])
                
                if condition_data.empty:
                    print_color(f"Warning: No valid data for pairwise differences in condition {condition}", bold_color.YELLOW)
                    continue

                # Get unique levels for this condition
                levels = condition_data[condition].unique()
                
                if len(levels) > 1:
                    # Create pairwise difference matrix
                    diff_matrix = pd.DataFrame(index=levels, columns=levels)
                    
                    # Calculate mean syllable usage for each level
                    level_means = {}
                    for level in levels:
                        level_data = condition_data[condition_data[condition] == level]
                        level_means[level] = level_data[syllable_cols_list].mean()
                    
                    # Calculate pairwise differences
                    for level1 in levels:
                        for level2 in levels:
                            if level1 != level2:
                                # Calculate mean difference across all syllables
                                diff = level_means[level1] - level_means[level2]
                                diff_matrix.loc[level1, level2] = diff.mean()

                    # Calculate total observations for this condition
                    total_obs = condition_data.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0]

                    # Save difference matrix
                    diff_matrix.to_csv(os.path.join(condition_dir, f'pairwise_differences_{condition}_n_{total_obs}.csv'))

                    # Plot heatmap of differences
                    plt.figure(figsize=(10, 8))
                    # Convert to numeric values for plotting
                    numeric_matrix = diff_matrix.astype(float)
                    sns.heatmap(numeric_matrix, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
                    plt.title(f'Pairwise Differences in Syllable Usage - {condition} (n={total_obs})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(condition_dir, f'pairwise_differences_{condition}_n_{total_obs}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    print_color(f"Warning: Not enough levels for pairwise differences in {condition}. Found {len(levels)} levels.", bold_color.YELLOW)

            except Exception as e:
                print_color(f"Could not generate pairwise differences for {condition}: {str(e)}", bold_color.YELLOW)

        print_color(f"\nCompleted processing {block_name} data", bold_color.GREEN)
        
    return block_chi_square_results

@tictoc
def main():
    parser = argparse.ArgumentParser(description='Process syllable counts from datasets (light version).')
    parser.add_argument('--labels-path', type=str, required=True,
                      help='Path to the labels dataset CSV file')
    parser.add_argument('--label-column', type=str, required=True,
                      help='Name of the column containing the labels')
    parser.add_argument('--vr-metadata-path', type=str, required=True,
                      help='Path to the VR metadata dataset CSV file')
    parser.add_argument('--failed-trials-path', type=str, required=False, default=None,
                      help='Path to CSV file containing metadata of failed trials (must have ppid, session, trial columns)')
    parser.add_argument('--no-plots', action='store_true',
                      help='Disable plot generation')
    parser.add_argument('--output-dir', type=str, required=False, default=None,
                      help='Path to the output directory')
    parser.add_argument('--colour-map-object', type=str, required=False, default=None,
                      help='Path to the colour map object')
    parser.add_argument('--perform-stats', action='store_true',
                      help='Perform statistical tests (chi-square) on the data')
    
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

    MERGE_COLUMNS_RIGHT = ['ppid_vr', 'session_vr', 'trial_vr']
    TARGET_COLS_DICT = [
        'end_state','scenario', 'scenario_id', 'threat', 
        'visibility', 'threat_distance', 'threat_action', 
        'weapons', 'safehouse', 'ppid_vr', 'trial_vr', 'reward'
    ]
    TARGET_COLS_INST = ['instructed_action']
    MAPPING = {
        'dictionary': {
            'scenario': 0,
            'scenario_id': 1,
            'threat': 2,
            'visibility': 3,
            'threat_distance': 4,
            'threat_action': 5,
            'weapons': 6,
            'safehouse': 7,
        },
        'fixed': {
            'scenario': 0,
            'scenario_id': 3,
            'threat': 4,
            'fixed_order': 2,
        },
        'stump': {
            'scenario': 0,
            'threat': 1,
            'threat_action': 4,
        },
        'instructed': {
            'scenario': 0,
            'scenario_id': 1,
            'instructed_action': 2,
        }
    }

    if args.output_dir is None:
        output_dir = os.path.dirname(args.vr_metadata_path)
    else:
        output_dir = args.output_dir

    output_dir = os.path.join(output_dir, 'syllable_counts_output')
    os.makedirs(output_dir, exist_ok=True)

    # Load the datasets
    dataset1 = pd.read_csv(args.labels_path)
    dataset2 = pd.read_csv(args.vr_metadata_path)
    coda_columns = ['ppid_coda', 'session_coda', 'trial_coda']
    
    if set(coda_columns).issubset(dataset2.columns):
        print_color(f"\nCoda columns found in dataset2. Merging on {coda_columns}...\n", bold_color.GREEN)
        MERGE_COLUMNS_RIGHT = coda_columns
    
    if not set(MERGE_COLUMNS_RIGHT).issubset(dataset2.columns):
        MERGE_COLUMNS_RIGHT = ['ppid', 'session', 'trial']
        TARGET_COLS_DICT = [
            'end_state','scenario', 'scenario_id', 'threat', 
            'visibility', 'threat_distance', 'threat_action', 
            'weapons', 'safehouse', 'ppid', 'trial', 'reward', 'fixed_order',
        ]

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

    # Get unique values and create color map
    unique_child_values = dataset1[args.label_column].unique()
    le = LabelEncoder()
    le.fit(list(unique_child_values))
    color_map = create_color_map(le, colour_map_arg=args.colour_map_object)

    # Initialize columns for syllable counts
    for value in unique_child_values:
        dataset2[value] = 0

    # Get raw integer counts instead of normalized frequencies.
    child_freq = dataset1.groupby(['ppid', 'session', 'trial'])[args.label_column].value_counts().unstack(fill_value=0)
    child_freq = child_freq.reset_index()

    # Merge datasets
    dataset3 = pd.merge(child_freq, dataset2, left_on=['ppid', 'session', 'trial'], right_on=MERGE_COLUMNS_RIGHT, suffixes=('', '_right'))
    
    # Track rows that didn't merge (missing from metadata)
    merged_keys = set(zip(dataset3['ppid'], dataset3['session'], dataset3['trial']))
    original_keys = set(zip(child_freq['ppid'], child_freq['session'], child_freq['trial']))
    missing_from_metadata = original_keys - merged_keys
    
    for ppid, session, trial in missing_from_metadata:
        log_dropped_row(
            ppid, session, trial,
            "Missing from metadata dataset",
            f"Row exists in labels dataset but not found in VR metadata dataset. Merge keys: {MERGE_COLUMNS_RIGHT}",
            "Initial merge",
            None
        )

    # Map experiment names
    experiment_names_mapping = {
        'Dictionary of Actions Replication - Main': 'dictionary',
        'Dictionary of Actions Replication - Instructed Actions': 'instructed',
        'Dictionary of Actions Replication - Fixed Trials': 'fixed',
        'Dictionary of Actions Replication - Stump Trials': 'stump'
    }

    # Store original episode_set_name before mapping
    dataset3['episode_set_name_original'] = dataset3['episode_set_name']
    dataset3['episode_set_name'] = dataset3['episode_set_name'].map(experiment_names_mapping)
    
    # Track rows with unmapped episode_set_name
    unmapped_episodes = dataset3[dataset3['episode_set_name'].isna()]
    for _, row in unmapped_episodes.iterrows():
        original_episode = row['episode_set_name_original']
        log_dropped_row(
            row['ppid'], row['session'], row['trial'],
            "Unmapped episode_set_name",
            f"Episode set name '{original_episode}' not found in mapping: {list(experiment_names_mapping.keys())}",
            "Episode mapping",
            None
        )

    # Process each experimental block
    data_to_process = {
        "instructed": {
            "data": dataset3[dataset3['episode_set_name'] == "instructed"],
            "target_cols": TARGET_COLS_INST,
            "output_dir": os.path.join(output_dir, 'instructed')
        },
        "dictionary": {
            "data": dataset3[dataset3['episode_set_name'] == "dictionary"],
            "target_cols": TARGET_COLS_DICT + ["day_time"],
            "output_dir": os.path.join(output_dir, 'dictionary')
        },
        "fixed": {
            "data": dataset3[dataset3['episode_set_name'] == "fixed"],
            "target_cols": TARGET_COLS_DICT + ["day_time"],
            "output_dir": os.path.join(output_dir, 'fixed')
        },
        "stump": {
            "data": dataset3[dataset3['episode_set_name'] == "stump"],
            "target_cols": TARGET_COLS_DICT,
            "output_dir": os.path.join(output_dir, 'stump')
        }
    }

    # Initialize list to store all chi-square results
    all_chi_square_results = []

    # Process each block
    for key, value in tqdm(data_to_process.items(), desc="Processing experimental blocks", leave=True):
        if key == "fixed":
            # Track rows being filtered out for fixed block
            before_count = len(value["data"])
            filtered_mask = value["data"]["episode"].str.contains("Forest_Habituation|NOTUSED", case=False, regex=True)
            filtered_rows = value["data"][filtered_mask]
            value["data"] = value["data"][~filtered_mask]
            after_count = len(value["data"])
            
            if before_count > after_count:
                for _, row in filtered_rows.iterrows():
                    log_dropped_row(
                        row['ppid'], row['session'], row['trial'],
                        "Fixed block episode filtering",
                        f"Episode '{row['episode']}' contains 'Forest_Habituation' or 'NOTUSED' which are excluded from fixed block analysis",
                        "Fixed block filtering",
                        key
                    )
        
        if not value["data"].empty:
            # Process columns according to mapping
            for col in value["target_cols"]:
                if col not in value["data"].columns and col in MAPPING[key]:
                    def process_episode(x, key, col):
                        try:
                            if key == "stump":
                                parts = x.strip().split('-')  
                                value["data"]["scenario_id"] = parts[0].split('_')[1][:3] + "s"
                            else:
                                parts = x.strip().split('_')
                            
                            if key == "instructed":
                                if MAPPING[key][col] == 0:
                                    return '_'.join(parts[:2])
                                elif MAPPING[key][col] == 1:
                                    return '_'.join(parts[1:2])
                                elif MAPPING[key][col] >= 2:
                                    return '_'.join(parts[2:]).replace("instructed_", "")
                                return None
                            else:
                                if len(parts) > MAPPING[key][col]:
                                    return parts[MAPPING[key][col]]
                                return None
                        except Exception as e:
                            print(f"Error processing episode '{x}' for key '{key}', column '{col}': {str(e)}")
                            return None
                    
                    value["data"][col] = value["data"]["episode"].apply(
                        lambda x: process_episode(x, key, col)
                    )
                    
                    # Track rows where episode parsing failed
                    failed_parsing = value["data"][value["data"][col].isna()]
                    for _, row in failed_parsing.iterrows():
                        log_dropped_row(
                            row['ppid'], row['session'], row['trial'],
                            "Episode parsing failed",
                            f"Failed to parse episode '{row['episode']}' for column '{col}' in block '{key}'. Expected format not matched.",
                            "Episode parsing",
                            key
                        )
            
            # Add day_time column if needed
            if 'day_time' not in value["data"].columns:
                value["data"]["day_time"] = value["data"]["episode"].str.lower().apply(lambda x: "night" if "night" in x else "day")
            
            # Special processing for stump and fixed blocks
            if key == "stump":
                value["data"]["scenario"] = "Forest 2"
                value["data"]["threat_action"] = "threaten"
                value["data"]["weapons"] = "noweapons"
                value["data"]["safehouse"] = "safehouse"
                value["data"]["visibility"] = "visible"
                value["data"]["threat_distance"] = "close"
                value["data"]["threat2"] = value["data"]["threat"].str.lower()
                value["data"]["threat"] = value["data"]["threat"].str.lower()
                value["data"]["episode_set_name2"] = "stump"
            
            if key == "fixed":
                value["data"]["threat_action"] = "attack"
                value["data"]["weapons"] = "noweapons"
                value["data"]["safehouse"] = "safehouse"
                value["data"]["visibility"] = "novisible"
                value["data"]["threat_distance"] = "medium"
                value["data"]["threat"] = value["data"]["threat"].str.lower()
                value["data"]["threat"] = value["data"]["threat"].apply(lambda x: "human" if "human a" in x else x)
                value["data"]["threat2"] = value["data"]["threat"].str.lower()
                value["data"]["scenario_id"] = value["data"]["scenario_id"].apply(lambda x: str(x).lower() + "f")
                value["data"]["episode_set_name2"] = "fixed"
    
            if key == "dictionary":
                value["data"]["threat2"] = value["data"].apply(
                    lambda row: row["threat"] if not (row["threat_action"].lower() == "null" and row["scenario_id"] == "026") else "null",
                    axis=1)
                value["data"]["threat_action2"] = value["data"].apply(
                    lambda row: "loiter" if row["threat_action"].lower() == "null" and row["scenario_id"] != "026" else 
                                ("null" if row["threat_action"].lower() == "null" and row["scenario_id"] == "026" else row["threat_action"]),
                    axis=1)
                value["target_cols"].append("threat2")
                value["target_cols"].append("threat_action2")
                value["data"]["episode_set_name2"] = "dictionary"
            
            if key == "instructed":
                value["data"]["episode_set_name2"] = "instructed"
                
        data_to_process[key]["data"] = value["data"]

    # NOW identify failed trials after all blocks have been parsed
    if failed_trials_data is not None:
        print_color("\nIdentifying failed trials after parsing all experimental blocks...", bold_color.BLUE)
        
        # Collect all processed data to identify failed trials
        all_processed_data = pd.concat([value["data"] for value in data_to_process.values() if not value["data"].empty], axis=0, ignore_index=True)
        
        # Create a mask for failed trials
        failed_mask = all_processed_data.set_index(['ppid', 'session', 'trial']).index.isin(
            failed_trials_data.set_index(['ppid', 'session', 'trial']).index
        )
        
        if failed_mask.sum() > 0:
            # Extract failed trials with their parsed metadata
            failed_trials = all_processed_data[failed_mask].copy()
            failed_trials['episode_set_name2'] = 'failed'
            
            # Remove failed trials from their original blocks
            for key, value in data_to_process.items():
                if not value["data"].empty:
                    # Create mask to remove failed trials from this block
                    block_failed_mask = value["data"].set_index(['ppid', 'session', 'trial']).index.isin(
                        failed_trials_data.set_index(['ppid', 'session', 'trial']).index
                    )
                    if block_failed_mask.sum() > 0:
                        # Track the failed trials being removed
                        failed_trials_in_block = value["data"][block_failed_mask]
                        for _, row in failed_trials_in_block.iterrows():
                            log_dropped_row(
                                row['ppid'], row['session'], row['trial'],
                                "Failed trial removal",
                                f"Trial marked as failed in failed trials file. Moved to separate 'failed' block for analysis.",
                                "Failed trial processing",
                                key
                            )
                        
                        value["data"] = value["data"][~block_failed_mask]
                        print_color(f"Removed {block_failed_mask.sum()} failed trials from {key} block", bold_color.YELLOW)
            
            # Add failed trials as a new block
            data_to_process["failed"] = {
                "data": failed_trials,
                "target_cols": TARGET_COLS_DICT + ["day_time"],
                "output_dir": os.path.join(output_dir, 'failed')
            }
            print_color(f"Created failed trials block with {len(failed_trials)} trials (with parsed metadata)", bold_color.GREEN)
        else:
            print_color("No failed trials found in the processed dataset", bold_color.YELLOW)

    # Generate plots for each block
    for key, value in tqdm(data_to_process.items(), desc="Generating plots", leave=True):
        if not value["data"].empty and not args.no_plots:
            existing_cols = [col for col in value["target_cols"] if col in value["data"].columns]
            
            # Generate mean heatmaps for each condition
            for condition in tqdm(existing_cols, desc=f"Generating heatmaps for {key}", leave=False):
                try:
                    current_block_data = value["data"].copy()
                    before_count = len(current_block_data)
                    current_block_data = current_block_data.dropna(subset=[condition])
                    after_count = len(current_block_data)
                    
                    # Track rows dropped due to missing condition values
                    if before_count > after_count:
                        missing_condition_rows = value["data"][value["data"][condition].isna()]
                        for _, row in missing_condition_rows.iterrows():
                            log_dropped_row(
                                row['ppid'], row['session'], row['trial'],
                                f"Missing condition value: {condition}",
                                f"Condition column '{condition}' has NaN/None value. Required for condition-specific analysis.",
                                "Condition filtering",
                                key
                            )
                    
                    if not current_block_data.empty:
                        mean_data = current_block_data.groupby(condition)[list(unique_child_values)].mean()
                        
                        if not mean_data.empty:
                            mean_heatmap_dir = os.path.join(value["output_dir"], condition)
                            os.makedirs(mean_heatmap_dir, exist_ok=True)
                            
                            group_sizes = (
                                current_block_data
                                .groupby(condition)
                                .apply(lambda df: df.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0])
                                .to_dict()
                            )
                            total_obs = current_block_data.drop_duplicates(subset=['ppid', 'session', 'trial']).shape[0]
                            
                            plot_heatmap(data=mean_data, 
                                       title=f'Mean Syllable Counts per {condition} - {key} block',
                                       xtick_labels=list(mean_data.columns),
                                       yaxis_label=str(condition),
                                       xaxis_label='Syllables',
                                       output_path=mean_heatmap_dir,
                                       save=True, 
                                       yticks_on=True, 
                                       ytick_obs_counts=group_sizes, 
                                       total_obs=total_obs)

                            # Perform chi-square tests if requested
                            if args.perform_stats:
                                results = perform_chi_square_tests(mean_data, condition, value["output_dir"], value["data"], list(unique_child_values), log_dropped_row)
                                if results is not None:
                                    # Add block name to results
                                    results['block'] = key
                                    all_chi_square_results.append(results)
                                    print_color("Chi-square tests completed successfully", bold_color.GREEN)
                except Exception as e:
                    print_color(f"Could not generate heatmap for {condition} in {key}: {str(e)}", bold_color.YELLOW)
            
            # Generate overall syllable usage plot
            syllable_columns = list(unique_child_values)
            syllable_data_raw = value["data"][syllable_columns]
            
            # Calculate row sums (total syllables per trial)
            trial_sums = syllable_data_raw.sum(axis=1)
            
            # Perform normalization. Trials with 0 syllables will result in NaNs which are handled by fillna.
            normalized_syllable_data = syllable_data_raw.divide(trial_sums, axis=0).fillna(0)
            
            # Create a new dataframe with normalized data for plotting
            plot_data_normalized = value["data"].copy()
            plot_data_normalized[syllable_columns] = normalized_syllable_data
            
            # Call the plotting function with the corrected, normalized data
            create_overall_syllable_plot(plot_data_normalized,
                                       syllable_columns,
                                       key,
                                       value["output_dir"],
                                       color_map)

    # Save the combined mapping with only specified columns (excluding failed trials)
    vr_to_mocap_mapping = pd.concat([value["data"] for key, value in data_to_process.items() 
                                    if not value["data"].empty and key != "failed"], axis=0, ignore_index=True)
    
    # Define the columns to keep in the main output file
    main_output_columns = [
        "ppid", "session", "trial", "start_time", "end_time", "episode", 
        "end_state", "reward", "instructed_action", "day_time", 
        "scenario", "scenario_id", "threat", "visibility", "threat_distance", 
        "threat_action", "weapons", "safehouse", "threat2", "threat_action2", 
        "fixed_order", "episode_set_name2"
    ]
    
    # Filter columns that exist in the dataset
    existing_columns = [col for col in main_output_columns if col in vr_to_mocap_mapping.columns]
    missing_columns = [col for col in main_output_columns if col not in vr_to_mocap_mapping.columns]
    
    if missing_columns:
        print_color(f"Warning: Some requested columns are missing from the dataset: {missing_columns}", bold_color.YELLOW)
    
    # Save the filtered dataset
    filtered_mapping = vr_to_mocap_mapping[existing_columns]
    filtered_mapping.to_csv(os.path.join(output_dir, 'vr_to_mocap_mapping.csv'), index=False)
    print_color(f"Saved filtered dataset with {len(existing_columns)} columns: {existing_columns}", bold_color.GREEN)
    
    # Save dataset with metadata and syllable counts only
    metadata_columns = ["ppid", "session", "trial"]
    syllable_columns = [col for col in vr_to_mocap_mapping.columns if col not in main_output_columns and col not in metadata_columns]
    
    # Create metadata + syllable counts dataset
    metadata_syllable_columns = metadata_columns + syllable_columns
    metadata_syllable_mapping = vr_to_mocap_mapping[metadata_syllable_columns]
    metadata_syllable_mapping.to_csv(os.path.join(output_dir, 'syllable_counts_only.csv'), index=False)
    print_color(f"Saved syllable counts dataset with {len(metadata_syllable_columns)} columns ({len(metadata_columns)} metadata + {len(syllable_columns)} syllables)", bold_color.GREEN)
    
    # Also save the full dataset with all columns for reference (excluding failed trials)
    vr_to_mocap_mapping.to_csv(os.path.join(output_dir, 'vr_to_mocap_mapping_full.csv'), index=False)
    print_color(f"Saved full dataset with all {len(vr_to_mocap_mapping.columns)} columns (failed trials excluded)", bold_color.BLUE)

    # Process combined data for all blocks (excluding failed trials)
    if not vr_to_mocap_mapping.empty and not args.no_plots:
        overall_summary_output_dir = os.path.join(output_dir, 'overall_summary_all_blocks')
        os.makedirs(overall_summary_output_dir, exist_ok=True)
        all_blocks_results = process_combined_data(vr_to_mocap_mapping, list(unique_child_values), overall_summary_output_dir, color_map, 'all_blocks', perform_stats=args.perform_stats, log_dropped_row=log_dropped_row)
        if all_blocks_results:
            all_chi_square_results.extend(all_blocks_results)

        # Process combined data excluding instructed blocks and failed trials
        overall_combined_data_excl_instructed = vr_to_mocap_mapping[
            (vr_to_mocap_mapping['episode_set_name'] != 'instructed') & 
            (vr_to_mocap_mapping['episode_set_name2'] != 'failed')
        ].copy()
        if not overall_combined_data_excl_instructed.empty:
            overall_summary_excl_instr_output_dir = os.path.join(output_dir, 'overall_summary_no_instructed')
            os.makedirs(overall_summary_excl_instr_output_dir, exist_ok=True)
            no_instructed_results = process_combined_data(overall_combined_data_excl_instructed, list(unique_child_values), overall_summary_excl_instr_output_dir, color_map, 'no_instructed', perform_stats=args.perform_stats, log_dropped_row=log_dropped_row)
            if no_instructed_results:
                all_chi_square_results.extend(no_instructed_results)
            print_color(f"Processed combined data excluding instructed blocks and failed trials: {len(overall_combined_data_excl_instructed)} trials", bold_color.GREEN)
        else:
            print_color("Warning: No data remaining after excluding instructed blocks and failed trials", bold_color.YELLOW)

    # Save combined chi-square results if any were performed
    if args.perform_stats and all_chi_square_results:
        try:
            combined_results = pd.concat(all_chi_square_results, ignore_index=True)
            combined_results_path = os.path.join(output_dir, 'combined_chi_square_results.csv')
            combined_results.to_csv(combined_results_path, index=False)
            print_color(f"\nSaved combined chi-square results to: {combined_results_path}", bold_color.GREEN)
        except Exception as e:
            print_color(f"Error saving combined chi-square results: {str(e)}", bold_color.YELLOW)

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

    print_color("\nSyllable counts completed successfully!".capitalize(), bold_color.BLUE)
    print_color(f"\nThe data can be found in: ".capitalize() + f"{output_dir}", bold_color.BLUE)

if __name__ == "__main__":
    print_welcome_message()
    main()