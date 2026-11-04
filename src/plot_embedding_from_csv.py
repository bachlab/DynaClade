#!/usr/bin/env python3
"""
Embedding Visualization Tool

This script loads embeddings from a CSV file and creates 2D and 3D visualizations.
The CSV should contain:
- Embedding components named as 0, 1, 2, ..., n-1
- A 'label' column for clustering/coloring

Usage:
    python plot_embedding_from_csv.py <csv_file_path>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import argparse
import os
import sys
import pickle
from datetime import datetime


def load_embedding_data(csv_path, label_mapping_csv=None, new_label_column=None):
    """
    Load embedding data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        label_mapping_csv (str, optional): Path to CSV file for label mapping
        new_label_column (str, optional): Column name in mapping CSV containing the new labels
        
    Returns:
        tuple: (embedding_matrix, labels, component_names)
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if 'label' column exists
        if 'label' not in df.columns:
            raise ValueError("CSV file must contain a 'label' column")
        
        # Extract embedding components (numeric columns excluding 'label')
        embedding_cols = []
        for col in df.columns:
            if col != 'label' and col.isdigit():
                embedding_cols.append(col)
        
        if not embedding_cols:
            raise ValueError("No embedding component columns found (expected numeric column names: 0, 1, 2, ...)")
        
        # Sort embedding columns numerically
        embedding_cols = sorted(embedding_cols, key=int)
        print(f"Found embedding components: {embedding_cols}")
        
        # Extract embedding matrix and labels
        embedding_matrix = df[embedding_cols].values
        labels = df['label'].values
        
        # Apply label mapping if provided
        if label_mapping_csv is not None and new_label_column is not None:
            labels = apply_label_mapping(labels, label_mapping_csv, new_label_column)
        
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        print(f"Number of unique labels: {len(np.unique(labels))}")
        print(f"Labels: {np.unique(labels)}")
        
        return embedding_matrix, labels, embedding_cols
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def apply_label_mapping(labels, mapping_csv_path, new_label_column):
    """
    Apply label mapping using a CSV file.
    
    Args:
        labels (np.array): Original labels
        mapping_csv_path (str): Path to CSV file containing label mappings
        new_label_column (str): Column name in mapping CSV containing the new labels
        
    Returns:
        np.array: Mapped labels
    """
    try:
        # Load mapping CSV
        mapping_df = pd.read_csv(mapping_csv_path)
        print(f"Loaded mapping CSV with shape: {mapping_df.shape}")
        print(f"Mapping CSV columns: {list(mapping_df.columns)}")
        
        # Check if required columns exist
        if 'label' not in mapping_df.columns:
            raise ValueError("Mapping CSV must contain a 'label' column")
        if new_label_column not in mapping_df.columns:
            raise ValueError(f"Mapping CSV must contain column '{new_label_column}'")
        
        # Create mapping dictionary
        label_mapping = dict(zip(mapping_df['label'], mapping_df[new_label_column]))
        print(f"Created label mapping: {label_mapping}")
        
        # Apply mapping
        mapped_labels = np.array([label_mapping.get(label, label) for label in labels])
        
        # Rename squat_reach to squat_manipulate if it exists
        mapped_labels = np.where(mapped_labels == 'squat_reach', 'squat_manipulate', mapped_labels)
        
        print(f"Applied label mapping. Original unique labels: {len(np.unique(labels))}, "
              f"Mapped unique labels: {len(np.unique(mapped_labels))}")
        
        return mapped_labels
        
    except Exception as e:
        print(f"Error applying label mapping: {e}")
        sys.exit(1)


def load_colormap_from_pickle(colormap_path):
    """
    Load colormap from pickle file.
    
    Args:
        colormap_path (str): Path to pickle file containing colormap dictionary
        
    Returns:
        dict or None: Dictionary mapping labels to colors
    """
    try:
        with open(colormap_path, 'rb') as f:
            colormap_dict = pickle.load(f)
        print(f"Loaded colormap dictionary from: {colormap_path}")
        print(f"Colormap contains {len(colormap_dict)} label-color mappings")
        return colormap_dict
    except Exception as e:
        print(f"Error loading colormap from pickle file: {e}")
        return None


def create_2d_plot(embedding_matrix, labels, component_names, output_dir, colormap=None, show_plots=False):
    """
    Create a 2D scatter plot of the embedding.
    
    Args:
        embedding_matrix (np.array): The embedding data
        labels (np.array): The label data for coloring
        component_names (list): Names of the embedding components
        output_dir (str): Directory to save the plot
        colormap (dict, optional): Dictionary mapping labels to colors
        show_plots (bool): Whether to display the plot
    """
    # Use the first two components for 2D plot
    if len(component_names) < 2:
        print("Warning: Need at least 2 components for 2D plot")
        return
    
    x = embedding_matrix[:, 0]
    y = embedding_matrix[:, 1]
    
    # Create the plot with fixed size to prevent legend from stretching it
    plt.figure(figsize=(12, 8))
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    
    # Create color map using provided dictionary or default colors
    colors = []
    if colormap is not None:
        # Use provided colormap dictionary
        for label in unique_labels:
            if label in colormap:
                colors.append(colormap[label])
            else:
                # Special handling for "other" label - use grey
                if label == "other":
                    colors.append('grey')
                    print(f"Using grey color for 'other' label")
                else:
                    # Fallback color for labels not in the dictionary
                    colors.append('lightblue')
                    print(f"Warning: Label '{label}' not found in colormap, using fallback color")
    else:
        # Use default color scheme
        default_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            if label == -1 or label == "-1" or label == "other":
                colors.append('grey')
            else:
                colors.append(default_colors[i])
    
    # Plot each label with different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(x[mask], y[mask], 
                   c=[colors[i]], 
                   label=f'{label}', 
                   alpha=1.0, 
                   s=20,
                   marker='.')
    
    plt.xlabel(f'Component {component_names[0]}')
    plt.ylabel(f'Component {component_names[1]}')
    plt.title('2D Embedding Visualization')
    
    # Show legend with as many entries as possible without elongating the figure
    if len(unique_labels) <= 50:  # Reasonable limit for legend entries
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    else:
        # For many labels, show only the most frequent ones
        label_counts = [(label, np.sum(labels == label)) for label in unique_labels]
        label_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count
        
        # Show top 30 most frequent labels
        top_labels = [label for label, count in label_counts[:30]]
        top_colors = [colors[list(unique_labels).index(label)] for label in top_labels]
        
        # Create custom legend with top labels only
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     markersize=8, label=f'{label} (n={np.sum(labels == label)})') 
                          for label, color in zip(top_labels, top_colors)]
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=7, ncol=1)
        print(f"Showing top 30 most frequent labels out of {len(unique_labels)} total labels")
    
    plt.grid(True, alpha=0.3)
      # Ensure equal aspect ratio
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'embedding_2d_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D plot saved to: {output_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def create_3d_plot(embedding_matrix, labels, component_names, output_dir, colormap=None, show_plots=False):
    """
    Create a 3D scatter plot of the embedding.
    
    Args:
        embedding_matrix (np.array): The embedding data
        labels (np.array): The label data for coloring
        component_names (list): Names of the embedding components
        output_dir (str): Directory to save the plot
        colormap (dict, optional): Dictionary mapping labels to colors
        show_plots (bool): Whether to display the plot
    """
    # Use the first three components for 3D plot
    if len(component_names) < 3:
        print("Warning: Need at least 3 components for 3D plot")
        return
    
    x = embedding_matrix[:, 0]
    y = embedding_matrix[:, 1]
    z = embedding_matrix[:, 2]
    
    # Create the 3D plot with fixed size
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    
    # Create color map using provided dictionary or default colors
    colors = []
    if colormap is not None:
        # Use provided colormap dictionary
        for label in unique_labels:
            if label in colormap:
                colors.append(colormap[label])
            else:
                # Special handling for "other" label - use grey
                if label == "other":
                    colors.append('grey')
                    print(f"Using grey color for 'other' label")
                else:
                    # Fallback color for labels not in the dictionary
                    colors.append('lightblue')
                    print(f"Warning: Label '{label}' not found in colormap, using fallback color")
    else:
        # Use default color scheme
        default_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            if label == -1 or label == "-1" or label == "other":
                colors.append('grey')
            else:
                colors.append(default_colors[i])
    
    # Plot each label with different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(x[mask], y[mask], z[mask], 
                  c=[colors[i]], 
                  label=f'{label}', 
                  alpha=1.0, 
                  s=20,
                  marker='.')
    
    ax.set_xlabel(f'Component {component_names[0]}')
    ax.set_ylabel(f'Component {component_names[1]}')
    ax.set_zlabel(f'Component {component_names[2]}')
    ax.set_title('3D Embedding Visualization')
    
    # Show legend with as many entries as possible without elongating the figure
    if len(unique_labels) <= 50:  # Reasonable limit for legend entries
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    else:
        # For many labels, show only the most frequent ones
        label_counts = [(label, np.sum(labels == label)) for label in unique_labels]
        label_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count
        
        # Show top 30 most frequent labels
        top_labels = [label for label, count in label_counts[:30]]
        top_colors = [colors[list(unique_labels).index(label)] for label in top_labels]
        
        # Create custom legend with top labels only
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     markersize=8, label=f'{label} (n={np.sum(labels == label)})') 
                          for label, color in zip(top_labels, top_colors)]
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=7, ncol=1)
        print(f"Showing top 30 most frequent labels out of {len(unique_labels)} total labels")
    
    # Set equal aspect ratio for 3D plot
    ax.set_box_aspect([1,1,1])
    
    # Save the plot
    output_path = os.path.join(output_dir, 'embedding_3d_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D plot saved to: {output_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def create_combined_plot(embedding_matrix, labels, component_names, output_dir, colormap=None, show_plots=False):
    """
    Create a combined plot showing both 2D and 3D views side by side.
    
    Args:
        embedding_matrix (np.array): The embedding data
        labels (np.array): The label data for coloring
        component_names (list): Names of the embedding components
        output_dir (str): Directory to save the plot
        colormap (dict, optional): Dictionary mapping labels to colors
        show_plots (bool): Whether to display the plot
    """
    if len(component_names) < 3:
        print("Warning: Need at least 3 components for combined plot")
        return
    
    # Create subplots with fixed size
    fig = plt.figure(figsize=(16, 6))
    
    # 2D plot
    ax1 = fig.add_subplot(121)
    x = embedding_matrix[:, 0]
    y = embedding_matrix[:, 1]
    
    unique_labels = np.unique(labels)
    
    # Create color map with special handling for label -1
    colors = []
    for label in unique_labels:
        if label == -1 or label == "-1":
            colors.append('grey')
        else:
            colors.append(None)  # Will be filled with contrasting colors
    
    # Generate contrasting colors for non-grey labels
    non_grey_count = sum(1 for c in colors if c is None)
    if non_grey_count > 0:
        # Use specified colormap or default to Set3
        if colormap is not None:
            try:
                contrasting_colors = colormap(np.linspace(0, 1, non_grey_count))
            except Exception as e:
                print(f"Warning: Error using provided colormap: {e}, using default 'Set3'")
                contrasting_colors = plt.cm.Set3(np.linspace(0, 1, non_grey_count))
        else:
            contrasting_colors = plt.cm.Set3(np.linspace(0, 1, non_grey_count))
        color_idx = 0
        for i, color in enumerate(colors):
            if color is None:
                colors[i] = contrasting_colors[color_idx]
                color_idx += 1
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(x[mask], y[mask], 
                   c=[colors[i]], 
                   label=f'{label}', 
                   alpha=1.0, 
                   s=20,
                   marker='.')
    
    ax1.set_xlabel(f'Component {component_names[0]}')
    ax1.set_ylabel(f'Component {component_names[1]}')
    ax1.set_title('2D View (Components 0-1)')
    ax1.grid(True, alpha=0.3)
      # Ensure equal aspect ratio
    
    # 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    z = embedding_matrix[:, 2]
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax2.scatter(x[mask], y[mask], z[mask], 
                   c=[colors[i]], 
                   label=f'{label}', 
                   alpha=1.0, 
                   s=20,
                   marker='.')
    
    ax2.set_xlabel(f'Component {component_names[0]}')
    ax2.set_ylabel(f'Component {component_names[1]}')
    ax2.set_zlabel(f'Component {component_names[2]}')
    ax2.set_title('3D View (Components 0-2)')
    ax2.set_box_aspect([1,1,1])  # Set equal aspect ratio for 3D plot
    
    # Add legend with as many entries as possible without elongating the figure
    if len(unique_labels) <= 50:  # Reasonable limit for legend entries
        handles, labels_legend = ax1.get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=8)
    else:
        # For many labels, show only the most frequent ones
        label_counts = [(label, np.sum(labels == label)) for label in unique_labels]
        label_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count
        
        # Show top 30 most frequent labels
        top_labels = [label for label, count in label_counts[:30]]
        top_colors = [colors[list(unique_labels).index(label)] for label in top_labels]
        
        # Create custom legend with top labels only
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     markersize=6, label=f'{label} (n={np.sum(labels == label)})') 
                          for label, color in zip(top_labels, top_colors)]
        
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5), 
                  fontsize=7)
        print(f"Showing top 30 most frequent labels out of {len(unique_labels)} total labels")
    
    plt.tight_layout()
    
    # Save the combined plot
    output_path = os.path.join(output_dir, 'embedding_combined_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def create_individual_label_plots(embedding_matrix, labels, component_names, output_dir, colormap=None, show_plots=False):
    """
    Create individual plots for each label showing its location in the embedding space.
    
    Args:
        embedding_matrix (np.array): The embedding data
        labels (np.array): The label data for coloring
        component_names (list): Names of the embedding components
        output_dir (str): Directory to save the plots
        colormap (dict, optional): Dictionary mapping labels to colors
        show_plots (bool): Whether to display the plots
    """
    unique_labels = np.unique(labels)
    
    # Create individual plots directory
    individual_dir = os.path.join(output_dir, 'individual_labels')
    os.makedirs(individual_dir, exist_ok=True)
    
    print(f"Creating individual plots for {len(unique_labels)} labels...")
    
    for label in unique_labels:
        # Create figure with 2D and 3D subplots (fixed size)
        fig = plt.figure(figsize=(16, 6))
        
        # 2D plot
        ax1 = fig.add_subplot(121)
        x = embedding_matrix[:, 0]
        y = embedding_matrix[:, 1]
        
        # Plot all points in light gray
        ax1.scatter(x, y, c='lightgray', alpha=0.3, s=10, marker='.')
        
        # Highlight current label
        mask = labels == label
        if colormap is not None and label in colormap:
            color = colormap[label]
        elif label == "other":
            color = 'grey'  # Special handling for "other" label
        else:
            color = 'red'  # Default highlight color
        
        ax1.scatter(x[mask], y[mask], c=[color], alpha=1.0, s=20, marker='.')
        
        ax1.set_xlabel(f'Component {component_names[0]}')
        ax1.set_ylabel(f'Component {component_names[1]}')
        ax1.set_title(f'Label: {label} (2D View)')
        ax1.grid(True, alpha=0.3)
          # Ensure equal aspect ratio
        
        # 3D plot (if we have enough components)
        if len(component_names) >= 3:
            ax2 = fig.add_subplot(122, projection='3d')
            z = embedding_matrix[:, 2]
            
            # Plot all points in light gray
            ax2.scatter(x, y, z, c='lightgray', alpha=0.3, s=10, marker='.')
            
            # Highlight current label
            ax2.scatter(x[mask], y[mask], z[mask], c=[color], alpha=1.0, s=20, marker='.')
            
            ax2.set_xlabel(f'Component {component_names[0]}')
            ax2.set_ylabel(f'Component {component_names[1]}')
            ax2.set_zlabel(f'Component {component_names[2]}')
            ax2.set_title(f'Label: {label} (3D View)')
            ax2.set_box_aspect([1,1,1])  # Set equal aspect ratio for 3D plot
        else:
            # If not enough components for 3D, create a second 2D view with different components
            ax2 = fig.add_subplot(122)
            if len(component_names) >= 2:
                x2 = embedding_matrix[:, 1] if len(component_names) > 1 else embedding_matrix[:, 0]
                y2 = embedding_matrix[:, 2] if len(component_names) > 2 else embedding_matrix[:, 1]
                
                ax2.scatter(x2, y2, c='lightgray', alpha=0.3, s=10, marker='.')
                ax2.scatter(x2[mask], y2[mask], c=[color], alpha=1.0, s=20, marker='.')
                
                ax2.set_xlabel(f'Component {component_names[1] if len(component_names) > 1 else component_names[0]}')
                ax2.set_ylabel(f'Component {component_names[2] if len(component_names) > 2 else component_names[1]}')
                ax2.set_title(f'Label: {label} (Alternative 2D View)')
                ax2.grid(True, alpha=0.3)
                  # Ensure equal aspect ratio
        
        plt.tight_layout()
        
        # Save the plot
        safe_label = str(label).replace('/', '_').replace('\\', '_').replace('?', '_')
        output_path = os.path.join(individual_dir, f'label_{safe_label}_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print(f"  Saved individual plot for label '{label}' to: {output_path}")
    
    print(f"All individual plots saved to: {individual_dir}")


def create_subplot_overview(embedding_matrix, labels, component_names, output_dir, colormap=None, show_plots=False):
    """
    Create a single figure with 2D subplots for each label.
    
    Args:
        embedding_matrix (np.array): The embedding data
        labels (np.array): The label data for coloring
        component_names (list): Names of the embedding components
        output_dir (str): Directory to save the plot
        colormap (dict, optional): Dictionary mapping labels to colors
        show_plots (bool): Whether to display the plot
    """
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(n_labels)))
    rows = int(np.ceil(n_labels / cols))
    
    print(f"Creating subplot overview with {rows}x{cols} grid for {n_labels} labels...")
    
    # Create the main figure with fixed size
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single subplot case
    if n_labels == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    x = embedding_matrix[:, 0]
    y = embedding_matrix[:, 1]
    
    for i, label in enumerate(unique_labels):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Plot all points in light gray
        ax.scatter(x, y, c='lightgray', alpha=0.3, s=5, marker='.')
        
        # Highlight current label
        mask = labels == label
        if colormap is not None and label in colormap:
            color = colormap[label]
        elif label == "other":
            color = 'grey'  # Special handling for "other" label
        else:
            color = 'red'  # Default highlight color
        
        ax.scatter(x[mask], y[mask], c=[color], alpha=1.0, s=10, marker='.')
        
        ax.set_xlabel(f'Comp {component_names[0]}')
        ax.set_ylabel(f'Comp {component_names[1]}')
        ax.set_title(f'{label}', fontsize=10)
        ax.grid(True, alpha=0.3)
          # Ensure equal aspect ratio
        
        # Add count information
        count = np.sum(mask)
        ax.text(0.02, 0.98, f'n={count}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_labels, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'label_subplot_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()  # Close to free memory
    
    print(f"Subplot overview saved to: {output_path}")


def create_zoomed_plots(embedding_matrix, labels, component_names, output_dir, colormap=None, show_plots=False):
    """
    Create zoomed-in 2D plots for specific coordinate ranges.
    
    Args:
        embedding_matrix (np.array): The embedding data
        labels (np.array): The label data for coloring
        component_names (list): Names of the embedding components
        output_dir (str): Directory to save the plots
        colormap (dict, optional): Dictionary mapping labels to colors
        show_plots (bool): Whether to display the plots
    """
    if len(component_names) < 2:
        print("Warning: Need at least 2 components for zoomed plots")
        return
    
    # Define the zoom regions (component 0 range, component 1 range)
    zoom_regions = [
        ((-2, 0), (2, 4), "Region 1: Comp0(-2:0), Comp1(2:4)"),
        ((4, 8), (2, 4), "Region 2: Comp0(4:8), Comp1(2:4)"),
        ((8, 12), (2, 4), "Region 3: Comp0(8:12), Comp1(2:4)")
    ]
    
    x = embedding_matrix[:, 0]
    y = embedding_matrix[:, 1]
    
    # Create zoomed plots directory
    zoomed_dir = os.path.join(output_dir, 'zoomed_regions')
    os.makedirs(zoomed_dir, exist_ok=True)
    
    print(f"Creating zoomed plots for {len(zoom_regions)} regions...")
    
    for i, ((x_min, x_max), (y_min, y_max), title) in enumerate(zoom_regions):
        # Create mask for points in this region
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        
        if not np.any(mask):
            print(f"  No points found in region {i+1}, skipping...")
            continue
        
        # Create the plot with fixed size
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique labels in this region
        region_labels = labels[mask]
        unique_region_labels = np.unique(region_labels)
        
        # Create color map for this region
        colors = []
        if colormap is not None:
            for label in unique_region_labels:
                if label in colormap:
                    colors.append(colormap[label])
                elif label == "other":
                    colors.append('grey')
                else:
                    colors.append('lightblue')
        else:
            default_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_region_labels)))
            for j, label in enumerate(unique_region_labels):
                if label == -1 or label == "-1" or label == "other":
                    colors.append('grey')
                else:
                    colors.append(default_colors[j])
        
        # Plot each label in this region
        for j, label in enumerate(unique_region_labels):
            label_mask = (labels == label) & mask
            if np.any(label_mask):
                ax.scatter(x[label_mask], y[label_mask], 
                          c=[colors[j]], 
                          label=f'{label} (n={np.sum(label_mask)})', 
                          alpha=1.0, 
                          s=20,
                          marker='.')
        
        # Set axis limits to fit the actual data in this region
        region_x = x[mask]
        region_y = y[mask]
        
        # Add small padding (5% of range) to the data limits
        x_padding = (region_x.max() - region_x.min()) * 0.05
        y_padding = (region_y.max() - region_y.min()) * 0.05
        
        ax.set_xlim(region_x.min() - x_padding, region_x.max() + x_padding)
        ax.set_ylim(region_y.min() - y_padding, region_y.max() + y_padding)
        
        # Add labels and title
        ax.set_xlabel(f'Component {component_names[0]}')
        ax.set_ylabel(f'Component {component_names[1]}')
        
        # Update title to show actual data range
        actual_x_min, actual_x_max = region_x.min(), region_x.max()
        actual_y_min, actual_y_max = region_y.min(), region_y.max()
        title_with_range = f"{title}\nData range: Comp0({actual_x_min:.2f}:{actual_x_max:.2f}), Comp1({actual_y_min:.2f}:{actual_y_max:.2f})"
        ax.set_title(title_with_range)
        ax.grid(True, alpha=0.3)
          # Ensure equal aspect ratio
        
        # Add legend with as many entries as possible without elongating the figure
        if len(unique_region_labels) <= 30:  # Reasonable limit for zoomed plots
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # For many labels, show only the most frequent ones in this region
            region_label_counts = [(label, np.sum((labels == label) & mask)) for label in unique_region_labels]
            region_label_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count
            
            # Show top 20 most frequent labels in this region
            top_region_labels = [label for label, count in region_label_counts[:20]]
            top_region_colors = [colors[list(unique_region_labels).index(label)] for label in top_region_labels]
            
            # Create custom legend with top labels only
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                         markersize=6, label=f'{label} (n={np.sum((labels == label) & mask)})') 
                              for label, color in zip(top_region_labels, top_region_colors)]
            
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
                      fontsize=7)
            print(f"Showing top 20 most frequent labels out of {len(unique_region_labels)} total labels in region {i+1}")
        
        # Add region info
        total_points = np.sum(mask)
        ax.text(0.02, 0.98, f'Total points: {total_points}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(zoomed_dir, f'zoomed_region_{i+1}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        print(f"  Saved zoomed plot for region {i+1} to: {output_path}")
    
    print(f"All zoomed plots saved to: {zoomed_dir}")


def main():
    """Main function to run the embedding visualization."""
    parser = argparse.ArgumentParser(description='Plot embeddings from CSV file')
    parser.add_argument('--csv-file', required=True, help='Path to the CSV file containing embeddings')
    parser.add_argument('--output-dir', '-o', 
                       help='Parent directory for timestamped cluster_plots folder (default: current directory)')
    parser.add_argument('--show-plots', action='store_true', 
                       help='Display plots in addition to saving them (default: only save)')
    parser.add_argument('--label-mapping-csv', 
                       help='Path to CSV file for label mapping (must be used with --new-label-column)')
    parser.add_argument('--new-label-column', 
                       help='Column name in mapping CSV containing the new labels (must be used with --label-mapping-csv)')
    parser.add_argument('--colormap', 
                       help='Path to pickle file containing the colormap')
    parser.add_argument('--individual-plots', action='store_true',
                       help='Create individual plots for each label showing its location')
    parser.add_argument('--subplot-overview', action='store_true',
                       help='Create a single figure with 2D subplots for each label')
    parser.add_argument('--zoomed-plots', action='store_true',
                       help='Create zoomed-in 2D plots for specific coordinate ranges')
    
    args = parser.parse_args()
    
    # Validate label mapping arguments
    if (args.label_mapping_csv is None) != (args.new_label_column is None):
        print("Error: --label-mapping-csv and --new-label-column must be used together")
        sys.exit(1)
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found")
        sys.exit(1)
    
    # Check if label mapping CSV exists
    if args.label_mapping_csv and not os.path.exists(args.label_mapping_csv):
        print(f"Error: Label mapping CSV file '{args.label_mapping_csv}' not found")
        sys.exit(1)
    
    # Check if colormap pickle file exists
    if args.colormap and not os.path.exists(args.colormap):
        print(f"Error: Colormap pickle file '{args.colormap}' not found")
        sys.exit(1)
    
    # Create timestamped output directory: cluster_plots_YYYY-MM-DD_HH-mm-ss
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%m-%S")
    
    # Use output-dir as parent directory, or current directory if not specified
    parent_dir = args.output_dir if args.output_dir else "."
    output_dir = os.path.join(parent_dir, f"cluster_plots_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load the data
    print("Loading embedding data...")
    embedding_matrix, labels, component_names = load_embedding_data(
        args.csv_file, 
        args.label_mapping_csv, 
        args.new_label_column
    )
    
    # Load colormap if provided
    colormap = None
    if args.colormap:
        print("Loading colormap from pickle file...")
        colormap = load_colormap_from_pickle(args.colormap)
        if colormap is None:
            print("Warning: Failed to load colormap, using default")
    
    # Create plots
    print("Creating 2D plot...")
    create_2d_plot(embedding_matrix, labels, component_names, output_dir, colormap, args.show_plots)
    
    print("Creating 3D plot...")
    create_3d_plot(embedding_matrix, labels, component_names, output_dir, colormap, args.show_plots)
    
    print("Creating combined plot...")
    create_combined_plot(embedding_matrix, labels, component_names, output_dir, colormap, args.show_plots)
    
    # Create individual plots if requested
    if args.individual_plots:
        print("Creating individual label plots...")
        create_individual_label_plots(embedding_matrix, labels, component_names, output_dir, colormap, args.show_plots)
    
    # Create subplot overview if requested
    if args.subplot_overview:
        print("Creating subplot overview...")
        create_subplot_overview(embedding_matrix, labels, component_names, output_dir, colormap, args.show_plots)
    
    # Create zoomed plots if requested
    if args.zoomed_plots:
        print("Creating zoomed plots...")
        create_zoomed_plots(embedding_matrix, labels, component_names, output_dir, colormap, args.show_plots)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Plot files:")
    print("- embedding_2d_plot.png")
    print("- embedding_3d_plot.png") 
    print("- embedding_combined_plot.png")
    if args.individual_plots:
        print("- individual_labels/ (directory with individual label plots)")
    if args.subplot_overview:
        print("- label_subplot_overview.png")
    if args.zoomed_plots:
        print("- zoomed_regions/ (directory with zoomed region plots)")


if __name__ == "__main__":
    main()
