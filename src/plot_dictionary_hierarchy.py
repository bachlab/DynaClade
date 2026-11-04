import pandas as pd
import plotly.express as px
import pickle
import re
import os
import argparse
from utils import print_welcome_message, tictoc, print_color, bold_color

@tictoc
def create_radial_hierarchy_plot(csv_path, colors_pickle_path, output_filename='movement_hierarchy.html', output_dir='.', plot_title='Dictionary of Human Movements Under Threat in VR - by Ulises Serratos', hierarchy_levels=['L00', 'L01', 'L012']):
    """
    Reads movement data from a CSV and creates a radial sunburst chart,
    coloring the top-level categories based on a provided pickle file.

    Args:
        csv_path (str): The file path to the input CSV file.
        colors_pickle_path (str): The file path to the colors pickle file.
        output_filename (str): The output figure name (HTML filename, default: movement_hierarchy.html).
        output_dir (str): The output directory for the HTML file (default: current directory).
        plot_title (str): The plot title (default: Dictionary of Human Movements Under Threat in VR - by Ulises Serratos).
        hierarchy_levels (list): List of three hierarchy level column names (default: ['L00', 'L01', 'L012']).
    """
    try:
        # --- Clean file paths (remove quotes if present) ---
        csv_path = csv_path.strip().strip('"\'')
        colors_pickle_path = colors_pickle_path.strip().strip('"\'')
        
        # --- Debug: Print file paths ---
        print(f"CSV path: {repr(csv_path)}")
        print(f"Colors path: {repr(colors_pickle_path)}")
        print(f"File exists (CSV): {os.path.exists(csv_path)}")
        print(f"File exists (Colors): {os.path.exists(colors_pickle_path)}")
        
        # --- Load and Process Data ---
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df.dropna(subset=hierarchy_levels, inplace=True)

        df_unique = df[hierarchy_levels].drop_duplicates().reset_index(drop=True)

        # Add a single root node to the hierarchy.
        df_unique['root'] = 'Movements'

        # --- FONT STYLING TO SIMULATE RADIUS CHANGE ---
        df_unique['L0_styled'] = '<b><span style="font-size: 16px;">' + df_unique[hierarchy_levels[0]] + '</span></b>'
        df_unique['L1_styled'] = '<span style="font-size: 18px;">' + df_unique[hierarchy_levels[1]] + '</span>'
        df_unique['L2_styled'] = '<span style="font-size: 22px;">' + df_unique[hierarchy_levels[2]] + '</span>'

        # --- VALUE CALCULATION ---
        df_unique['values'] = 1

        # --- Load and Process Colors ---
        with open(colors_pickle_path, 'rb') as f:
            color_data = pickle.load(f)

        color_map = {}
        for label, color in color_data.items():
            if isinstance(color, tuple) and len(color) == 4:
                r, g, b, a = color
                color_map[str(label)] = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
            elif color == 'grey':
                color_map[str(label)] = 'grey'
        
        color_map['other'] = 'grey'

        # --- Create Plot (without automatic color arguments) ---
        fig = px.sunburst(
            df_unique,
            path=['root', 'L0_styled', 'L1_styled', 'L2_styled'],
            values='values',
            title=plot_title,
        )

        # --- Manual Color Mapping ---
        # We manually build a list of colors to ensure the root is styled correctly
        # and not overridden by the default color sequence.
        marker_colors = []
        # fig.data[0].ids provides the unique path for each slice in the plot
        for id_path in fig.data[0].ids:
            parts = id_path.split('/')
            if len(parts) == 1:  # This is the root node
                marker_colors.append('darkgrey')
            else:
                # The second part of the path is the styled L0 label
                styled_l0 = parts[1]
                # Extract the raw label from the HTML style span to look up its color
                match = re.search(r'>([^<]+)<', styled_l0)
                if match:
                    l0_label = match.group(1)
                    # Get the color from our map, default to grey if not found
                    marker_colors.append(color_map.get(l0_label, 'grey'))
                else:
                    marker_colors.append('grey')  # Fallback color

        # --- Layout and Trace Updates ---
        fig.update_layout(
            margin=dict(t=50, l=20, r=20, b=20),
            font=dict(family="Arial, sans-serif", size=14, color="black")
        )
        
        fig.update_traces(
            hoverinfo='label+percent parent+percent root',
            insidetextorientation='radial',
            outsidetextfont=dict(size=32, color='white'),  # Style the root text
            marker=dict(colors=marker_colors)  # Apply the manually created color list
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full output path
        output_path = os.path.join(output_dir, output_filename)
        
        fig.write_html(output_path, full_html=False, include_plotlyjs='cdn')

        print(f"Successfully created the plot! Please open '{output_path}' in your browser.")

    except FileNotFoundError as e:
        print(f"Error: The file '{e.filename}' was not found.")
    except KeyError as e:
        print(f"Error: A required column was not found in the CSV: {e}. Please ensure 'L0', 'L1', and 'L2' columns exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a radial hierarchy plot from movement data CSV and colors pickle file.')
    parser.add_argument('--csv', '--csv_path', type=str, required=True,
                        help='Path to the input CSV file containing movement data')
    parser.add_argument('--colors', '--colors_pickle', type=str, required=True,
                        help='Path to the colors pickle file')
    parser.add_argument('--output', '--output_file', type=str, default='movement_hierarchy.html',
                        help='Output figure name (HTML filename, default: movement_hierarchy.html)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for the HTML file (default: current directory)')
    parser.add_argument('--title', type=str, default='Dictionary of Human Movements Under Threat in VR - by Ulises Serratos',
                        help='Plot title (default: Dictionary of Human Movements Under Threat in VR - by Ulises Serratos)')
    parser.add_argument('--levels', nargs=3, default=['L0', 'L1', 'L2'],
                        help='Three hierarchy level column names (default: L0 L1 L2)')
    
    args = parser.parse_args()
    
    os.system("cls||clear")
    print_welcome_message()
    
    create_radial_hierarchy_plot(
        csv_path=args.csv,
        colors_pickle_path=args.colors,
        output_filename=args.output,
        output_dir=args.output_dir,
        plot_title=args.title,
        hierarchy_levels=args.levels
    )
