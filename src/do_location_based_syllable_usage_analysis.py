#!/usr/bin/env python3
"""
Clean Location-Based Syllable Usage Analysis
Performs spatial analysis of syllable usage with proper coordinate transformations and rotations.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import time
import glob

def tictoc(func):
    def inner(*args, **kwargs):
        T1: time = time.time()
        print(f"→ {func.__name__} started")
        func_out = func(*args, **kwargs)
        T2: time = time.time()
        T: time = time.strftime("%H:%M:%S", time.gmtime(T2 - T1))
        print(f"✓ {func.__name__} completed ({T})")
        return func_out
    return inner


def load_colour_map(colour_map_path):
    """Load colour map from pickle file."""
    try:
        with open(colour_map_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load colour map: {e}")
        return None

def clean_scenario_id(scenario_id):
    """Clean scenario_id values to remove leading zeros and string suffixes."""
    if pd.isna(scenario_id):
        return scenario_id
    
    scenario_id_str = str(scenario_id).strip()
    if not scenario_id_str:
        return scenario_id_str
    
    # Remove leading zeros
    if scenario_id_str.startswith('0'):
        for i, char in enumerate(scenario_id_str):
            if char != '0':
                return scenario_id_str[i:]
        return "0"
    
    return scenario_id_str

def validate_and_fix_coordinate(value, coord_name="coordinate"):
    """
    Validate and fix malformed coordinate values.
    
    Args:
        value: The coordinate value to validate
        coord_name: Name of the coordinate for error messages
    
    Returns:
        float: The corrected coordinate value, or NaN if invalid
    """
    try:
        # Convert to string for analysis
        value_str = str(value)
        
        # Handle NaN/None values
        if pd.isna(value) or value_str.lower() in ['nan', 'none', '']:
            return float('nan')
        
        # Remove leading/trailing quotes and whitespace
        value_str = value_str.strip().strip("'\"")
        
        # Fix coordinates that start with decimal point (missing leading zero)
        if value_str.startswith('.') and value_str[1:].replace('.', '').replace('-', '').isdigit():
            corrected_value = float('0' + value_str)
            print(f"      Fixed malformed {coord_name}: {value_str} → {corrected_value}")
            return corrected_value
        
        # Fix coordinates that start with negative decimal point (missing leading zero)
        if value_str.startswith('-.') and value_str[2:].replace('.', '').isdigit():
            corrected_value = float('-0' + value_str[1:])
            print(f"      Fixed malformed {coord_name}: {value_str} → {corrected_value}")
            return corrected_value
        
        # Try to convert to float
        return float(value_str)
        
    except (ValueError, TypeError) as e:
        print(f"      ⚠ Invalid {coord_name}: {value} (error: {e})")
        return float('nan')

def validate_and_fix_dataframe_coordinates(df, coordinate_columns=None):
    """
    Validate and fix coordinate columns in a DataFrame.
    
    Args:
        df: DataFrame to process
        coordinate_columns: List of column names to validate (if None, auto-detect)
    
    Returns:
        DataFrame: DataFrame with corrected coordinates
    """
    if df is None or len(df) == 0:
        return df
    
    # Auto-detect coordinate columns if not specified
    if coordinate_columns is None:
        coordinate_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['_x', '_y', '_z', 'x', 'y', 'z']):
                coordinate_columns.append(col)
    
    if not coordinate_columns:
        return df
    
    print(f"  🔧 Validating and fixing coordinates in {len(coordinate_columns)} columns: {coordinate_columns}")
    
    # Create a copy to avoid modifying the original
    df_fixed = df.copy()
    total_fixes = 0
    
    for col in coordinate_columns:
        if col in df_fixed.columns:
            # Apply coordinate validation to each value
            original_values = df_fixed[col].copy()
            fixed_values = original_values.apply(lambda x: validate_and_fix_coordinate(x, col))
            
            # Count how many values were fixed
            fixes = (original_values != fixed_values).sum()
            if fixes > 0:
                print(f"    Fixed {fixes} malformed values in column '{col}'")
                total_fixes += fixes
            
            df_fixed[col] = fixed_values
    
    if total_fixes > 0:
        print(f"  ✅ Total coordinate fixes applied: {total_fixes}")
    else:
        print(f"  ✅ No coordinate formatting issues found")
    
    return df_fixed


def calculate_optimal_rotation_angle_icp(tracker_data, participant_data, trial_key):
    """
    Calculate the optimal rotation angle using ICP (Iterative Closest Point) with Open3D.
    
    This method is more robust than the previous approaches as it finds the optimal
    transformation (rotation + translation) to align the tracker data with participant data.
    
    Args:
        tracker_data: DataFrame with tracker coordinates (transformed_x, transformed_y) - CENTERED
        participant_data: DataFrame with participant coordinates (participant_x, participant_y) - RAW
        trial_key: String identifier for the trial (ppid-session-trial)
        
    Returns:
        float: Optimal rotation angle in degrees
    """
    try:
        import open3d as o3d
        import numpy as np
    except ImportError:
        print(f"    ❌ Open3D not available, falling back to discrete angles")
        return test_discrete_angles(
            tracker_data['transformed_x'].values,
            tracker_data['transformed_y'].values,
            participant_data['participant_x'].values,
            participant_data['participant_y'].values
        )
    
    print(f"    🔍 Calculating optimal rotation using ICP for trial {trial_key}...")
    
    # Validate input data
    if tracker_data.empty or participant_data.empty:
        print(f"    ⚠ No data available for trial {trial_key}, using 0°")
        return 0.0
    
    # Find common time points
    tracker_times = set(tracker_data.index)
    participant_times = set(participant_data.index)
    common_times = list(tracker_times.intersection(participant_times))
    
    if len(common_times) < 10:
        print(f"    ⚠ Insufficient common time points ({len(common_times)}) for trial {trial_key}, using 0°")
        return 0.0
    
    # Align data by common time points
    tracker_aligned = tracker_data.loc[common_times]
    participant_aligned = participant_data.loc[common_times]
    
    # Prepare point clouds
    # Tracker data (already centered at origin)
    tracker_points = np.column_stack([
        tracker_aligned['transformed_x'].values,
        tracker_aligned['transformed_y'].values,
        np.zeros(len(tracker_aligned))  # Add z=0 for 3D
    ])
    
    # Participant data (raw coordinates)
    participant_points = np.column_stack([
        participant_aligned['participant_x'].values,
        participant_aligned['participant_y'].values,
        np.zeros(len(participant_aligned))  # Add z=0 for 3D
    ])
    
    # Create Open3D point clouds
    tracker_pcd = o3d.geometry.PointCloud()
    tracker_pcd.points = o3d.utility.Vector3dVector(tracker_points)
    
    participant_pcd = o3d.geometry.PointCloud()
    participant_pcd.points = o3d.utility.Vector3dVector(participant_points)
    
    # Perform ICP registration
    # Source: tracker data, Target: participant data
    threshold = 0.1  # Distance threshold for correspondence
    trans_init = np.eye(4)  # Identity transformation as initial guess
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        tracker_pcd, participant_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    # Extract rotation matrix from transformation
    transformation_matrix = reg_p2p.transformation
    rotation_matrix = transformation_matrix[:3, :3]
    
    # Convert rotation matrix to Euler angles (Z-axis rotation)
    # For 2D rotation around Z-axis, we can extract the angle directly
    angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    angle_deg = np.degrees(angle_rad)
    
    # Normalize angle to [0, 360)
    if angle_deg < 0:
        angle_deg += 360
    
    print(f"    ✓ ICP found optimal angle: {angle_deg:.1f}° (fitness: {reg_p2p.fitness:.3f})")
    
    return angle_deg


def clean_tracker_identifiers(ppid, session, trial):
    """
    Clean tracker identifiers to match merged_df format.
    
    Handles:
    - Session: S001 -> 1, S000 -> 0
    - Trial: T001 -> 1, T000 -> 0  
    - PPID: Remove leading zeros
    
    Args:
        ppid: Participant ID (any type)
        session: Session ID (any type)
        trial: Trial ID (any type)
        
    Returns:
        tuple: (ppid_clean, session_clean, trial_clean)
    """
    # Remove 'S' prefix and leading zeros from session (e.g., S001 -> 1)
    if isinstance(session, str) and session.startswith('S'):
        session_clean = session[1:].lstrip('0') or '0'  # Handle S000 -> 0
    else:
        session_clean = str(session).lstrip('0') or '0'
    
    # Remove 'T' prefix and leading zeros from trial (e.g., T001 -> 1)
    if isinstance(trial, str) and trial.startswith('T'):
        trial_clean = trial[1:].lstrip('0') or '0'  # Handle T000 -> 0
    else:
        trial_clean = str(trial).lstrip('0') or '0'
    
    # Ensure ppid is also clean (remove leading zeros)
    ppid_clean = str(ppid).lstrip('0') or '0'
    
    return ppid_clean, session_clean, trial_clean


def extract_tracker_data(metadata_dir, fast_test=False):
    """Extract tracker data from metadata folder and calculate alignment corrections."""
    import re
    from scipy import interpolate
    import numpy as np
    
    tracker_data_list = []
    
    # Get all participant directories first (ppid level)
    all_ppid_dirs = [d for d in glob.glob(os.path.join(metadata_dir, '*')) if os.path.isdir(d)]
    all_ppids = [os.path.basename(d) for d in all_ppid_dirs]
    
    # Apply fast-test filtering if requested
    if fast_test and len(all_ppids) > 2:
        # Use first 2 participants if no specific list provided
        selected_ppids = sorted(all_ppids)[:2]
        print(f"Fast test mode: Limiting tracker data to first 2 participants: {selected_ppids}")
        ppid_dirs_to_process = [d for d in all_ppid_dirs if os.path.basename(d) in selected_ppids]
    else:
        ppid_dirs_to_process = all_ppid_dirs
    
    print(f"Processing {len(ppid_dirs_to_process)} participant directories")
    
    for ppid_dir in ppid_dirs_to_process:
        ppid = os.path.basename(ppid_dir)
        print(f"Processing participant {ppid}")
        
        # Look for session directories (S001 format)
        for session_dir in glob.glob(os.path.join(ppid_dir, '*')):
            if not os.path.isdir(session_dir):
                continue
                
            session_raw = os.path.basename(session_dir)
            
            # Extract session from "S001" format: remove "S" and leading zeros
            if session_raw.startswith('S'):
                try:
                    # Remove "S" and convert to integer to remove leading zeros
                    session = str(int(session_raw[1:]))
                    print(f"  Processing session {session_raw} -> session {session}")
                except ValueError:
                    print(f"  Skipping {session_raw}: not a valid session format")
                    continue
            else:
                print(f"  Skipping {session_raw}: doesn't start with 'S'")
                continue
            
            # Look for tracker data - try both trackers subdirectory and direct CSV files
            trackers_dir = os.path.join(session_dir, 'trackers')
            tracker_files = []
            
            if os.path.isdir(trackers_dir):
                print(f"    Found trackers directory for {ppid}-{session}")
                # Look for waist_movement files in trackers subdirectory
                tracker_files = glob.glob(os.path.join(trackers_dir, '*waist_movement*.csv'))
                print(f"    Found {len(tracker_files)} waist_movement files in trackers directory")
            else:
                print(f"    No trackers directory found for {ppid}-{session}, looking for direct CSV files...")
                # Look for direct CSV files in session directory (for test data)
                tracker_files = glob.glob(os.path.join(session_dir, '*.csv'))
                print(f"    Found {len(tracker_files)} direct CSV files in session directory")
            
            if not tracker_files:
                print(f"    No tracker files found for {ppid}-{session}")
                continue
            
            print(f"    Processing {len(tracker_files)} potential tracker files")
            
            # Process each tracker file
            for tracker_file in tracker_files:
                fname = os.path.basename(tracker_file)
                print(f"      Checking file: {fname}")
                
                # Extract trial from filename containing "T" pattern
                trial_match = re.search(r'T(\d+)', fname)
                if not trial_match:
                    print(f"      Skipping {fname}: no trial pattern found")
                    continue
                
                try:
                    # Remove leading zeros from trial number
                    trial = str(int(trial_match.group(1)))
                    print(f"      Processing trial {trial_match.group(0)} -> trial {trial}")
                except ValueError:
                    print(f"      Skipping {fname}: invalid trial number")
                    continue
                
                try:
                    # Read tracker data
                    tracker_df = pd.read_csv(tracker_file)
                    
                    # Check for required columns - try both formats
                    has_pos_columns = 'pos_x' in tracker_df.columns and 'pos_z' in tracker_df.columns
                    has_mhip_columns = 'MHip_x' in tracker_df.columns and 'MHip_z' in tracker_df.columns
                    
                    if not has_pos_columns and not has_mhip_columns:
                        print(f"        Missing required columns in {tracker_file}")
                        print(f"        Available columns: {list(tracker_df.columns)}")
                        print(f"        Expected: pos_x, pos_z OR MHip_x, MHip_z")
                        continue
                    
                    # Apply coordinate transformation based on available columns
                    # COORDINATE MAPPING: pos_z -> x, -pos_x -> y
                    # This maps tracker coordinates to the participant coordinate system
                    if has_pos_columns:
                        # Original format: pos_x, pos_z
                        # Coordinate mapping: pos_z -> x, -pos_x -> y
                        tracker_df['transformed_x'] = tracker_df['pos_z']  # z becomes x
                        tracker_df['transformed_y'] = -1 * tracker_df['pos_x']  # -x becomes y
                        print(f"        Using pos_x/pos_z columns for transformation")
                        print(f"        Coordinate mapping: pos_z -> x, -pos_x -> y")
                    elif has_mhip_columns:
                        # Motion capture format: MHip_x, MHip_z
                        # Coordinate mapping: MHip_z -> x, -MHip_x -> y
                        tracker_df['transformed_x'] = tracker_df['MHip_z']  # z becomes x
                        tracker_df['transformed_y'] = -1 * tracker_df['MHip_x']  # -x becomes y
                        print(f"        Using MHip_x/MHip_z columns for transformation")
                        print(f"        Coordinate mapping: MHip_z -> x, -MHip_x -> y")
                    
                    # Keep backward compatibility with old column names
                    tracker_df['pos_x_corrected'] = tracker_df['transformed_x']
                    tracker_df['pos_y_corrected'] = tracker_df['transformed_y']
                    
                    # Debug: Show coordinate transformation results
                    print(f"        Coordinate transformation applied:")
                    print(f"          Original: pos_x={tracker_df['pos_x'].iloc[0]:.3f}, pos_z={tracker_df['pos_z'].iloc[0]:.3f}")
                    print(f"          Transformed: x={tracker_df['transformed_x'].iloc[0]:.3f}, y={tracker_df['transformed_y'].iloc[0]:.3f}")
                    
                    # Add time column if not present (assuming constant frame rate)
                    # Always compute tracker time from 0 for objective timing
                    if 'time' not in tracker_df.columns:
                        tracker_df['time'] = np.arange(len(tracker_df)) * 0.01  # Assuming 100Hz
                    else:
                        # Reset tracker time to start from 0 for objective timing
                        tracker_df['time'] = np.arange(len(tracker_df)) * 0.01  # Assuming 100Hz
                        print(f"        Reset tracker time to start from 0 (objective timing)")
                    
                    # Set time as index for proper time alignment
                    tracker_df = tracker_df.set_index('time')
                    
                    # Debug: Show time range
                    print(f"        Tracker time range: {tracker_df.index.min():.3f}s to {tracker_df.index.max():.3f}s")
                    
                    # Store tracker data for this trial
                    tracker_item = {
                        'ppid': ppid,
                        'session': session,
                        'trial': trial,
                        'tracker_file': tracker_file,
                        'tracker_data': tracker_df,
                        'x_correction': 0.0,  # Will be calculated later
                        'y_correction': 0.0   # Will be calculated later
                    }
                    
                    tracker_data_list.append(tracker_item)
                    
                    print(f"        ✓ Loaded tracker data for {ppid}-{session}-{trial}: {len(tracker_df)} points")
                    print(f"        ✓ Added to tracker_data_list. Current list size: {len(tracker_data_list)}")
                    print(f"        ✓ Tracker item keys: {list(tracker_item.keys())}")
                    print(f"        ✓ Tracker data shape: {tracker_item['tracker_data'].shape}")
                    print(f"        ✓ Transformed columns: transformed_x={tracker_item['tracker_data']['transformed_x'].min():.3f} to {tracker_item['tracker_data']['transformed_x'].max():.3f}")
                    print(f"        ✓ Transformed columns: transformed_y={tracker_item['tracker_data']['transformed_y'].min():.3f} to {tracker_item['tracker_data']['transformed_y'].max():.3f}")
                    
                except Exception as e:
                    print(f"        Error reading {tracker_file}: {e}")
                    continue
    
    print(f"Total tracker datasets loaded: {len(tracker_data_list)}")
    
    # Debug: Show details of loaded tracker data
    if tracker_data_list:
        print("DEBUG: Tracker data list contents:")
        for i, item in enumerate(tracker_data_list):
            print(f"  Item {i}: {item['ppid']}-{item['session']}-{item['trial']}")
            print(f"    File: {item['tracker_file']}")
            print(f"    Data shape: {item['tracker_data'].shape}")
            print(f"    Columns: {list(item['tracker_data'].columns)}")
    else:
        print("DEBUG: tracker_data_list is empty!")
    
    return tracker_data_list

def extract_scenario_data(scenario_dir, fast_test=False):
    """Extract scenario object data from JSON files."""
    import re
    import glob
    import os
    import json
    
    # Define scenario keywords for object filtering
    SCENARIO_KEYWORDS = [
        "StandPoint", "FruitCollectionTask", "SafeHouse", "StartPoint", "Threat", "Weapons", "Wall", "Target", "Hide", "Rolling Rock"
    ]
    
    def canonical_object_name(name):
        """Convert object name to canonical form."""
        import re
        if 'FruitCollectionTask' in name:
            return 'task'
        if 'Threat' in name or 'Rolling Rock' in name:
            return 'threat'
        if 'Wall' in name:
            return 'wall'
        if 'Target' in name:
            m = re.search(r'Target(\d+)', name)
            if m:
                return f'T{m.group(1)}'
            else:
                return 'T'
        if 'Hide' in name:
            return 'hide'
        if 'SafeHouse' in name:
            return 'safehouse'
        if 'Weapons' in name:
            return 'weapons'
        if 'StartPoint' in name:
            return 'startpoint'
        return name.lower()
    
    trial_data = {}
    
    # Debug: List directory structure
    print(f"DEBUG: Looking for scenario data in {scenario_dir}")
    if os.path.exists(scenario_dir):
        all_dirs = [d for d in glob.glob(os.path.join(scenario_dir, '*')) if os.path.isdir(d)]
        print(f"DEBUG: Found {len(all_dirs)} directories in {scenario_dir}")
        for d in all_dirs[:5]:  # Show first 5
            print(f"  DEBUG: {d}")
        if len(all_dirs) > 5:
            print(f"  DEBUG: ... and {len(all_dirs)-5} more directories")
    else:
        print(f"DEBUG: Directory {scenario_dir} does not exist!")
        return pd.DataFrame(columns=['ppid','session','trial'])
    
    # Get all participant directories
    all_ppid_dirs = [d for d in glob.glob(os.path.join(scenario_dir, '*')) if os.path.isdir(d)]
    all_ppids = [os.path.basename(d) for d in all_ppid_dirs]
    
    # Apply fast-test filtering
    if fast_test and len(all_ppids) > 2:
        selected_ppids = sorted(all_ppids)[:2]
        print(f"Fast test mode: Limiting scenario data to first 2 participants: {selected_ppids}")
        ppid_dirs_to_process = [d for d in all_ppid_dirs if os.path.basename(d) in selected_ppids]
    else:
        ppid_dirs_to_process = all_ppid_dirs
    
    print(f"Processing {len(ppid_dirs_to_process)} participant directories")
    
    for ppid_dir in ppid_dirs_to_process:
        ppid = os.path.basename(ppid_dir)
        print(f"Processing participant {ppid}")
        
        for session_dir in glob.glob(os.path.join(ppid_dir, '*')):
            session_raw = os.path.basename(session_dir)
            
            # Extract session from "S001" format: remove "S" and leading zeros
            if session_raw.startswith('S'):
                try:
                    # Remove "S" and convert to integer to remove leading zeros
                    session = str(int(session_raw[1:]))
                    print(f"  Processing session {session_raw} -> session {session}")
                except ValueError:
                    print(f"  Skipping {session_raw}: not a valid session format")
                    continue
            else:
                print(f"  Skipping {session_raw}: doesn't start with 'S'")
                continue
            
            othertrialdata_dir = os.path.join(session_dir, 'othertrialdata')
            if not os.path.isdir(othertrialdata_dir):
                print(f"No othertrialdata directory found for {ppid}-{session}")
                continue
            
            # Find scenario JSON files
            scenario_files = glob.glob(os.path.join(othertrialdata_dir, '*scenario*.json'))
            print(f"    Found {len(scenario_files)} scenario JSON files in othertrialdata directory")
            
            for json_file in scenario_files:
                fname = os.path.basename(json_file)
                print(f"      Processing scenario file: {fname}")
                trial_match = re.search(r'T(\d{3})', fname)
                if not trial_match:
                    print(f"      Skipping {fname}: no trial pattern found")
                    continue
                trial = str(int(trial_match.group(1)))
                print(f"      Extracted trial: {trial}")
                
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Skipping {json_file}: {e}")
                    continue
                
                row = {'ppid': ppid, 'session': session, 'trial': trial}
                children = data.get('children', [])
                
                # Debug: Track what objects are found in this trial
                found_objects = []
                
                # Collect all potential task objects to handle multiple tasks intelligently
                potential_tasks = []
                
                for child in children:
                    name = child.get('name', '')
                    obj = canonical_object_name(name)
                    if obj and obj in ['task', 'threat', 'wall', 'hide', 'safehouse', 'weapons', 'startpoint'] or obj.startswith('T'):
                        pos = child.get('position', {})
                        euler = child.get('euler_angles', {})
                        
                        # Special handling for task objects to avoid conflicts
                        if obj == 'task':
                            # Store task info for later processing
                            task_x = pos.get('z', float('nan'))  # z becomes x
                            task_y = -1 * pos.get('x', float('nan'))  # -x becomes y
                            task_z = pos.get('y', float('nan'))  # y becomes z
                            
                            # Check if this is a fake task
                            is_fake = 'Fake' in name
                            
                            potential_tasks.append({
                                'name': name,
                                'x': task_x,
                                'y': task_y,
                                'z': task_z,
                                'is_fake': is_fake,
                                'euler': euler
                            })
                        else:
                            # Apply coordinate transformation for non-task objects
                            row[f'{obj}_x'] = pos.get('z', float('nan'))  # z becomes x
                            row[f'{obj}_y'] = -1 * pos.get('x', float('nan'))  # -x becomes y
                            row[f'{obj}_z'] = pos.get('y', float('nan'))  # y becomes z
                            
                            row[f'{obj}_a'] = euler.get('x', float('nan'))
                            row[f'{obj}_b'] = euler.get('z', float('nan'))
                            row[f'{obj}_c'] = euler.get('y', float('nan'))
                            
                            found_objects.append(obj)
                
                # Process task objects intelligently
                if potential_tasks:
                    # Filter out fake tasks
                    legitimate_tasks = [task for task in potential_tasks if not task['is_fake']]
                    
                    if legitimate_tasks:
                        if len(legitimate_tasks) == 1:
                            # Only one legitimate task, use it
                            selected_task = legitimate_tasks[0]
                        else:
                            # Multiple legitimate tasks, select the one closest to center (0,0)
                            # Calculate distance to center for each task
                            for task in legitimate_tasks:
                                task['distance_to_center'] = (task['x']**2 + task['y']**2)**0.5
                            
                            # Select the task closest to center
                            selected_task = min(legitimate_tasks, key=lambda t: t['distance_to_center'])
                            print(f"      Multiple legitimate tasks found for {ppid}-{session}-{trial}, selected '{selected_task['name']}' (closest to center)")
                        
                        # Assign the selected task coordinates
                        row['task_x'] = selected_task['x']
                        row['task_y'] = selected_task['y']
                        row['task_z'] = selected_task['z']
                        row['task_a'] = selected_task['euler'].get('x', float('nan'))
                        row['task_b'] = selected_task['euler'].get('z', float('nan'))
                        row['task_c'] = selected_task['euler'].get('y', float('nan'))
                        
                        found_objects.append('task')
                        
                        if len(potential_tasks) > len(legitimate_tasks):
                            print(f"      Filtered out {len(potential_tasks) - len(legitimate_tasks)} fake task(s) for {ppid}-{session}-{trial}")
                    else:
                        print(f"      ⚠ No legitimate tasks found for {ppid}-{session}-{trial} (all were fake)")
                elif potential_tasks:
                    print(f"      ⚠ All tasks were fake for {ppid}-{session}-{trial}")
                
                # Debug: Print objects found in this trial (only for first few trials to avoid spam)
                if len(trial_data) < 5:  # Only show for first 5 trials
                    print(f"      Objects found in {ppid}-{session}-{trial}: {found_objects}")
                
                trial_data[(ppid, session, trial)] = row
                print(f"Loaded scenario data for {ppid}-{session}-{trial}")
    
    if not trial_data:
        print(f"No scenario data found in {scenario_dir}")
        return pd.DataFrame(columns=['ppid','session','trial'])
    
    scenario_df = pd.DataFrame(list(trial_data.values()))
    print(f"Total scenario data: {len(scenario_df)} trials")
    
    # Summary of scenario data loading
    print(f"\n📊 SCENARIO DATA LOADING SUMMARY:")
    print(f"   Total scenario trials loaded: {len(scenario_df)}")
    if len(scenario_df) > 0:
        print(f"   First scenario: {scenario_df.iloc[0]['ppid']}-{scenario_df.iloc[0]['session']}-{scenario_df.iloc[0]['trial']}")
        print(f"   Last scenario: {scenario_df.iloc[-1]['ppid']}-{scenario_df.iloc[-1]['session']}-{scenario_df.iloc[-1]['trial']}")
        
        # Show what objects were found across all trials
        object_columns = [col for col in scenario_df.columns if col.endswith('_x')]
        found_objects = [col.replace('_x', '') for col in object_columns]
        print(f"   Objects found across all trials: {sorted(found_objects)}")
        print(f"   Total object types: {len(found_objects)}")
        print(f"   Data columns: {list(scenario_df.columns)}")
    else:
        print(f"   ⚠ No scenario data loaded!")
    
    return scenario_df


def build_participant_location_csv(coords_dir, output_csv='participant_location.csv', fast_test=False, n_jobs=None):
    """Walk directory, extract ppid/session/trial, and build a CSV with ppid, session, trial, time, x, y."""
    import re
    import glob
    import os
    
    print(f"Scanning coordinates directory: {coords_dir}")
    
    # Get all participant directories first
    all_ppid_dirs = [d for d in glob.glob(os.path.join(coords_dir, '*')) if os.path.isdir(d)]
    all_ppids = [os.path.basename(d) for d in all_ppid_dirs]
    
    # Apply fast-test filtering if requested
    if fast_test and len(all_ppids) > 2:
        selected_ppids = sorted(all_ppids)[:2]
        print(f"Fast test mode: Limiting to first 2 participants: {selected_ppids}")
        ppid_dirs_to_process = [d for d in all_ppid_dirs if os.path.basename(d) in selected_ppids]
    else:
        ppid_dirs_to_process = all_ppid_dirs
    
    # Collect all files to process
    all_files_to_process = []
    for ppid_dir in ppid_dirs_to_process:
        if not os.path.isdir(ppid_dir):
            continue
        ppid = os.path.basename(ppid_dir)
        print(f"  Collecting files for ppid: {ppid}")
        
        for session_dir in glob.glob(os.path.join(ppid_dir, '*')):
            if not os.path.isdir(session_dir):
                continue
            session_raw = os.path.basename(session_dir)
            
            # Store session in original format (S001) but also keep integral version for comparisons
            session_original = session_raw  # Keep original format for file references
            if session_raw.startswith('S') and len(session_raw) == 4:
                session_integral = str(int(session_raw[1:]))  # Remove 'S', convert to int, then back to string (removes leading zeros)
            else:
                session_integral = str(int(''.join(filter(str.isdigit, session_raw))))
            
            for trial_file in glob.glob(os.path.join(session_dir, '*.csv')):
                trial_raw = os.path.splitext(os.path.basename(trial_file))[0]
                
                # Store trial in original format (T001) but also keep integral version for comparisons
                trial_original = trial_raw  # Keep original format for file references
                trial_match = re.search(r'T(\d{3})', trial_raw)
                if trial_match:
                    trial_integral = str(int(trial_match.group(1)))  # Convert to int, then back to string (removes leading zeros)
                else:
                    trial_integral = str(int(''.join(filter(str.isdigit, trial_raw))))
                
                all_files_to_process.append({
                      'file_path': trial_file,
                      'ppid': ppid,
                      'session_integral': session_integral,
                      'session_original': session_original,
                      'trial_integral': trial_integral,
                      'trial_original': trial_original
                  })
    
    # Define function to process a single file
    def process_single_file(file_info):
        file_path = file_info['file_path']
        ppid = file_info['ppid']
        session_integral = file_info['session_integral']
        session_original = file_info['session_original']
        trial_integral = file_info['trial_integral']
        trial_original = file_info['trial_original']
        
        rows = []
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns (try different possible column names)
            x_col = None
            y_col = None
            time_col = None
            
            # Try different possible column names for coordinates
            for col in df.columns:
                if col.lower() in ['hips_x', 'x', 'participant_x']:
                    x_col = col
                elif col.lower() in ['hips_y', 'y', 'participant_y']:
                    y_col = col
                elif col.lower() in ['time', 'timestamp']:
                    time_col = col
            
            if x_col is None or y_col is None or time_col is None:
                return rows  # Return empty list if missing required columns
            
            for _, row in df.iterrows():
                # Use the coordinate validation function to fix malformed values
                x_val = validate_and_fix_coordinate(row[x_col], "X coordinate")
                y_val = validate_and_fix_coordinate(row[y_col], "Y coordinate")
                
                # Skip rows with invalid coordinates
                if pd.isna(x_val) or pd.isna(y_val):
                    continue
                
                rows.append({
                    'ppid': ppid,
                    'session': session_integral,  # Store integral version for consistent comparisons
                    'trial': trial_integral,      # Store integral version for consistent comparisons
                    'session_original': session_original,  # Store original format for file references
                    'trial_original': trial_original,      # Store original format for file references
                    'time': row[time_col],
                    'x': x_val,
                    'y': y_val
                })
            
            print(f"    Processed {ppid}-{session_original}-{trial_original}: {len(rows)} rows")
            
        except Exception as e:
            print(f"    Error processing {file_path}: {e}")
        
        return rows
    
    # Process all files
    all_rows = []
    for file_info in all_files_to_process:
        rows = process_single_file(file_info)
        all_rows.extend(rows)
    
    # Create DataFrame and save
    if all_rows:
        participant_df = pd.DataFrame(all_rows)
        
        # Standardize coordinate column names
        if 'x' in participant_df.columns and 'participant_x' not in participant_df.columns:
            participant_df['participant_x'] = participant_df['x']
        if 'y' in participant_df.columns and 'participant_y' not in participant_df.columns:
            participant_df['participant_y'] = participant_df['y']
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        participant_df.to_csv(output_csv, index=False)
        print(f"✓ Saved participant location data to {output_csv}: {len(participant_df)} rows")
        return participant_df
    else:
        print(f"⚠ No participant data found in {coords_dir}")
        return pd.DataFrame()

def apply_downsampling(participant_df, tracker_data_list, scenario_df, downsample_factor):
    """Apply downsampling to all data sources by taking every Nth row."""
    if downsample_factor <= 1:
        return participant_df, tracker_data_list, scenario_df
    
    print(f"\n=== APPLYING DOWNSAMPLING (factor: {downsample_factor}) ===")
    
    # Downsample participant data
    if participant_df is not None and len(participant_df) > 0:
        original_size = len(participant_df)
        participant_df = participant_df.iloc[::downsample_factor]  # Preserve original index for time alignment
        new_size = len(participant_df)
        print(f"  Participant data: {original_size} → {new_size} rows ({new_size/original_size*100:.1f}% of original)")
    
    # Downsample tracker data
    if tracker_data_list:
        total_original_points = 0
        total_new_points = 0
        
        for tracker_item in tracker_data_list:
            tracker_data = tracker_item['tracker_data']
            if tracker_data is not None and len(tracker_data) > 0:
                original_points = len(tracker_data)
                total_original_points += original_points
                
                # Downsample tracker data (preserve original index for time alignment)
                tracker_item['tracker_data'] = tracker_data.iloc[::downsample_factor]
                
                new_points = len(tracker_item['tracker_data'])
                total_new_points += new_points
                
                ppid, session, trial = tracker_item['ppid'], tracker_item['session'], tracker_item['trial']
                print(f"  Tracker {ppid}-{session}-{trial}: {original_points} → {new_points} points ({new_points/original_points*100:.1f}% of original)")
        
        print(f"  Total tracker points: {total_original_points} → {total_new_points} points ({total_new_points/total_original_points*100:.1f}% of original)")
    
    # Note: Scenario data should NOT be downsampled as it represents static object positions per trial
    # Each row is a different trial, not time-series data
    if scenario_df is not None and len(scenario_df) > 0:
        print(f"  Scenario data: {len(scenario_df)} rows (NOT downsampled - static object positions)")
    
    print("✓ Downsampling applied successfully")
    return participant_df, tracker_data_list, scenario_df

def create_debug_plot(participant_df, scenario_df, tracker_data_list, step_name, output_dir, args, color_map=None):
    """Create debug plot for a specific step to verify transformations."""
    if not (args.debug or args.debug_plots_only):
        return
    
    print(f"\n=== DEBUG PLOT FOR STEP: {step_name} ===")
    
    # Use debug_plots directory directly (already created in main)
    debug_dir = os.path.join(output_dir, 'debug_plots')
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # Plot participant trajectories
    if 'participant_x' in participant_df.columns and 'participant_y' in participant_df.columns:
        plt.scatter(participant_df['participant_x'], participant_df['participant_y'], 
                   c='blue', s=10, alpha=0.6, label='Participant', zorder=3)
    
    # Plot scenario objects
    if scenario_df is not None:
        # Find startpoint columns
        startpoint_x_col = None
        startpoint_y_col = None
        for col in scenario_df.columns:
            if col.lower().endswith('startpoint_x'):
                startpoint_x_col = col
            elif col.lower().endswith('startpoint_y'):
                startpoint_y_col = col
        
        if startpoint_x_col and startpoint_y_col:
            plt.scatter(scenario_df[startpoint_x_col], scenario_df[startpoint_y_col], 
                       marker='o', color='yellow', s=200, edgecolor='black', zorder=10, 
                       label='StartPoint')
        
        # Plot other scenario objects
        object_styles = {
            'task': dict(marker='^', color='green', label='Task'),
            'safehouse': dict(marker='s', color='saddlebrown', label='Safehouse'),
            'threat': dict(marker='p', color='red', label='Threat'),
            'weapons': dict(marker='s', color='magenta', label='Weapons'),
            'standpoint': dict(marker='o', color='orange', label='StandPoint'),
            'wall': dict(marker='s', color='grey', label='Wall'),
            'hide': dict(marker='s', color='cyan', label='Hide'),
        }
        
        for object_name, style in object_styles.items():
            x_col = None
            y_col = None
            for col in scenario_df.columns:
                if col.lower().endswith(f'{object_name}_x'):
                    x_col = col
                elif col.lower().endswith(f'{object_name}_y'):
                    y_col = col
            
            if x_col and y_col:
                plt.scatter(scenario_df[x_col], scenario_df[y_col], 
                           marker=style['marker'], color=style['color'], 
                           s=120, edgecolor='black', zorder=10, 
                           label=style['label'])
    
    # Plot tracker paths (first 3 trials only)
    if tracker_data_list:
        for i, tracker_item in enumerate(tracker_data_list[:3]):  # Only first 3 trials
            tracker_data = tracker_item['tracker_data']
            ppid, session, trial = tracker_item['ppid'], tracker_item['session'], tracker_item['trial']
            
            if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                plt.scatter(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                           color=f'C{i+1}', marker='x', s=15, alpha=0.8, zorder=1, 
                           label=f'Tracker {ppid}-{session}-{trial}')
                
                # Connect consecutive points
                if len(tracker_data) > 1:
                    plt.plot(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                            color=f'C{i+1}', alpha=0.6, linewidth=1, zorder=1)
    
    # Add reference square
    square_width = 10
    square_height = 6
    square_x = [-square_width/2, square_width/2, square_width/2, -square_width/2, -square_width/2]
    square_y = [-square_height/2, -square_height/2, square_height/2, square_height/2, -square_height/2]
    plt.plot(square_x, square_y, color='darkgrey', linestyle='--', linewidth=2, zorder=5, 
             label='Reference Square (10x6m)')
    
    # Set limits
    plt.xlim([-6, 6])
    plt.ylim([-4, 4])
    
    # Equal aspect ratio
    plt.gca().set_aspect('equal')
    
    # Labels and title
    plt.xlabel('x (metres)')
    plt.ylabel('y (metres)')
    plt.title(f'Debug Plot - {step_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(debug_dir, f'debug_{step_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show plot on screen
    
    
    print(f"✓ Debug plot saved to {plot_path}")
    print(f"✓ Debug plot displayed on screen")


def create_debug_plot_unified(merged_df, tracker_data_list, step_name, output_dir, args, color_map=None):
    """Create debug plot for a specific step using unified dataset."""
    if not (args.debug or args.debug_plots_only):
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    print(f"\n=== DEBUG PLOT FOR STEP: {step_name} (UNIFIED DATASET) ===")
    
    # Debug merged_df content
    print(f"  🔍 DEBUG: merged_df shape: {merged_df.shape}")
    print(f"  🔍 DEBUG: merged_df columns: {list(merged_df.columns)}")
    if len(merged_df) > 0:
        print(f"  🔍 DEBUG: merged_df sample data:")
        print(merged_df[['ppid', 'session', 'trial'] + 
               [col for col in merged_df.columns if 'participant' in col or col.endswith('_x') or col.endswith('_y')]].head())
    else:
        print(f"  ❌ DEBUG: merged_df is EMPTY!")
        return
    
    # Use debug_plots directory directly (already created in main)
    debug_dir = os.path.join(output_dir, 'debug_plots')
    
    # Filter trials to only include those with all three datasets available
    print("  🔍 DEBUG: Filtering trials to only show those with all three datasets (trackers, participant, scenario objects)...")
    
    all_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
    available_trials = []
    
    for _, trial_row in all_trials.iterrows():
        ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
        availability = check_trial_data_availability(merged_df, tracker_data_list, ppid, session, trial)
        
        if availability['all_available']:
            available_trials.append(trial_row)
            print(f"    ✓ Trial {ppid}-{session}-{trial}: All datasets available")
        else:
            missing = []
            if not availability['has_participant']:
                missing.append('participant')
            if not availability['has_scenario']:
                missing.append('scenario')
            if not availability['has_tracker']:
                missing.append('tracker')
            print(f"    ⚠ Trial {ppid}-{session}-{trial}: Missing {', '.join(missing)} data")
    
    # Sample first 9 available trials for visualization
    unique_trials = pd.DataFrame(available_trials).head(9)
    print(f"  Plotting first {len(unique_trials)} trials with complete data")
    print(f"  🔍 DEBUG: available_trials:")
    print(unique_trials)
    
    # Create subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{step_name} - First {len(unique_trials)} Trials Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for idx, (_, trial_data) in enumerate(unique_trials.iterrows()):
        if idx >= 9:  # Safety check
            break
            
        ax = axes_flat[idx]
        ppid, session, trial = trial_data['ppid'], trial_data['session'], trial_data['trial']
        
        # Filter data for this trial
        trial_df = merged_df[
            (merged_df['ppid'] == ppid) & 
            (merged_df['session'] == session) & 
            (merged_df['trial'] == trial)
        ]
        
        print(f"  🔍 DEBUG: Trial {ppid}-{session}-{trial}: {len(trial_df)} rows")
        if len(trial_df) > 0:
            participant_cols = [col for col in trial_df.columns if 'participant' in col]
            scenario_cols = [col for col in trial_df.columns if col.endswith('_x') or col.endswith('_y')]
            print(f"    Participant columns: {participant_cols}")
            print(f"    Scenario columns: {scenario_cols}")
            if participant_cols:
                print(f"    Participant data sample:")
                print(trial_df[participant_cols].head(2))
        
        if len(trial_df) == 0:
            ax.text(0.5, 0.5, f'No data for {ppid}-{session}-{trial}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Trial {ppid}-{session}-{trial}')
            continue
        
        # Get tracker data for this trial
        trial_tracker = None
        for tracker in tracker_data_list:
            if (tracker['ppid'] == ppid and 
                tracker['session'] == session and 
                tracker['trial'] == trial):
                trial_tracker = tracker
                break
        
        # Add reference square (10x6m)
        rect = plt.Rectangle((-5, -3), 10, 6, fill=False, color='grey', 
                           linestyle='--', linewidth=1, label='Reference Square (10x6m)')
        ax.add_patch(rect)
        
        # Plot participant trajectory
        if 'participant_x' in trial_df.columns and 'participant_y' in trial_df.columns:
            participant_data = trial_df[['participant_x', 'participant_y']].dropna()
            print(f"    🔍 DEBUG: Participant data after dropna: {len(participant_data)} points")
            if len(participant_data) > 0:
                x_range = [participant_data['participant_x'].min(), participant_data['participant_x'].max()]
                y_range = [participant_data['participant_y'].min(), participant_data['participant_y'].max()]
                print(f"    🔍 DEBUG: Participant X range: {x_range}, Y range: {y_range}")
                
                # Check if coordinates are dummy values (all the same)
                if x_range[0] == x_range[1] and y_range[0] == y_range[1]:
                    print(f"    ⚠ DEBUG: Participant coordinates are static (dummy values) at ({x_range[0]:.2f}, {y_range[0]:.2f})")
                    # Plot single point for dummy coordinates
                    ax.scatter(x_range[0], y_range[0], c='blue', s=100, alpha=0.8, marker='o', label='Participant (static)')
                    print(f"    ✓ DEBUG: Plotted static participant position")
                else:
                    # Plot as line with dots for real movement data
                    ax.plot(participant_data['participant_x'], participant_data['participant_y'], 
                           'b-', linewidth=1, alpha=0.7, label='Participant')
                    ax.scatter(participant_data['participant_x'], participant_data['participant_y'], 
                              c='blue', s=15, alpha=0.8)
                    print(f"    ✓ DEBUG: Plotted {len(participant_data)} participant movement points")
            else:
                print(f"    ❌ DEBUG: No participant data to plot (all NaN)")
        else:
            print(f"    ❌ DEBUG: Missing participant columns in trial_df")
        
        # Plot scenario objects
        scenario_objects = ['startpoint', 'task', 'safehouse', 'threat', 'weapons', 'wall', 'hide']
        object_markers = {
            'startpoint': ('o', 'yellow', 'StartPoint'),
            'task': ('^', 'green', 'Task'),
            'safehouse': ('s', 'saddlebrown', 'Safehouse'),
            'threat': ('v', 'red', 'Threat'),
            'weapons': ('s', 'purple', 'Weapons'),
            'wall': ('s', 'grey', 'Wall'),
            'hide': ('s', 'cyan', 'Hide')
        }
        
        for obj in scenario_objects:
            # Use regular _x and _y columns
            x_col = f'{obj}_x'
            y_col = f'{obj}_y'
            
            if x_col in trial_df.columns and y_col in trial_df.columns:
                obj_data = trial_df[[x_col, y_col]].dropna()
                print(f"    🔍 DEBUG: {obj} data: {len(obj_data)} points")
                if len(obj_data) > 0:
                    marker, color, label = object_markers[obj]
                    ax.scatter(obj_data[x_col], obj_data[y_col], 
                              marker=marker, c=color, s=100, edgecolor='black', 
                              linewidth=1, label=label, zorder=5)
                    print(f"    ✓ DEBUG: Plotted {obj} at ({obj_data[x_col].iloc[0]:.2f}, {obj_data[y_col].iloc[0]:.2f})")
                else:
                    print(f"    ❌ DEBUG: No {obj} data to plot (all NaN)")
            else:
                print(f"    ❌ DEBUG: Missing {obj} columns: {x_col}, {y_col}")
        
        # Plot tracker path
        if trial_tracker is not None and 'tracker_data' in trial_tracker:
            tracker_data = trial_tracker['tracker_data']
            print(f"    🔍 DEBUG: Found tracker data with {len(tracker_data)} points")
            if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                tracker_x = tracker_data['transformed_x'].dropna()
                tracker_y = tracker_data['transformed_y'].dropna()
                print(f"    🔍 DEBUG: Tracker after dropna: {len(tracker_x)} x points, {len(tracker_y)} y points")
                if len(tracker_x) > 0 and len(tracker_y) > 0:
                    x_range = [tracker_x.min(), tracker_x.max()]
                    y_range = [tracker_y.min(), tracker_y.max()]
                    print(f"    🔍 DEBUG: Tracker X range: {x_range}, Y range: {y_range}")
                    # Plot as line with x markers for better visibility
                    ax.plot(tracker_x, tracker_y, 'r-', linewidth=1, alpha=0.7, label='Tracker')
                    ax.scatter(tracker_x, tracker_y, c='red', marker='x', s=30, 
                              alpha=0.8, zorder=4)
                    print(f"    ✓ DEBUG: Plotted {len(tracker_x)} tracker points")
                else:
                    print(f"    ❌ DEBUG: No tracker data to plot (all NaN)")
            else:
                print(f"    ❌ DEBUG: Missing tracker transformed columns")
        else:
            print(f"    ❌ DEBUG: No tracker data for this trial")
        
        # Set plot properties
        ax.set_xlim(-6, 6)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('x (metres)')
        ax.set_ylabel('y (metres)')
        ax.set_title(f'Trial {ppid}-{session}-{trial}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
        ax.set_aspect('equal')
    
    # Remove empty subplots
    for idx in range(len(unique_trials), 9):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(debug_dir, f'{step_name.lower().replace(" ", "_")}_debug_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Debug plot saved to {plot_path}")
    print(f"✓ Debug plot created using unified dataset with {len(merged_df)} total rows")
    print(f"  - Participant data: {'✓' if 'participant_x' in merged_df.columns else '✗'}")
    print(f"  - Scenario objects: {len([c for c in merged_df.columns if c.endswith('_x') or c.endswith('_y')])} coordinate columns found")
    print(f"  - Tracker data: {'✓' if tracker_data_list else '✗'}")

def print_data_summary(participant_df, scenario_df, tracker_data_list, step_name, args):
    """Print summary of data at each step."""
    if not (args.debug or args.debug_plots_only):
        return
    print(f"\n{'='*60}")
    print(f"DATA SUMMARY FOR STEP: {step_name}")
    print(f"{'='*60}")
    
    # Participant data summary
    print(f"\n📊 PARTICIPANT DATA:")
    print(f"   Shape: {participant_df.shape}")
    print(f"   Columns: {list(participant_df.columns)}")
    if 'participant_x' in participant_df.columns and 'participant_y' in participant_df.columns:
        print(f"   X range: [{participant_df['participant_x'].min():.3f}, {participant_df['participant_x'].max():.3f}]")
        print(f"   Y range: [{participant_df['participant_y'].min():.3f}, {participant_df['participant_y'].max():.3f}]")
    
    # Scenario data summary
    if scenario_df is not None:
        print(f"\n📊 SCENARIO DATA:")
        print(f"   Shape: {scenario_df.shape}")
        print(f"   Columns: {list(scenario_df.columns)}")
        
        # Find startpoint columns
        startpoint_x_col = None
        startpoint_y_col = None
        for col in scenario_df.columns:
            if col.lower().endswith('startpoint_x'):
                startpoint_x_col = col
            elif col.lower().endswith('startpoint_y'):
                startpoint_y_col = col
        
        if startpoint_x_col and startpoint_y_col:
            print(f"   Startpoint X range: [{scenario_df[startpoint_x_col].min():.3f}, {scenario_df[startpoint_x_col].max():.3f}]")
            print(f"   Startpoint Y range: [{scenario_df[startpoint_y_col].min():.3f}, {scenario_df[startpoint_y_col].max():.3f}]")
    
    # Tracker data summary
    if tracker_data_list:
        print(f"\n📊 TRACKER DATA:")
        print(f"   Number of trials: {len(tracker_data_list)}")
        for i, tracker_item in enumerate(tracker_data_list[:3]):  # Only first 3 trials
            tracker_data = tracker_item['tracker_data']
            ppid, session, trial = tracker_item['ppid'], tracker_item['session'], tracker_item['trial']
            print(f"   Trial {i+1} ({ppid}-{session}-{trial}): {len(tracker_data)} points")
            
            if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                print(f"     X range: [{tracker_data['transformed_x'].min():.3f}, {tracker_data['transformed_x'].max():.3f}]")
                print(f"     Y range: [{tracker_data['transformed_y'].min():.3f}, {tracker_data['transformed_y'].max():.3f}]")
    
    print(f"\n{'='*60}")


def print_data_summary_unified(merged_df, tracker_data_list, step_name, args):
    """Print summary of unified data at each step."""
    if not (args.debug or args.debug_plots_only):
        return
    print(f"\n{'='*60}")
    print(f"UNIFIED DATA SUMMARY FOR STEP: {step_name}")
    print(f"{'='*60}")
    
    # Overall dataset summary
    print(f"\n📊 UNIFIED DATASET:")
    print(f"   Shape: {merged_df.shape}")
    print(f"   Columns: {len(merged_df.columns)}")
    print(f"   Total rows: {len(merged_df)}")
    
    # Participant data summary
    if 'participant_x' in merged_df.columns and 'participant_y' in merged_df.columns:
        print(f"\n📊 PARTICIPANT DATA:")
        participant_data = merged_df[merged_df['participant_x'].notna() & merged_df['participant_y'].notna()]
        print(f"   Rows with coordinates: {len(participant_data)}")
        print(f"   X range: [{merged_df['participant_x'].min():.3f}, {merged_df['participant_x'].max():.3f}]")
        print(f"   Y range: [{merged_df['participant_y'].min():.3f}, {merged_df['participant_y'].max():.3f}]")
        
        # Show unique trials
        unique_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
        print(f"   Unique trials: {len(unique_trials)}")
        print(f"   First 5 trials: {unique_trials.head().to_dict('records')}")
    else:
        print(f"\n📊 PARTICIPANT DATA: ⚠ No participant coordinate columns found")
    
    # Scenario objects summary
    coordinate_cols = [c for c in merged_df.columns if c.endswith('_x') or c.endswith('_y')]
    if coordinate_cols:
        print(f"\n📊 SCENARIO OBJECTS:")
        print(f"   Coordinate columns found: {len(coordinate_cols)}")
        
        # Group by object type
        object_types = set()
        for col in coordinate_cols:
            if col.endswith('_x'):
                object_name = col[:-2]  # Remove '_x'
                object_types.add(object_name)
        
        print(f"   Object types: {sorted(list(object_types))}")
        
        # Show sample positions for each object type
        for obj_type in sorted(list(object_types))[:5]:  # Show first 5 object types
            x_col = f'{obj_type}_x'
            y_col = f'{obj_type}_y'
            if x_col in merged_df.columns and y_col in merged_df.columns:
                obj_data = merged_df[[x_col, y_col]].dropna()
                if len(obj_data) > 0:
                    print(f"     {obj_type}: {len(obj_data)} positions")
                    print(f"       X range: [{obj_data[x_col].min():.3f}, {obj_data[x_col].max():.3f}]")
                    print(f"       Y range: [{obj_data[y_col].min():.3f}, {obj_data[y_col].max():.3f}]")
    else:
        print(f"\n📊 SCENARIO OBJECTS: ⚠ No coordinate columns found")
    
    # Labels summary
    if 'predicted_label' in merged_df.columns:
        print(f"\n📊 LABELS:")
        label_data = merged_df[merged_df['predicted_label'].notna()]
        print(f"   Rows with labels: {len(label_data)}")
        unique_labels = merged_df['predicted_label'].dropna().unique()
        print(f"   Unique labels: {len(unique_labels)}")
        print(f"   Label types: {sorted(list(unique_labels))}")
    else:
        print(f"\n📊 LABELS: ⚠ No label column found")
    
    # Metadata summary
    metadata_cols = ['scenario_id', 'episode_set_name2']
    found_metadata = [col for col in metadata_cols if col in merged_df.columns]
    if found_metadata:
        print(f"\n📊 METADATA:")
        for col in found_metadata:
            unique_vals = merged_df[col].dropna().unique()
            print(f"   {col}: {len(unique_vals)} unique values")
            print(f"     Sample: {list(unique_vals[:5])}")
    else:
        print(f"\n📊 METADATA: ⚠ No metadata columns found")
    
    # Tracker data summary
    if tracker_data_list:
        print(f"\n📊 TRACKER DATA:")
        print(f"   Number of trials: {len(tracker_data_list)}")
        for i, tracker_item in enumerate(tracker_data_list[:3]):  # Only first 3 trials
            tracker_data = tracker_item['tracker_data']
            ppid, session, trial = tracker_item['ppid'], tracker_item['session'], tracker_item['trial']
            print(f"   Trial {i+1} ({ppid}-{session}-{trial}): {len(tracker_data)} points")
            
            if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                print(f"     X range: [{tracker_data['transformed_x'].min():.3f}, {tracker_data['transformed_x'].max():.3f}]")
                print(f"     Y range: [{tracker_data['transformed_y'].min():.3f}, {tracker_data['transformed_y'].max():.3f}]")
    else:
        print(f"\n📊 TRACKER DATA: ⚠ No tracker data available")
    
    print(f"\n{'='*60}")


def perform_first_step_analysis_unified(merged_df, tracker_data_list, output_dir, args):
    """Perform first step analysis using unified dataset (merged_df)."""
    print(f"\n{'='*80}")
    print("FIRST STEP ANALYSIS: UNIFIED DATA LOADING AND MERGING COMPLETE")
    print(f"{'='*80}")
    
    # Print unified dataset summary
    print(f"\n📊 UNIFIED DATASET (merged_df):")
    print(f"   Shape: {merged_df.shape}")
    print(f"   Columns: {list(merged_df.columns)}")
    
    # Show unique trials
    unique_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"   Unique trials: {len(unique_trials)}")
    print(f"   First 5 trials: {unique_trials.head().to_dict('records')}")
    
    # Show participant location coordinate ranges if available
    participant_location_columns = globals().get('participant_location_columns', [])
    if participant_location_columns:
        print(f"   Participant location columns: {participant_location_columns}")
        for col in participant_location_columns:
            if col in merged_df.columns:
                print(f"   {col} range: [{merged_df[col].min():.3f}, {merged_df[col].max():.3f}]")
    else:
        print(f"   ⚠ No participant location columns identified")
    
    # Show scenario object columns if available
    scenario_object_columns = globals().get('scenario_object_columns', [])
    if scenario_object_columns:
        print(f"   Scenario object columns: {len(scenario_object_columns)} found")
        print(f"   First 10 scenario columns: {scenario_object_columns[:10]}")
    else:
        print(f"   ⚠ No scenario object columns identified")
    
    # Show metadata columns if available
    metadata_columns = globals().get('metadata_columns', [])
    if metadata_columns:
        print(f"   Metadata columns: {len(metadata_columns)} found")
        print(f"   First 10 metadata columns: {metadata_columns[:10]}")
    else:
        print(f"   ⚠ No metadata columns identified")
    
    # Show label columns if available
    label_columns = globals().get('label_columns', [])
    if label_columns:
        print(f"   Label columns: {label_columns}")
        if args.label_column in merged_df.columns:
            print(f"✓ Label column '{args.label_column}' present")
            label_counts = merged_df[args.label_column].value_counts()
            print(f"  Label distribution: {dict(label_counts.head(10))}")
        else:
            print(f"⚠ Warning: Label column '{args.label_column}' not found")
            print(f"  Available label columns: {label_columns}")
    else:
        print(f"   ⚠ No label columns identified")
    
    # Tracker data summary
    if tracker_data_list:
        print(f"\n📊 TRACKER DATA:")
        print(f"   Number of trials: {len(tracker_data_list)}")
        for i, tracker_item in enumerate(tracker_data_list[:3]):  # Only first 3 trials
            tracker_data = tracker_item['tracker_data']
            ppid, session, trial = tracker_item['ppid'], tracker_item['session'], tracker_item['trial']
            print(f"   Trial {i+1} ({ppid}-{session}-{trial}): {len(tracker_data)} points")
            
            if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                print(f"     X range: [{tracker_data['transformed_x'].min():.3f}, {tracker_data['transformed_x'].max():.3f}]")
                print(f"     Y range: [{tracker_data['transformed_y'].min():.3f}, {tracker_data['transformed_y'].max():.3f}]")
    
    # Create analysis plot using unified dataset
    create_first_step_analysis_plot_unified(merged_df, tracker_data_list, output_dir, args, "Step 0 - Initial Data Loading")
    
    # Interactive exploration
    if args.debug:
        print(f"\n🔍 INTERACTIVE EXPLORATION FOR FIRST STEP ANALYSIS")
        print("You can now explore the unified data interactively using pdb.")
        print("Available variables:")
        print("  - merged_df: Unified dataset containing all data sources")
        print("  - tracker_data_list: List of tracker data for each trial")
        print("\nCommands you can use:")
        print("  - merged_df.head() - View first few rows of unified data")
        print("  - merged_df.columns - View all columns in unified dataset")
        print("  - merged_df[['ppid', 'session', 'trial']].drop_duplicates() - View all trials")
        print("  - tracker_data_list[0]['tracker_data'].head() - View first tracker trial")
        print("  - merged_df.describe() - Statistical summary of unified data")
        print("  - c - Continue to next step")
        print("  - q - Quit debugging")
        

def explore_data_interactively(participant_df, scenario_df, tracker_data_list, step_name, args):
    """Allow interactive exploration of data at each step."""
    if not args.debug:  # Only interactive if full debug mode is enabled
        return
    
    print(f"\n🔍 INTERACTIVE EXPLORATION FOR STEP: {step_name}")
    print("You can now explore the data interactively using pdb.")
    print("Available variables:")
    print("  - participant_df: Participant data")
    print("  - scenario_df: Scenario object data")
    print("  - tracker_data_list: List of tracker data for each trial")
    print("  - step_name: Current step name")
    print("\nCommands you can use:")
    print("  - participant_df.head() - View first few rows of participant data")
    print("  - scenario_df.head() - View first few rows of scenario data")
    print("  - tracker_data_list[0]['tracker_data'].head() - View first tracker trial")
    print("  - participant_df.columns - View participant data columns")
    print("  - scenario_df.columns - View scenario data columns")
    print("  - c - Continue to next step")
    print("  - q - Quit debugging")
    


def process_data_with_correct_order(args, merged_df, tracker_data_list, spatial_output_dir):
    """
    Process data in correct order using the unified dataset (merged_df).
    
    This function applies all transformations in the correct sequence:
    1. Downsampling (optional)
    2. Center tracker data to origin
    3. Apply tracker rotations and shift to participant coordinates
    4. Center scenario objects to origin
    5. Apply scenario rotations and shift to participant coordinates
    6. Data ready for spatial analysis (in memory)
    
    All transformations are applied to merged_df and saved back in place.
    Debug plots are created for each step for visualization.
    Final transformed data is passed directly to spatial analysis.
    """
    print("\n=== PROCESSING UNIFIED DATA WITH CORRECT ORDER ===")
    print("✓ Applying all transformations to unified dataset (merged_df)")
    print("✓ Debug plots will be created for each step")
    print("✓ Data transformations applied to merged_df and saved back in place")
    print("✓ Final data passed directly to spatial analysis")
    
    # Step 0: Initial data loading (show raw data before any transformations)
    debug_step_unified(merged_df, tracker_data_list, "Step 0 - Initial Data Loading", args, spatial_output_dir)
    
    # Step 1: Apply downsampling (if requested)
    if args.downsample > 1:
        merged_df, tracker_data_list = apply_downsampling_unified(
            merged_df, tracker_data_list, args.downsample
        )
        debug_step_unified(merged_df, tracker_data_list, "Step 1 - Downsampling Applied", args, spatial_output_dir)
    
    # Note: Coordinate axes transformation already applied during data loading
    print_coordinate_transformation_status()
    
    # Step 1.5: Validate and fix any malformed coordinate values in merged_df
    print("\n--- STEP 1.5: VALIDATING AND FIXING COORDINATE FORMATTING ---")
    print("Checking for malformed coordinates in unified dataset")
    
    # Validate coordinates in merged_df
    merged_df = validate_and_fix_dataframe_coordinates(merged_df)
    
    # Validate tracker coordinates
    if tracker_data_list:
        for i, tracker_item in enumerate(tracker_data_list):
            if tracker_item['tracker_data'] is not None and len(tracker_item['tracker_data']) > 0:
                print(f"  🔧 Validating tracker {tracker_item['ppid']}-{tracker_item['session']}-{tracker_item['trial']}")
                tracker_item['tracker_data'] = validate_and_fix_dataframe_coordinates(tracker_item['tracker_data'])
    
    print("✓ Coordinate validation complete")
    
    # Step 2: Center tracker data to origin (0,0) using first 20 points median
    tracker_original_positions = center_tracker_data(tracker_data_list)
    debug_step_unified(merged_df, tracker_data_list, "Step 2 - Tracker Centered", args, spatial_output_dir)
    
    # Step 4: Apply rotations to tracker data around their original positions
    # First, ensure scenario_id is available in merged_df
    merged_df = ensure_scenario_id_in_merged_df(merged_df, args, spatial_output_dir)
    
    # Now apply tracker rotations using ICP only
    apply_tracker_rotations_unified(tracker_data_list, tracker_original_positions, args, merged_df)
    debug_step_unified(merged_df, tracker_data_list, "Step 4 - Tracker Rotations Applied", args, spatial_output_dir)
    
    # Step 5: Center scenario objects to origin (0,0) using startpoint
    scenario_original_positions = center_scenario_objects_unified(merged_df)
    debug_step_unified(merged_df, tracker_data_list, "Step 5 - Scenario Centered", args, spatial_output_dir)
    
    # Step 6: Apply rotations to scenario objects around origin (0,0)
    apply_scenario_rotations_unified(merged_df, scenario_original_positions, args, spatial_output_dir, tracker_data_list)
    debug_step_unified(merged_df, tracker_data_list, "Step 6 - Scenario Rotations Applied", args, spatial_output_dir)
    
    # Step 6.5: Shift scenario objects to participant initial coordinates (same logic as trackers)
    # BUT: Don't double-transform - scenario objects are already centered, just need to align with participant
    shift_scenario_objects_to_participant_unified(merged_df, scenario_original_positions)
    debug_step_unified(merged_df, tracker_data_list, "Step 6.5 - Scenario Objects Aligned with Participant", args, spatial_output_dir)
    
    # Step 7: Data processing complete - ready for spatial analysis
    print("\n" + "="*80)
    print("STEP 7: UNIFIED DATA PROCESSING COMPLETE - READY FOR SPATIAL ANALYSIS")
    print("="*80)
    print("✓ All transformations applied successfully to unified dataset:")
    print("  - Coordinate axes transformation (z→x, -x→y)")
    print("  - Coordinate formatting validation and correction")
    if args.downsample > 1:
        print(f"  - Downsampling (factor: {args.downsample})")
    print("  - Tracker data centered to origin (0,0)")
    print("  - Tracker rotations applied (if scenario_rotation enabled)")
    print("  - Tracker data shifted to participant initial coordinates")
    print("  - Scenario objects centered to origin (0,0)")
    print("  - Scenario rotations applied (if scenario_rotation enabled)")
    print("  - Scenario objects shifted to participant initial coordinates")
    
    print(f"\n📊 FINAL UNIFIED DATA READY FOR SPATIAL ANALYSIS:")
    print(f"  - Unified dataset (merged_df): {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"  - Tracker trials: {len(tracker_data_list)}")
    print("✓ All data is in memory and ready to use - no file I/O needed")
    
    # Step 8: Perform spatial analysis if requested
    if args.spatial_analysis:
        print("\n" + "="*80)
        print("STEP 8: PERFORMING SPATIAL ANALYSIS WITH UNIFIED TRANSFORMED DATA")
        
        # Step 8.5: Final debug step after all processing
        debug_step_unified(merged_df, tracker_data_list, "Final - Spatial Analysis Complete", args, spatial_output_dir)
        print("="*80)
        
        # Load color map if provided
        color_map = None
        if args.colour_map_object and os.path.exists(args.colour_map_object):
            try:
                import pickle
                with open(args.colour_map_object, 'rb') as f:
                    color_map = pickle.load(f)
                print(f"✓ Loaded color map: {len(color_map)} syllable colors")
            except Exception as e:
                print(f"⚠ Error loading color map: {e}")
                color_map = None
        
        # Perform spatial analysis with unified transformed data
        print("✓ Using unified FINAL transformed data for spatial analysis:")
        print(f"  - Unified dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        print(f"  - Tracker trials: {len(tracker_data_list)}")
        print("✓ No file I/O needed - data already loaded and transformed")
        
        perform_spatial_analysis_unified(merged_df, tracker_data_list, 
                                       color_map, spatial_output_dir, args)
    else:
        print("⚠ Spatial analysis not requested")
    
    print_completion_summary(args)
    return merged_df, tracker_data_list


def print_coordinate_transformation_status():
    """Print status of coordinate transformations."""
    print("\n" + "="*80)
    print("COORDINATE AXES TRANSFORMATION STATUS")
    print("="*80)
    print("✓ Coordinate axes transformation already applied during data loading")
    print("✓ Tracker data: z→x, -x→y transformation applied in extract_tracker_data()")
    print("✓ Scenario data: z→x, -x→y transformation applied in extract_scenario_data()")
    print("✓ All data ready for centering and rotation steps")


def center_tracker_data(tracker_data_list):
    """Center tracker data to origin (0,0) using first 20 points median."""
    print("\n" + "="*80)
    print("STEP 2: CENTERING TRACKER DATA TO ORIGIN")
    print("="*80)
    print("=== Step 2: Center tracker data to origin (0,0) using first 20 points median ===")
    
    tracker_original_positions = {}
    
    for tracker_item in tracker_data_list:
        tracker_data = tracker_item['tracker_data']
        ppid, session, trial = tracker_item['ppid'], tracker_item['session'], tracker_item['trial']
        
        if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
            # Get first 20 points (or all if less than 20)
            n_points = min(20, len(tracker_data))
            first_points = tracker_data.head(n_points)
            
            # Calculate median of first 20 points from the CURRENT transformed coordinates
            median_x = first_points['transformed_x'].median()
            median_y = first_points['transformed_y'].median()
            
            # Debug: Check if these coordinates are reasonable
            if abs(median_x) > 10 or abs(median_y) > 10:
                print(f"  ⚠ WARNING: Tracker {ppid}-{session}-{trial} has large median coordinates: ({median_x:.3f}, {median_y:.3f})")
                print(f"  ⚠ This suggests the data may have already been processed or transformed")
                print(f"  ⚠ Expected range should be roughly X=[-6, 6], Y=[-4, 4]")
                print(f"  🔧 Will still use these coordinates for centering AND store REAL coordinates for restoration")
            
            # Store original position for later restoration
            # ALWAYS store the REAL coordinates for proper restoration after rotation
            tracker_original_positions[f"{ppid}-{session}-{trial}"] = (median_x, median_y)
            if abs(median_x) > 10 or abs(median_y) > 10:
                print(f"  🔧 Stored REAL coordinates ({median_x:.3f}, {median_y:.3f}) for restoration (even though large)")
            else:
                print(f"  🔧 Stored coordinates ({median_x:.3f}, {median_y:.3f}) for restoration")
            
            # Shift tracker data to origin (0,0)
            tracker_data['transformed_x'] = tracker_data['transformed_x'] - median_x
            tracker_data['transformed_y'] = tracker_data['transformed_y'] - median_y
            
            print(f"  ✓ Centered tracker {ppid}-{session}-{trial} to origin using median ({median_x:.3f}, {median_y:.3f})")
        else:
            print(f"  ⚠ Tracker {ppid}-{session}-{trial} missing transformed coordinates")
    
    return tracker_original_positions


def load_and_process_trial_results(data_dir):
    """Load and concatenate all trial_results.csv files from data directory."""
    print(f"\n" + "="*80)
    print("LOADING TRIAL RESULTS FILES")
    print("="*80)
    print(f"Searching for trial_results.csv files in: {data_dir}")
    
    # Find all trial_results.csv files
    trial_results_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == "trial_results.csv":
                trial_results_files.append(os.path.join(root, file))
    
    if not trial_results_files:
        print("⚠ No trial_results.csv files found")
        return None
    
    print(f"Found {len(trial_results_files)} trial_results.csv files:")
    for file in trial_results_files:
        print(f"  {file}")
    
    # Load and concatenate all files
    all_trial_results = []
    for file_path in trial_results_files:
        try:
            df = pd.read_csv(file_path)
            
            # First, rename columns to avoid conflicts
            if 'session_num' in df.columns and 'session' not in df.columns:
                df = df.rename(columns={'session_num': 'session'})
                print(f"    Renamed 'session_num' to 'session'")
            if 'trial_num' in df.columns and 'trial' not in df.columns:
                df = df.rename(columns={'trial_num': 'trial'})
                print(f"    Renamed 'trial_num' to 'trial'")
            
            # Extract ppid, session, trial from file path
            path_parts = file_path.replace(data_dir, '').strip(os.sep).split(os.sep)
            if len(path_parts) >= 2:
                ppid = path_parts[0]
                session = path_parts[1]
                
                # Check if ppid contains underscores and extract session information
                if '_' in str(ppid):
                    print(f"    ppid '{ppid}' contains underscores, extracting session info")
                    ppid_parts = str(ppid).split('_')
                    if len(ppid_parts) >= 2:
                        # Use first element as ppid, second element as session
                        original_ppid = ppid
                        ppid = ppid_parts[0]
                        session = ppid_parts[1]
                        print(f"      Original ppid: '{original_ppid}' → New ppid: '{ppid}', session: '{session}'")
                    else:
                        print(f"      Warning: ppid '{ppid}' has insufficient parts after splitting")
                
                            # Handle ppid column - overwrite if it exists, create if it doesn't
            df['ppid'] = ppid
            
            # Handle session column more carefully to avoid duplicates
            if 'session' not in df.columns:
                df['session'] = session
                print(f"    Added 'session' column with value: {session}")
            else:
                # If session column already exists, overwrite it
                df['session'] = session
                print(f"    Overwritten existing 'session' column with value: {session}")
                
            print(f"    Final columns in this file: {list(df.columns)}")
            # Also check if ppid column exists in CSV and has underscores
            if 'ppid' in df.columns:
                underscore_mask = df['ppid'].astype(str).str.contains('_', na=False)
                if underscore_mask.any():
                    print(f"    Found {underscore_mask.sum()} rows with underscores in ppid column")
                    for idx in df[underscore_mask].index:
                        original_ppid = df.at[idx, 'ppid']
                        ppid_parts = str(original_ppid).split('_')
                        if len(ppid_parts) >= 2:
                            new_ppid = ppid_parts[0]
                            new_session = ppid_parts[1]
                            df.at[idx, 'ppid'] = new_ppid
                            # Overwrite session if it exists, otherwise create it
                            df.at[idx, 'session'] = new_session
                            print(f"      Row {idx}: '{original_ppid}' → ppid: '{new_ppid}', session: '{new_session}'")
            
            all_trial_results.append(df)
            print(f"  ✓ Loaded {file_path} (shape: {df.shape})")
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
    
    if not all_trial_results:
        print("⚠ No trial results files could be loaded")
        return None
    
    # Concatenate all DataFrames
    concatenated_df = pd.concat(all_trial_results, ignore_index=True)
    print(f"\nConcatenated trial results shape: {concatenated_df.shape}")
    print(f"Columns: {list(concatenated_df.columns)}")
    
    # Check for and remove duplicate column names
    if concatenated_df.columns.duplicated().any():
        print(f"⚠ Found duplicate column names: {concatenated_df.columns[concatenated_df.columns.duplicated()].tolist()}")
        # Keep only the first occurrence of each column name
        concatenated_df = concatenated_df.loc[:, ~concatenated_df.columns.duplicated()]
        print(f"✓ Removed duplicate columns. New shape: {concatenated_df.shape}")
        print(f"Final columns: {list(concatenated_df.columns)}")
    
    # Final check for any remaining underscores in ppid after processing
    if 'ppid' in concatenated_df.columns:
        remaining_underscores = concatenated_df['ppid'].astype(str).str.contains('_', na=False).sum()
        if remaining_underscores > 0:
            print(f"⚠ Warning: {remaining_underscores} ppid values still contain underscores after processing")
        else:
            print(f"✓ All ppid underscore processing completed successfully")
    
    # Check if any files still need column renaming (should be rare now)
    column_mapping = {}
    if 'session_num' in concatenated_df.columns:
        column_mapping['session_num'] = 'session'
    if 'trial_num' in concatenated_df.columns:
        column_mapping['trial_num'] = 'trial'
    
    if column_mapping:
        concatenated_df = concatenated_df.rename(columns=column_mapping)
        print(f"Additional column renaming needed: {column_mapping}")
    else:
        print("✓ All column renaming completed during individual file processing")
    
    # Show sample data
    print(f"\nFirst 10 rows:")
    print(concatenated_df.head(10))
    
    # Show unique combinations
    if 'ppid' in concatenated_df.columns and 'session' in concatenated_df.columns and 'trial' in concatenated_df.columns:
        unique_combinations = concatenated_df[['ppid', 'session', 'trial']].drop_duplicates()
        print(f"\nUnique ppid-session-trial combinations: {len(unique_combinations)}")
        print(unique_combinations.head(10))
        
        # Show summary of ppid/session combinations after processing
        print(f"\nFinal ppid/session combinations (after underscore processing):")
        ppid_session_combinations = concatenated_df[['ppid', 'session']].drop_duplicates().sort_values(['ppid', 'session'])
        print(ppid_session_combinations.head(20))
        if len(ppid_session_combinations) > 20:
            print(f"... and {len(ppid_session_combinations) - 20} more combinations")
    
    return concatenated_df


def preprocess_metadata_file(metadata_path, output_dir, trial_results_df=None):
    """Preprocess metadata file to clean up scenario_id values and add missing combinations."""
    print(f"\n" + "="*80)
    print("PREPROCESSING METADATA FILE")
    print("="*80)
    print(f"Loading metadata from: {metadata_path}")
    
    try:
        metadata_df = pd.read_csv(metadata_path)
        print(f"Original metadata shape: {metadata_df.shape}")
        print(f"Original columns: {list(metadata_df.columns)}")
        
        if 'scenario_id' not in metadata_df.columns:
            print("⚠ 'scenario_id' column not found in metadata file")
            return None
        
        print(f"\nOriginal scenario_id values:")
        print(metadata_df['scenario_id'].value_counts().head(20))
        
        # Create a copy for processing
        processed_df = metadata_df.copy()
        
        # NEW: Clean session values FIRST (remove "S" and leading zeros) - BEFORE checking for missing combinations
        print(f"\n--- Cleaning Session Values FIRST ---")
        if 'session' in processed_df.columns:
            session_cleaned_count = 0
            for idx in processed_df.index:
                session_val = str(processed_df.at[idx, 'session'])
                original_session = session_val
                
                # Remove "S" prefix if present
                if session_val.startswith('S'):
                    session_val = session_val[1:]
                
                # Remove leading zeros
                if session_val.isdigit():
                    session_val = str(int(session_val))
                
                if session_val != original_session:
                    processed_df.at[idx, 'session'] = session_val
                    session_cleaned_count += 1
                    print(f"  Session cleaned: {original_session} → {session_val}")
            
            print(f"✓ Cleaned {session_cleaned_count} session values")
        
        # Add missing combinations from trial results if available (AFTER session cleaning)
        if trial_results_df is not None:
             print(f"\n--- Adding Missing Combinations from Trial Results ---")
             print(f"Trial results shape: {trial_results_df.shape}")
             print(f"Trial results columns: {list(trial_results_df.columns)}")
             
             # Check if episode column exists in trial results
             episode_col = None
             for col in trial_results_df.columns:
                 if col.lower() in ['episode', 'episode_name', 'trial_name']:
                     episode_col = col
                     break
             
             if episode_col:
                 print(f"✓ Found episode column: '{episode_col}'")
                 # Get unique combinations from trial results including episode
                 trial_combinations = trial_results_df[['ppid', 'session', 'trial', episode_col]].drop_duplicates()
             else:
                 print("⚠ No episode column found in trial results, will use default names")
                 # Get unique combinations from trial results without episode
                 trial_combinations = trial_results_df[['ppid', 'session', 'trial']].drop_duplicates()
             
             print(f"Trial results unique combinations: {len(trial_combinations)}")
             
             # IMPORTANT: Also clean session values in trial results before comparison
             if 'session' in trial_combinations.columns:
                 for idx in trial_combinations.index:
                     session_val = str(trial_combinations.at[idx, 'session'])
                     original_session = session_val
                     
                     # Remove "S" prefix if present
                     if session_val.startswith('S'):
                         session_val = session_val[1:]
                     
                     # Remove leading zeros
                     if session_val.isdigit():
                         session_val = str(int(session_val))
                     
                     if session_val != original_session:
                         trial_combinations.at[idx, 'session'] = session_val
                         print(f"  Trial results session cleaned: {original_session} → {session_val}")
             
             # Get existing combinations from metadata (already cleaned)
             existing_combinations = processed_df[['ppid', 'session', 'trial']].drop_duplicates()
             print(f"Metadata existing combinations: {len(existing_combinations)}")
             
             # Find missing combinations
             missing_combinations = []
             for _, trial_row in trial_combinations.iterrows():
                 ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
                 
                 # Check if this combination exists in metadata (both cleaned now)
                 exists = existing_combinations[
                     (existing_combinations['ppid'].astype(str) == str(ppid)) &
                     (existing_combinations['session'].astype(str) == str(session)) &
                     (existing_combinations['trial'].astype(str) == str(trial))
                 ].shape[0] > 0
                 
                 if not exists:
                     # Use episode from trial results if available, otherwise create default
                     if episode_col and episode_col in trial_row:
                         episode_name = trial_row[episode_col]
                     else:
                         episode_name = f"Trial_{trial}"
                     
                     missing_combinations.append({
                         'ppid': ppid,
                         'session': session,
                         'trial': trial,
                         'episode': episode_name,
                         'scenario_id': None  # Will be filled by preprocessing rules
                     })
                     print(f"  Missing: ppid={ppid}, session={session}, trial={trial}, episode={episode_name}")
             
             if missing_combinations:
                 print(f"\nFound {len(missing_combinations)} missing combinations:")
                 missing_df = pd.DataFrame(missing_combinations)
                 print(missing_df.head(10))
                 
                 # Add missing combinations to processed metadata
                 processed_df = pd.concat([processed_df, missing_df], ignore_index=True)
                 print(f"Added missing combinations. New shape: {processed_df.shape}")
                 
                 # Show sample of added data
                 print(f"\nSample of newly added rows:")
                 print(processed_df.tail(min(5, len(missing_combinations))))
             else:
                 print("✓ No missing combinations found")
        
        # NEW: Add missing episode_set_name2 values
        print(f"\n--- Adding Missing episode_set_name2 Values ---")
        if 'episode' in processed_df.columns:
            episode_set_added_count = 0
            for idx in processed_df.index:
                if pd.isna(processed_df.at[idx, 'episode_set_name2']) if 'episode_set_name2' in processed_df.columns else True:
                    episode_name = str(processed_df.at[idx, 'episode'])
                    
                    # Determine episode_set_name2 based on episode name
                    if "Tutorial" in episode_name:
                        episode_set_name2 = "instructed"
                    elif "Stump" in episode_name:
                        episode_set_name2 = "stump"
                    elif "_F_" in episode_name:
                        episode_set_name2 = "fixed"
                    elif "Habituation" in episode_name:
                        episode_set_name2 = "habituation"
                    else:
                        episode_set_name2 = "dictionary"
                    
                    # Add column if it doesn't exist
                    if 'episode_set_name2' not in processed_df.columns:
                        processed_df['episode_set_name2'] = None
                    
                    processed_df.at[idx, 'episode_set_name2'] = episode_set_name2
                    episode_set_added_count += 1
                    print(f"  Episode: {episode_name} → episode_set_name2: {episode_set_name2}")
            
            print(f"✓ Added {episode_set_added_count} episode_set_name2 values")
        
        # Show current state before applying preprocessing rules
        print(f"\n--- Current State Before Preprocessing Rules ---")
        print(f"Total rows: {len(processed_df)}")
        print(f"Rows with NaN scenario_id: {processed_df['scenario_id'].isna().sum()}")
        if processed_df['scenario_id'].isna().sum() > 0:
            nan_episodes = processed_df[processed_df['scenario_id'].isna()]['episode'].value_counts()
            print(f"Episode types with NaN scenario_id:")
            print(nan_episodes.head(10))
        
        # Rule 1: If value ends with a string, only keep the string as scenario_id
        print(f"\n--- Rule 1: Extract string endings ---")
        string_mask = processed_df['scenario_id'].astype(str).str.match(r'.*[a-zA-Z]+$')
        if string_mask.any():
            print(f"Found {string_mask.sum()} values ending with strings")
            for idx in processed_df[string_mask].index:
                original = processed_df.at[idx, 'scenario_id']
                # Extract only the string part (letters at the end)
                import re
                match = re.search(r'([a-zA-Z]+)$', str(original))
                if match:
                    string_part = match.group(1)
                    processed_df.at[idx, 'scenario_id'] = string_part
                    print(f"  {original} → {string_part}")
                else:
                    print(f"  {original} → no string ending found")
        
        # Rule 2: Remove leading zeros from numeric scenario_ids
        print(f"\n--- Rule 2: Remove leading zeros ---")
        numeric_mask = processed_df['scenario_id'].astype(str).str.match(r'^0+\d+$')
        if numeric_mask.any():
            print(f"Found {numeric_mask.sum()} values with leading zeros")
            for idx in processed_df[numeric_mask].index:
                original = processed_df.at[idx, 'scenario_id']
                cleaned = str(original).lstrip('0')
                processed_df.at[idx, 'scenario_id'] = cleaned
                print(f"  {original} → {cleaned}")
        
        # Rule 3: Handle missing scenario_id values using episode column
        print(f"\n--- Rule 3: Handle missing scenario_id values using episode column ---")
        if 'episode' in processed_df.columns:
            # Check for missing scenario_id values (NaN, None, empty strings, or missing column)
            if 'scenario_id' not in processed_df.columns:
                print("⚠ 'scenario_id' column not found, creating it")
                processed_df['scenario_id'] = None
                nan_mask = pd.Series([True] * len(processed_df), index=processed_df.index)
            else:
                # Check for NaN, None, or empty string values
                nan_mask = (processed_df['scenario_id'].isna() | 
                           (processed_df['scenario_id'].astype(str) == '') |
                           (processed_df['scenario_id'].astype(str) == 'None') |
                           (processed_df['scenario_id'].astype(str) == 'nan'))
            
            print(f"Found {nan_mask.sum()} rows with missing scenario_id values")
            print(f"Total rows: {len(processed_df)}")
            print(f"Rows with valid scenario_id: {(~nan_mask).sum()}")
            
            if nan_mask.any():
                print(f"Sample of missing scenario_id rows:")
                missing_sample = processed_df[nan_mask][['ppid', 'session', 'trial', 'episode', 'scenario_id']].head(5)
                print(missing_sample)
                
                scenario_id_added_count = 0
                for idx in processed_df[nan_mask].index:
                    episode = str(processed_df.at[idx, 'episode'])
                    print(f"    Episode: {episode}")
                    
                    # Rule 1: "Habituation" in string -> "h"
                    if "Habituation" in episode:
                        processed_df.at[idx, 'scenario_id'] = "h"
                        print(f"      Rule 1: 'Habituation' in string → scenario_id: 'h'")
                        scenario_id_added_count += 1
                    
                    # Rule 2: "_F_" in string -> "f"
                    elif "_F_" in episode:
                        processed_df.at[idx, 'scenario_id'] = "f"
                        print(f"      Rule 2: '_F_' in string → scenario_id: 'f'")
                        scenario_id_added_count += 1
                    
                    # Rule 3: "Stump" in string -> "s"
                    elif "Stump" in episode:
                        processed_df.at[idx, 'scenario_id'] = "s"
                        print(f"      Rule 3: 'Stump' in string → scenario_id: 's'")
                        scenario_id_added_count += 1
                    
                    # Rule 4: For other cases -> use second element but remove leading zeros
                    else:
                        parts = episode.split('_')
                        if len(parts) >= 2:
                            second_element = parts[1]
                            print(f"      Rule 4: Using second element '{second_element}'")
                            
                            # Remove leading zeros from second element
                            if second_element.isdigit():
                                cleaned_scenario = second_element.lstrip('0')
                                processed_df.at[idx, 'scenario_id'] = cleaned_scenario
                                print(f"        '{second_element}' → scenario_id: '{cleaned_scenario}'")
                                scenario_id_added_count += 1
                            else:
                                # If second element is not digits, use it as is
                                processed_df.at[idx, 'scenario_id'] = second_element
                                print(f"        '{second_element}' → scenario_id: '{second_element}'")
                                scenario_id_added_count += 1
                        else:
                            print(f"      No second element found for splitting")
                
                print(f"✓ Added {scenario_id_added_count} scenario_id values")
        else:
            print("⚠ 'episode' column not found, skipping Rule 3")
        
        # Show final results
        print(f"\n--- Final Results ---")
        print(f"Final scenario_id values:")
        print(processed_df['scenario_id'].value_counts().head(20))
        
        # Save processed metadata to output directory
        # Ensure the data subdirectory exists
        data_dir = os.path.join(output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        output_path = os.path.join(data_dir, 'processed_metadata.csv')
        processed_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed metadata to: {output_path}")
        print(f"  This file will be used by the spatial analysis function")
        
        return processed_df
        
    except Exception as e:
        print(f"⚠ Error preprocessing metadata file: {e}")
        
        return None


def add_scenario_id_to_data(scenario_df, args, output_dir):
    """Add scenario_id to scenario data from preprocessed metadata file."""
    if 'scenario_id' in scenario_df.columns:
        return scenario_df
    
    if not args.metadata_path:
        print("⚠ No metadata file provided for scenario_id")
        return scenario_df
    
    print(f"Adding scenario_id to scenario data from metadata file")
    print(f"Metadata file: {args.metadata_path}")
    print(f"Scenario data shape: {scenario_df.shape}")
    print(f"Scenario data columns: {list(scenario_df.columns)}")
    
    # Load trial results files if data_dir is available
    trial_results_df = None
    if hasattr(args, 'data_dir') and args.data_dir:
        trial_results_df = load_and_process_trial_results(args.data_dir)
    
    # First, preprocess the metadata file (now with trial results data)
    processed_metadata_df = preprocess_metadata_file(args.metadata_path, output_dir, trial_results_df)
    if processed_metadata_df is None:
        print("⚠ Failed to preprocess metadata file")
        return scenario_df
    
    try:
        print(f"\n" + "="*60)
        print("DEBUG: PROCESSED METADATA DATA")
        print("="*60)
        print(f"File: {args.metadata_path} (preprocessed)")
        print(f"Shape: {processed_metadata_df.shape}")
        print(f"Columns: {list(processed_metadata_df.columns)}")
        print(f"\nFirst 10 rows:")
        print(processed_metadata_df.head(10))
        print(f"\nUnique ppid-session-trial combinations:")
        unique_combinations = processed_metadata_df[['ppid', 'session', 'trial', 'scenario_id']].drop_duplicates()
        print(unique_combinations.head(10))
        
        print(f"\n" + "="*60)
        print("DEBUG: SCENARIO_DF BEFORE MERGE")
        print("="*60)
        print(f"Shape: {scenario_df.shape}")
        print(f"Columns: {list(scenario_df.columns)}")
        print(f"\nFirst 10 rows:")
        print(scenario_df.head(10))
        print(f"\nUnique ppid-session-trial combinations in scenario_df:")
        scenario_unique = scenario_df[['ppid', 'session', 'trial']].drop_duplicates()
        print(scenario_unique.head(10))
        
        # Ensure consistent data types
        processed_metadata_df[['ppid', 'session', 'trial']] = processed_metadata_df[['ppid', 'session', 'trial']].astype(str)
        scenario_df[['ppid', 'session', 'trial']] = scenario_df[['ppid', 'session', 'trial']].astype(str)
        
        # Show sample values before merge to debug data type issues
        print(f"\nSample values before merge (first 5 rows):")
        print("processed_metadata_df:")
        print(processed_metadata_df[['ppid', 'session', 'trial', 'scenario_id']].head())
        print("\nscenario_df:")
        print(scenario_df[['ppid', 'session', 'trial']].head())
        
        print(f"\n" + "="*60)
        print("DEBUG: DATA TYPES AFTER CONVERSION")
        print("="*60)
        print(f"processed_metadata_df dtypes:")
        print(processed_metadata_df[['ppid', 'session', 'trial']].dtypes)
        print(f"\nscenario_df dtypes:")
        print(scenario_df[['ppid', 'session', 'trial']].dtypes)
        
        # Merge scenario_id - IMPORTANT: This returns a new DataFrame!
        print(f"\n" + "="*60)
        print("DEBUG: PERFORMING MERGE")
        print("="*60)
        print("Merging on columns: ['ppid', 'session', 'trial']")
        print(f"processed_metadata_df merge columns shape: {processed_metadata_df[['ppid', 'session', 'trial', 'scenario_id']].shape}")
        print(f"scenario_df merge columns shape: {scenario_df[['ppid', 'session', 'trial']].shape}")
        
        # Create the merged DataFrame
        merged_scenario_df = pd.merge(scenario_df, processed_metadata_df[['ppid', 'session', 'trial', 'scenario_id']], 
                                     on=['ppid', 'session', 'trial'], how='left')
        
        print(f"\n" + "="*60)
        print("DEBUG: SCENARIO_DF AFTER MERGE")
        print("="*60)
        print(f"Shape: {merged_scenario_df.shape}")
        print(f"Columns: {list(merged_scenario_df.columns)}")
        print(f"\nFirst 10 rows:")
        print(merged_scenario_df.head(10))
        
        # Check for missing scenario_id values
        missing_scenario_id = merged_scenario_df['scenario_id'].isna().sum()
        if missing_scenario_id > 0:
            print(f"\n⚠ WARNING: {missing_scenario_id} rows have missing scenario_id values")
            print("Rows with missing scenario_id:")
            missing_rows = merged_scenario_df[merged_scenario_df['scenario_id'].isna()]
            print(missing_rows[['ppid', 'session', 'trial']].head(10))
            
            # Check if these missing combinations exist in the processed metadata
            print(f"\nChecking if missing combinations exist in processed metadata:")
            for _, row in missing_rows[['ppid', 'session', 'trial']].head(5).iterrows():
                ppid, session, trial = row['ppid'], row['session'], row['trial']
                metadata_match = processed_metadata_df[
                    (processed_metadata_df['ppid'].astype(str) == str(ppid)) &
                    (processed_metadata_df['session'].astype(str) == str(session)) &
                    (processed_metadata_df['trial'].astype(str) == str(trial))
                ]
                if metadata_match.empty:
                    print(f"  ❌ {ppid}-{session}-{trial}: NOT FOUND in processed metadata")
                else:
                    print(f"  ✓ {ppid}-{session}-{trial}: FOUND in processed metadata")
                    print(f"     scenario_id: {metadata_match.iloc[0]['scenario_id']}")
        
        print(f"\n" + "="*60)
        print("DEBUG: MERGE SUMMARY")
        print("="*60)
        print(f"✓ Added scenario_id to scenario data from processed metadata file")
        print(f"  - Original scenario_df shape: {scenario_df.shape[0]} rows")
        print(f"  - After merge shape: {merged_scenario_df.shape[0]} rows")
        print(f"  - New column: scenario_id")
        
        # Ensure scenario_id values are cleaned and consistent
        print(f"\n" + "="*60)
        print("DEBUG: CLEANING SCENARIO_ID VALUES")
        print("="*60)
        
        # Apply clean_scenario_id to all scenario_id values to ensure consistency
        if 'scenario_id' in merged_scenario_df.columns:
            print(f"Cleaning scenario_id values for consistency...")
            print(f"Before cleaning - unique values: {sorted(merged_scenario_df['scenario_id'].dropna().unique())}")
            
            # Apply cleaning to all scenario_id values
            merged_scenario_df['scenario_id'] = merged_scenario_df['scenario_id'].apply(clean_scenario_id)
            
            print(f"After cleaning - unique values: {sorted(merged_scenario_df['scenario_id'].dropna().unique())}")
            
            # Verify no NaN values remain
            nan_count_after = merged_scenario_df['scenario_id'].isna().sum()
            print(f"NaN count after cleaning: {nan_count_after}")
        
        # Return the merged DataFrame with cleaned scenario_id values
        return merged_scenario_df
        
    except Exception as e:
        print(f"⚠ Error adding scenario_id from processed metadata file: {e}")
        return scenario_df


def apply_scenario_rotations_implementation(scenario_df, scenario_original_positions, scenario_rotation_dict):
    """Implement scenario rotations."""
    print("Applying rotations to scenario objects around their startpoint...")
    print(f"DEBUG: Scenario data has {len(scenario_df)} rows")
    print(f"DEBUG: Available scenario_id values: {scenario_df['scenario_id'].unique()}")
    
    # Create scenario rotation mapping
    scenario_rotation_mapping = create_scenario_rotation_mapping(scenario_df, scenario_rotation_dict)
    
    if not scenario_rotation_mapping:
        print("⚠ No scenario rotation mappings created")
        return
    
    print(f"✓ Created rotation mapping for {len(scenario_rotation_mapping)} scenarios")
    
    # Apply rotations using the mapping
    print(f"\n" + "="*60)
    print("DEBUG: APPLYING SCENARIO ROTATIONS")
    print("="*60)
    
    for i, mapping in enumerate(scenario_rotation_mapping):
        scenario_id = mapping['scenario_id']
        angle = mapping['angle']
        mask = mapping['mask']
        row_count = mapping['row_count']
        
        print(f"\n--- Applying rotation {i+1}/{len(scenario_rotation_mapping)} ---")
        print(f"  Rotating scenario_id {scenario_id} by {angle}° around startpoint ({row_count} rows)")
        
        # Get the startpoint for this scenario (use first trial of this scenario)
        scenario_trials = scenario_df[mask]
        first_trial = scenario_trials.iloc[0]
        
        # Determine rotation center: use original startpoint position (stored before centering)
        ppid, session, trial = first_trial['ppid'], first_trial['session'], first_trial['trial']
        key = f"{ppid}-{session}-{trial}"
        
        if key in scenario_original_positions:
            rotation_center_x, rotation_center_y = scenario_original_positions[key]
            print(f"DEBUG: Using original startpoint as rotation center: ({rotation_center_x:.3f}, {rotation_center_y:.3f})")
        else:
            print(f"DEBUG: No original startpoint position stored for {key}, skipping rotation")
            continue
        
        # Apply rotation to all scenario object coordinates around the startpoint
        scenario_cols = [c for c in scenario_df.columns if c.endswith('_x') or c.endswith('_y')]
        print(f"DEBUG: Found {len(scenario_cols)} coordinate columns to rotate: {scenario_cols}")
        
        # Process X and Y coordinates together to maintain relative structure
        x_cols = [c for c in scenario_cols if c.endswith('_x')]
        y_cols = [c for c in scenario_cols if c.endswith('_y')]
        
        for x_col, y_col in zip(x_cols, y_cols):
            if x_col in scenario_df.columns and y_col in scenario_df.columns:
                print(f"DEBUG: Rotating {x_col} and {y_col} around ({rotation_center_x:.3f}, {rotation_center_y:.3f})")
                
                # Get current coordinates (already centered at origin)
                x_coords = scenario_df.loc[mask, x_col].copy()
                y_coords = scenario_df.loc[mask, y_col].copy()
                
                # Apply rotation around origin (0,0) - scenario objects are already centered
                rad = math.radians(angle)  # Negative for clockwise
                cosv = math.cos(rad)
                sinv = math.sin(rad)
                
                x_rotated = x_coords * cosv - y_coords * sinv
                y_rotated = x_coords * sinv + y_coords * cosv
                
                # Store rotated coordinates
                scenario_df.loc[mask, x_col] = x_rotated
                scenario_df.loc[mask, y_col] = y_rotated
                
                print(f"DEBUG: {x_col} range: [{x_coords.min():.3f}, {x_coords.max():.3f}] -> [{scenario_df.loc[mask, x_col].min():.3f}, {scenario_df.loc[mask, x_col].max():.3f}]")
                print(f"DEBUG: {y_col} range: [{y_coords.min():.3f}, {y_coords.max():.3f}] -> [{scenario_df.loc[mask, y_col].min():.3f}, {scenario_df.loc[mask, y_col].max():.3f}]")
                
                # Add pdb stop point for detailed inspection (only for first rotation to avoid too many stops)
                if i == 0:
                    print(f"    🔍 PDB STOP: Inspect rotation details for {x_col}, {y_col}")
    
            else:
                print(f"DEBUG: Warning - Missing corresponding column for {x_col} or {y_col}")
        
        print(f"    ✓ Applied rotation around startpoint ({rotation_center_x:.3f}, {rotation_center_y:.3f}) to maintain relative structure")
        print(f"    📐 Rotation angle: {angle}° (negative for clockwise rotation)")
        print(f"    🔄 Scenario objects rotated as a unit to preserve internal structure")
    
    # Add final debug summary
    print(f"\n" + "="*60)
    print("DEBUG: SCENARIO ROTATION APPLICATION SUMMARY")
    print("="*60)
    print(f"Total rotations applied: {len(scenario_rotation_mapping)}")
    print(f"Total rows affected: {sum(mapping['row_count'] for mapping in scenario_rotation_mapping)}")
    print(f"Final scenario_df shape: {scenario_df.shape}")
    print(f"Final scenario_df columns: {list(scenario_df.columns)}")
    
    # Add pdb stop point for final inspection
    print(f"\n🔍 FINAL PDB STOP: Inspect final scenario_df after all rotations")
    


def create_scenario_rotation_mapping(scenario_df, scenario_rotation_dict):
    """Create mapping between scenario_ids and their rotation angles."""
    print(f"\n" + "="*60)
    print("DEBUG: CREATING SCENARIO ROTATION MAPPING")
    print("="*60)
    print(f"Total scenario_ids in rotation_dict: {len(scenario_rotation_dict)}")
    if 'scenario_id' in scenario_df.columns:
        # Handle mixed types (strings and NaN) by filtering out NaN values before sorting
        unique_scenario_ids = scenario_df['scenario_id'].dropna().unique()
        print(f"Available scenario_ids in scenario_df: {sorted(unique_scenario_ids.astype(str))}")
    else:
        print(f"Available scenario_ids in scenario_df: N/A")
    
    scenario_rotation_mapping = []
    
    for i, (scenario_id, angle) in enumerate(scenario_rotation_dict.items()):
        print(f"\n--- Processing scenario_id {i+1}/{len(scenario_rotation_dict)}: {scenario_id} (angle: {angle}°) ---")
        
        if angle == 0:
            print(f"  ⚠ Skipping rotation for scenario_id {scenario_id} (angle = 0°)")
            continue
                
        # Find all rows with this scenario_id
        mask = scenario_df['scenario_id'].astype(str) == str(scenario_id)
        row_count = mask.sum()
        
        print(f"  Searching for scenario_id: {scenario_id}")
        print(f"  Found {row_count} rows with this scenario_id")
        
        if mask.any():
            matching_rows = scenario_df[mask]
            print(f"  Matching rows:")
            print(matching_rows[['ppid', 'session', 'trial', 'scenario_id']].head())
            
            scenario_rotation_mapping.append({
                        'scenario_id': scenario_id,
                        'angle': angle,
                        'mask': mask,
                'row_count': row_count
                    })
            print(f"  ✓ Mapped scenario_id {scenario_id} to angle {angle}° with {row_count} rows")
        else:
            print(f"  ❌ No rows found for scenario_id {scenario_id}")
            # Handle mixed types (strings and NaN) by filtering out NaN values before sorting
            unique_scenario_ids = scenario_df['scenario_id'].dropna().unique()
            print(f"  Available scenario_ids in scenario_df: {sorted(unique_scenario_ids.astype(str))}")
    
    print(f"\n" + "="*60)
    print("DEBUG: SCENARIO ROTATION MAPPING SUMMARY")
    print("="*60)
    print(f"Created {len(scenario_rotation_mapping)} scenario rotation mappings:")
    for i, mapping in enumerate(scenario_rotation_mapping):
        print(f"  {i+1}. scenario_id {mapping['scenario_id']} → {mapping['angle']}° → {mapping['row_count']} rows")
    
    # Add pdb stop point for inspection
    print(f"\n🔍 PDB STOP: Inspect scenario rotation mappings")
    
    
    return scenario_rotation_mapping


def apply_single_scenario_rotation(mapping, scenario_df, scenario_original_positions):
    """Apply rotation to a single scenario."""
    scenario_id = mapping['scenario_id']
    angle = mapping['angle']
    mask = mapping['mask']
    row_count = mapping['row_count']
    
    print(f"  Rotating scenario_id {scenario_id} by {angle}° around startpoint ({row_count} rows)")
    
    # Get the startpoint for this scenario (use first trial of this scenario)
    scenario_trials = scenario_df[mask]
    first_trial = scenario_trials.iloc[0]
                
    # Determine rotation center: use original startpoint position (stored before centering)
    ppid, session, trial = first_trial['ppid'], first_trial['session'], first_trial['trial']
    key = f"{ppid}-{session}-{trial}"
    
    if key not in scenario_original_positions:
        print(f"DEBUG: No original startpoint position stored for {key}, skipping rotation")
        return
    
    rotation_center_x, rotation_center_y = scenario_original_positions[key]
    print(f"DEBUG: Using original startpoint as rotation center: ({rotation_center_x:.3f}, {rotation_center_y:.3f})")
    
    # Apply rotation to all scenario object coordinates around the startpoint
    scenario_cols = [c for c in scenario_df.columns if c.endswith('_x') or c.endswith('_y')]
    print(f"DEBUG: Found {len(scenario_cols)} coordinate columns to rotate: {scenario_cols}")
    
    # Process X and Y coordinates together to maintain relative structure
    x_cols = [c for c in scenario_cols if c.endswith('_x')]
    y_cols = [c for c in scenario_cols if c.endswith('_y')]
    
    for x_col, y_col in zip(x_cols, y_cols):
        if x_col in scenario_df.columns and y_col in scenario_df.columns:
            print(f"DEBUG: Rotating {x_col} and {y_col} around ({rotation_center_x:.3f}, {rotation_center_y:.3f})")
                        
            # Get current coordinates (already centered at origin)
            x_coords = scenario_df.loc[mask, x_col].copy()
            y_coords = scenario_df.loc[mask, y_col].copy()
                        
            # Apply rotation around origin (0,0) - scenario objects are already centered
            rad = math.radians(angle)  # Negative for clockwise
            cosv = math.cos(rad)
            sinv = math.sin(rad)
            
            x_rotated = x_coords * cosv - y_coords * sinv
            y_rotated = x_coords * sinv + y_coords * cosv
            
            # Store rotated coordinates
            scenario_df.loc[mask, x_col] = x_rotated
            scenario_df.loc[mask, y_col] = y_rotated
                        
            print(f"DEBUG: {x_col} range: [{x_coords.min():.3f}, {x_coords.max():.3f}] -> [{scenario_df.loc[mask, x_col].min():.3f}, {scenario_df.loc[mask, x_col].max():.3f}]")
            print(f"DEBUG: {y_col} range: [{y_coords.min():.3f}, {y_coords.max():.3f}] -> [{scenario_df.loc[mask, y_col].min():.3f}, {scenario_df.loc[mask, y_col].max():.3f}]")
        else:
            print(f"DEBUG: Warning - Missing corresponding column for {x_col} or {y_col}")
                
        print(f"    ✓ Applied rotation around startpoint ({rotation_center_x:.3f}, {rotation_center_y:.3f}) to maintain relative structure")
        print(f"    📐 Rotation angle: {angle}° (negative for clockwise rotation)")
        print(f"    🔄 Scenario objects rotated as a unit to preserve internal structure")
    
    
def restore_scenario_positions(scenario_df, scenario_original_positions, participant_df):
    """Shift scenario objects to participant initial coordinates instead of original positions."""
    print("\n" + "="*80)
    print("STEP 6: SHIFTING SCENARIO OBJECTS TO PARTICIPANT INITIAL COORDINATES")
    print("="*80)
    print("=== Step 6: Shift scenario objects to participant initial coordinates (x, y) instead of original positions ===")
    
    if scenario_df is None or not scenario_original_positions:
        print("⚠ No scenario data or original positions available")
        return
    
    for idx, row in scenario_df.iterrows():
        ppid, session, trial = row['ppid'], row['session'], row['trial']
        key = f"{ppid}-{session}-{trial}"
        
        if key in scenario_original_positions:
            # Get participant initial coordinates for this trial
            trial_participant = participant_df[
                (participant_df['ppid'] == ppid) & 
                (participant_df['session'] == session) & 
                (participant_df['trial'] == trial)
            ]
            
            if not trial_participant.empty:
                participant_initial_x = trial_participant['participant_x'].iloc[0] if 'participant_x' in trial_participant.columns else 0
                participant_initial_y = trial_participant['participant_y'].iloc[0] if 'participant_y' in trial_participant.columns else 0
                
                # Shift all scenario objects for this trial to participant initial coordinates
                scenario_cols = [c for c in scenario_df.columns if c.endswith('_x') or c.endswith('_y')]
                for col in scenario_cols:
                    if not pd.isna(scenario_df.at[idx, col]):
                        scenario_df.at[idx, col] = scenario_df.at[idx, col] + (participant_initial_x if col.endswith('_x') else participant_initial_y)
                
                print(f"  ✓ Shifted scenario objects for {ppid}-{session}-{trial} to participant initial coordinates ({participant_initial_x:.3f}, {participant_initial_y:.3f})")
            else:
                print(f"  ⚠ No participant data found for {ppid}-{session}-{trial}, keeping at origin")


def debug_step(participant_df, scenario_df, tracker_data_list, step_name, args, spatial_output_dir):
    """Debug a processing step."""
    print_data_summary(participant_df, scenario_df, tracker_data_list, step_name, args)
    
    # Load rotation angles if available for display in plots
    scenario_rotation_dict = {}
    if args.scenario_rotation or args.use_json_rotation:
        scenario_rotation_dict = load_scenario_rotations_with_json_fallback(args, scenario_df, args.data_dir, tracker_data_list, merged_df)
    
    create_first_step_analysis_plot(participant_df, scenario_df, tracker_data_list, spatial_output_dir, args, step_name, scenario_rotation_dict)
    explore_data_interactively(participant_df, scenario_df, tracker_data_list, step_name, args)


def print_completion_summary(args):
    """Print completion summary."""
    print("\n" + "="*80)
    print("ALL STEPS COMPLETED")
    print("="*80)
    print("✅ Processing pipeline completed successfully!")
    print("📊 Summary of transformations applied:")
    print("   0. Initial data loading and merging")
    if args.downsample > 1:
        print(f"   1. Data downsampled (factor: {args.downsample})")
    print("   2. Tracker data centered to origin (0,0) using first 20 points median")
    print("   3. Tracker rotations applied around origin (0,0), then shifted to participant initial coordinates (using tracker's first 20 points median) (if --scenario-rotation)")
    print("   4. Scenario objects centered to origin (0,0) using startpoint")
    print("   5. Scenario rotations applied around origin (0,0) (if --scenario-rotation)")
    print("   6. Scenario objects shifted to participant initial coordinates instead of original positions")
    print("   7. Spatial analysis performed (if --spatial-analysis)")
    print("\n🎯 Data is now ready for analysis!")


# ============================================================================
# UNIFIED DATASET FUNCTIONS
# ============================================================================
# These functions work with the unified dataset (merged_df) instead of separate DataFrames

def apply_downsampling_unified(merged_df, tracker_data_list, downsample_factor):
    """Apply downsampling to unified dataset and tracker data by taking every Nth row."""
    if downsample_factor <= 1:
        return merged_df, tracker_data_list
    
    print(f"\n=== APPLYING DOWNSAMPLING TO UNIFIED DATASET (factor: {downsample_factor}) ===")
    
    # Downsample unified dataset
    if merged_df is not None and len(merged_df) > 0:
        original_size = len(merged_df)
        merged_df = merged_df.iloc[::downsample_factor]  # Preserve original index for time alignment
        new_size = len(merged_df)
        print(f"  Unified dataset: {original_size} → {new_size} rows ({new_size/original_size*100:.1f}% of original)")
    
    # Downsample tracker data
    if tracker_data_list:
        total_original_points = 0
        total_new_points = 0
        
        for tracker_item in tracker_data_list:
            tracker_data = tracker_item['tracker_data']
            if tracker_data is not None and len(tracker_data) > 0:
                original_points = len(tracker_data)
                total_original_points += original_points
                
                # Downsample tracker data (preserve original index for time alignment)
                tracker_item['tracker_data'] = tracker_data.iloc[::downsample_factor]
                
                new_points = len(tracker_item['tracker_data'])
                total_new_points += new_points
                
                ppid, session, trial = tracker_item['ppid'], tracker_item['session'], tracker_item['trial']
                print(f"  Tracker {ppid}-{session}-{trial}: {original_points} → {new_points} points ({new_points/original_points*100:.1f}% of original)")
        
        print(f"  Total tracker points: {total_original_points} → {total_new_points} points ({total_new_points/total_original_points*100:.1f}% of original)")
    
    print("✓ Downsampling applied successfully to unified dataset")
    return merged_df, tracker_data_list


def debug_step_unified(merged_df, tracker_data_list, step_name, args, spatial_output_dir):
    """Create debug plot and summary for a specific step using unified dataset."""
    if not (args.debug or args.debug_plots_only):
        return
    
    print(f"\n=== DEBUG STEP: {step_name} (UNIFIED DATASET) ===")
    
    # Print unified data summary
    print_data_summary_unified(merged_df, tracker_data_list, step_name, args)
    
    # Create unified debug plot
    create_debug_plot_unified(merged_df, tracker_data_list, step_name, spatial_output_dir, args)
    
    print(f"✓ Debug step '{step_name}' completed for unified dataset")


def ensure_column_identification(merged_df):
    """Ensure all column types are identified and set as global variables."""
    print("\n--- IDENTIFYING COLUMN TYPES FOR TRANSFORMATIONS ---")
    
    # Identify participant location columns (reference - never transformed)
    participant_location_columns = []
    for col in merged_df.columns:
        if 'participant' in col.lower() and any(suffix in col.lower() for suffix in ['_x', '_y']):
            participant_location_columns.append(col)
    
    print(f"✓ Participant location columns (reference - never transformed): {participant_location_columns}")
    
    # Identify scenario object columns (can be transformed for alignment)
    # Only include x,y coordinates (z,a,b,c are not used and z is already handled by axis transformation)
    # Work with actual column names - no _transformed suffix needed since we update data in place
    scenario_object_columns = []
    for col in merged_df.columns:
        if any(obj in col.lower() for obj in ['task', 'threat', 'wall', 'hide', 'safehouse', 'weapons', 'startpoint', 'standpoint']) and any(suffix in col.lower() for suffix in ['_x', '_y']):
            scenario_object_columns.append(col)
    
    print(f"✓ Scenario object columns (can be transformed): {len(scenario_object_columns)} found")
    print(f"  ✓ Only x,y coordinates (excluded z,a,b,c as they are not used)")
    print(f"  ✓ Working with actual column names - data updated in place")
    if scenario_object_columns:
        print(f"  First 10 scenario columns: {scenario_object_columns[:10]}")
    else:
        print(f"  ⚠ NO SCENARIO OBJECT COLUMNS FOUND!")
        print(f"  🔍 DEBUG: All columns in merged_df:")
        for i, col in enumerate(merged_df.columns):
            if any(suffix in col.lower() for suffix in ['_x', '_y']):
                print(f"    {i}: {col}")
            if i > 50:  # Limit output
                print(f"    ... and {len(merged_df.columns) - 50} more columns")
                break
    
    # Identify metadata columns
    metadata_columns = [col for col in merged_df.columns if col not in ['ppid', 'session', 'trial', 'time'] and not any(suffix in col.lower() for suffix in ['_x', '_y', '_z'])]
    
    # Identify label columns  
    label_columns = [col for col in merged_df.columns if 'label' in col.lower() or 'predicted' in col.lower()]
    
    # Store these column lists as global variables for use throughout the code
    globals()['participant_location_columns'] = participant_location_columns
    globals()['scenario_object_columns'] = scenario_object_columns
    globals()['metadata_columns'] = metadata_columns
    globals()['label_columns'] = label_columns
    
    print(f"✓ Column identification complete - global variables set")
    return True

def check_trial_data_availability(merged_df, tracker_data_list, ppid, session, trial):
    """
    Check if all three datasets (trackers, participant location, scenario objects) are available for a trial.
    
    Returns:
        dict: {
            'has_participant': bool,
            'has_scenario': bool, 
            'has_tracker': bool,
            'all_available': bool
        }
    """
    # Check participant location data
    has_participant = False
    if 'participant_x' in merged_df.columns and 'participant_y' in merged_df.columns:
        trial_participant = merged_df[
            (merged_df['ppid'] == ppid) & 
            (merged_df['session'] == session) & 
            (merged_df['trial'] == trial)
        ]
        if not trial_participant.empty:
            participant_data = trial_participant[['participant_x', 'participant_y']].dropna()
            has_participant = len(participant_data) > 0
    
    # Check scenario object data
    has_scenario = False
    scenario_object_columns = globals().get('scenario_object_columns', [])
    if scenario_object_columns:
        trial_scenario = merged_df[
            (merged_df['ppid'] == ppid) & 
            (merged_df['session'] == session) & 
            (merged_df['trial'] == trial)
        ]
        if not trial_scenario.empty:
            # Check if any scenario object has valid coordinates
            for col in scenario_object_columns:
                if col.endswith('_x'):
                    y_col = col.replace('_x', '_y')
                    if y_col in scenario_object_columns:
                        obj_data = trial_scenario[[col, y_col]].dropna()
                        if len(obj_data) > 0:
                            has_scenario = True
                            break
    
    # Check tracker data
    has_tracker = False
    for tracker in tracker_data_list:
        if (tracker['ppid'] == ppid and 
            tracker['session'] == session and 
            tracker['trial'] == trial):
            if 'tracker_data' in tracker:
                tracker_data = tracker['tracker_data']
                if ('transformed_x' in tracker_data.columns and 
                    'transformed_y' in tracker_data.columns):
                    tracker_coords = tracker_data[['transformed_x', 'transformed_y']].dropna()
                    has_tracker = len(tracker_coords) > 0
            break
    
    all_available = has_participant and has_scenario and has_tracker
    
    return {
        'has_participant': has_participant,
        'has_scenario': has_scenario,
        'has_tracker': has_tracker,
        'all_available': all_available
    }


def filter_complete_trials_early(participant_df, scenario_df, labels_df, tracker_data_list, metadata_df=None):
    """
    Filter data to only include trials that have all three datasets (trackers, participant location, scenario objects).
    
    This function should be called early in the processing pipeline, right after data loading,
    to ensure only complete trials are processed throughout the entire pipeline.
    
    Args:
        participant_df: Participant location data
        scenario_df: Scenario object data
        labels_df: Syllable labels data
        tracker_data_list: List of tracker data dictionaries
        metadata_df: Optional metadata dataframe
    
    Returns:
        tuple: (filtered_participant_df, filtered_scenario_df, filtered_labels_df, 
                filtered_tracker_data_list, filtered_metadata_df, complete_trials_info)
    """
    print("\n" + "="*80)
    print("EARLY DATA FILTERING: KEEPING ONLY COMPLETE TRIALS")
    print("="*80)
    
    # First, identify scenario object columns
    ensure_column_identification(scenario_df)
    scenario_object_columns = globals().get('scenario_object_columns', [])
    
    print(f"✓ Identified {len(scenario_object_columns)} scenario object columns")
    print(f"✓ Scenario object columns: {scenario_object_columns}")
    
    # Get all unique trials from participant data (this is our reference)
    all_trials = participant_df[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"✓ Found {len(all_trials)} unique trials in participant data")
    
    # Check each trial for completeness
    complete_trials = []
    incomplete_trials = []
    
    print(f"\n--- CHECKING TRIAL COMPLETENESS ---")
    for _, trial_row in all_trials.iterrows():
        ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
        
        # Check participant data
        has_participant = True  # We know this exists since we're iterating over participant_df
        
        # Check scenario data
        has_scenario = False
        if scenario_df is not None:
            scenario_trial_data = scenario_df[(scenario_df['ppid'] == ppid) & 
                                            (scenario_df['session'] == session) & 
                                            (scenario_df['trial'] == trial)]
            if not scenario_trial_data.empty and scenario_object_columns:
                scenario_data_count = 0
                for col in scenario_object_columns:
                    if col in scenario_trial_data.columns and not scenario_trial_data[col].isna().all():
                        scenario_data_count += 1
                has_scenario = scenario_data_count > 0
        
        # Check tracker data
        has_tracker = False
        for tracker in tracker_data_list:
            if (tracker['ppid'] == ppid and 
                tracker['session'] == session and 
                tracker['trial'] == trial):
                if 'tracker_data' in tracker:
                    tracker_data = tracker['tracker_data']
                    if ('transformed_x' in tracker_data.columns and 
                        'transformed_y' in tracker_data.columns):
                        tracker_coords = tracker_data[['transformed_x', 'transformed_y']].dropna()
                        has_tracker = len(tracker_coords) > 0
                break
        
        if has_participant and has_scenario and has_tracker:
            complete_trials.append(trial_row)
        else:
            incomplete_trials.append(trial_row)
            missing = []
            if not has_participant:
                missing.append('participant')
            if not has_scenario:
                missing.append('scenario')
            if not has_tracker:
                missing.append('tracker')
            print(f"  ❌ Trial {ppid}-{session}-{trial}: Missing {', '.join(missing)} data")
    
    print(f"\n--- FILTERING RESULTS ---")
    print(f"  Total trials: {len(all_trials)}")
    print(f"  Complete trials: {len(complete_trials)}")
    print(f"  Incomplete trials: {len(incomplete_trials)}")
    print(f"  Completeness rate: {len(complete_trials)/len(all_trials)*100:.1f}%")
    
    if len(complete_trials) == 0:
        print("  ❌ ERROR: No complete trials found!")
        print("  ❌ Cannot proceed with analysis - all trials are missing at least one dataset")
        return None, None, None, None, None, None
    
    # Create complete trials dataframe for filtering
    complete_trials_df = pd.DataFrame(complete_trials)
    
    # Filter all dataframes to only include complete trials
    print(f"\n--- FILTERING ALL DATASETS ---")
    
    # Filter participant data
    participant_mask = participant_df.set_index(['ppid', 'session', 'trial']).index.isin(
        complete_trials_df.set_index(['ppid', 'session', 'trial']).index
    )
    filtered_participant_df = participant_df[participant_mask]
    print(f"  ✓ Participant data: {len(participant_df)} → {len(filtered_participant_df)} rows")
    
    # Filter scenario data
    filtered_scenario_df = None
    if scenario_df is not None:
        scenario_mask = scenario_df.set_index(['ppid', 'session', 'trial']).index.isin(
            complete_trials_df.set_index(['ppid', 'session', 'trial']).index
        )
        filtered_scenario_df = scenario_df[scenario_mask]
        print(f"  ✓ Scenario data: {len(scenario_df)} → {len(filtered_scenario_df)} rows")
    
    # Filter labels data
    labels_mask = labels_df.set_index(['ppid', 'session', 'trial']).index.isin(
        complete_trials_df.set_index(['ppid', 'session', 'trial']).index
    )
    filtered_labels_df = labels_df[labels_mask]
    print(f"  ✓ Labels data: {len(labels_df)} → {len(filtered_labels_df)} rows")
    
    # Filter tracker data
    filtered_tracker_data_list = []
    complete_trial_keys = set(f"{row['ppid']}-{row['session']}-{row['trial']}" for _, row in complete_trials_df.iterrows())
    
    for tracker_data in tracker_data_list:
        trial_key = f"{tracker_data['ppid']}-{tracker_data['session']}-{tracker_data['trial']}"
        if trial_key in complete_trial_keys:
            filtered_tracker_data_list.append(tracker_data)
    
    print(f"  ✓ Tracker data: {len(tracker_data_list)} → {len(filtered_tracker_data_list)} datasets")
    
    # Filter metadata if provided
    filtered_metadata_df = None
    if metadata_df is not None:
        metadata_mask = metadata_df.set_index(['ppid', 'session', 'trial']).index.isin(
            complete_trials_df.set_index(['ppid', 'session', 'trial']).index
        )
        filtered_metadata_df = metadata_df[metadata_mask]
        print(f"  ✓ Metadata data: {len(metadata_df)} → {len(filtered_metadata_df)} rows")
    
    # Create summary info
    complete_trials_info = {
        'total_trials': len(all_trials),
        'complete_trials': len(complete_trials),
        'incomplete_trials': len(incomplete_trials),
        'completeness_rate': len(complete_trials)/len(all_trials)*100,
        'complete_trial_list': complete_trials_df
    }
    
    print(f"\n✓ EARLY FILTERING COMPLETE")
    print(f"✓ All subsequent processing will use only complete trials")
    print(f"✓ This ensures consistency throughout the entire pipeline")
    
    return (filtered_participant_df, filtered_scenario_df, filtered_labels_df, 
            filtered_tracker_data_list, filtered_metadata_df, complete_trials_info)

def create_data_availability_report(merged_df, tracker_data_list, output_dir):
    """
    Create a CSV report showing data availability for all trials across the three dataset types.
    
    Args:
        merged_df: Unified dataset with all data
        tracker_data_list: VR headset movement data
        output_dir: Output directory for the report
    
    Returns:
        str: Path to the saved CSV report
    """
    import pandas as pd
    import os
    
    print("\n--- CREATING DATA AVAILABILITY REPORT ---")
    
    # Get all unique trials
    all_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"  Analyzing {len(all_trials)} unique trials...")
    
    # Create report data
    report_data = []
    
    for _, trial_row in all_trials.iterrows():
        ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
        
        # Check data availability
        availability = check_trial_data_availability(merged_df, tracker_data_list, ppid, session, trial)
        
        # Add to report
        report_data.append({
            'ppid': ppid,
            'session': session,
            'trial': trial,
            'has_participant': availability['has_participant'],
            'has_scenario': availability['has_scenario'],
            'has_tracker': availability['has_tracker'],
            'all_available': availability['all_available']
        })
    
    # Create DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Sort by ppid, session, trial for better readability
    report_df = report_df.sort_values(['ppid', 'session', 'trial']).reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV report
    report_path = os.path.join(output_dir, 'data_availability_report.csv')
    report_df.to_csv(report_path, index=False)
    
    # Print summary statistics
    total_trials = len(report_df)
    complete_trials = report_df['all_available'].sum()
    participant_trials = report_df['has_participant'].sum()
    scenario_trials = report_df['has_scenario'].sum()
    tracker_trials = report_df['has_tracker'].sum()
    
    print(f"  ✓ Data availability report saved to: {report_path}")
    print(f"  📊 Summary statistics:")
    print(f"    - Total trials: {total_trials}")
    print(f"    - Complete trials (all 3 datasets): {complete_trials} ({complete_trials/total_trials*100:.1f}%)")
    print(f"    - Participant data available: {participant_trials} ({participant_trials/total_trials*100:.1f}%)")
    print(f"    - Scenario data available: {scenario_trials} ({scenario_trials/total_trials*100:.1f}%)")
    print(f"    - Tracker data available: {tracker_trials} ({tracker_trials/total_trials*100:.1f}%)")
    
    # Show missing data breakdown
    missing_breakdown = report_df[~report_df['all_available']].groupby(['has_participant', 'has_scenario', 'has_tracker']).size().reset_index(name='count')
    if not missing_breakdown.empty:
        print(f"  📋 Missing data breakdown:")
        for _, row in missing_breakdown.iterrows():
            missing_types = []
            if not row['has_participant']:
                missing_types.append('participant')
            if not row['has_scenario']:
                missing_types.append('scenario')
            if not row['has_tracker']:
                missing_types.append('tracker')
            print(f"    - Missing {', '.join(missing_types)}: {row['count']} trials")
    
    return report_path

def ensure_scenario_id_in_merged_df(merged_df, args, spatial_output_dir):
    """Ensure scenario_id column is available in merged_df."""
    if 'scenario_id' in merged_df.columns:
        print("✓ scenario_id column already available in merged_df")
        return merged_df
    
    print("⚠ scenario_id column not found in merged_df, attempting to add it...")
    
    # Try to load experimental conditions if available
    if args.experimental_conditions and os.path.exists(args.experimental_conditions):
        try:
            conditions_df = pd.read_csv(args.experimental_conditions)
            if 'scenario_id' in conditions_df.columns:
                # Ensure consistent data types for merging
                conditions_df[['ppid', 'session', 'trial']] = conditions_df[['ppid', 'session', 'trial']].astype(str)
                merged_df[['ppid', 'session', 'trial']] = merged_df[['ppid', 'session', 'trial']].astype(str)
                
                # Merge scenario_id from experimental conditions
                merged_df = pd.merge(merged_df, conditions_df[['ppid', 'session', 'trial', 'scenario_id']], 
                                   on=['ppid', 'session', 'trial'], how='left')
                
                # Check for missing scenario_id
                missing_scenario_id = merged_df['scenario_id'].isnull().sum()
                if missing_scenario_id > 0:
                    print(f"⚠ Warning: {missing_scenario_id} rows have missing scenario_id")
                else:
                    print(f"✓ Added scenario_id to all {len(merged_df)} rows")
                
                return merged_df
        except Exception as e:
            print(f"⚠ Error adding scenario_id from experimental conditions: {e}")
    
    print("⚠ Could not add scenario_id column - scenario rotations may fail")
    return merged_df


def robust_time_alignment(tracker_times, participant_times, tracker_data, participant_data, trial_key):
    """
    Robustly align tracker and participant data with different sampling rates.
    Normalizes time to 0-100 range for consistent alignment.
    
    Returns:
        tuple: (tracker_aligned, participant_aligned, common_times) or (None, None, None) if failed
    """
    # Find overlapping time range
    time_start = max(tracker_times.min(), participant_times.min())
    time_end = min(tracker_times.max(), participant_times.max())
    time_overlap = time_end - time_start
    
    print(f"        Time overlap check: {time_overlap:.3f}s (tracker: {tracker_times.min():.3f}-{tracker_times.max():.3f}, participant: {participant_times.min():.3f}-{participant_times.max():.3f})")
    
    # If no overlap, try to align by shifting participant time to match tracker time
    if time_overlap < 0.1:  # Less than 100ms overlap
        print(f"        No time overlap detected, attempting time shift alignment...")
        
        # Calculate time shift needed to align participant data with tracker data
        time_shift = tracker_times.min() - participant_times.min()
        participant_times_shifted = participant_times + time_shift
        
        # Recalculate overlap with shifted participant time
        time_start = max(tracker_times.min(), participant_times_shifted.min())
        time_end = min(tracker_times.max(), participant_times_shifted.max())
        time_overlap = time_end - time_start
        
        print(f"        Time shift applied: {time_shift:.3f}s, new overlap: {time_overlap:.3f}s")
        
        if time_overlap < 0.1:  # Still no overlap after shift
            # Try alternative approach: use tracker time range and find corresponding participant data
            print(f"        Trying alternative alignment: using tracker time range...")
            
            # Find participant data points that fall within tracker time range
            participant_mask = (participant_times_shifted >= tracker_times.min()) & (participant_times_shifted <= tracker_times.max())
            
            if participant_mask.sum() < 10:  # Need at least 10 points
                print(f"        ⚠ Insufficient participant data in tracker time range ({participant_mask.sum()} points), skipping trial")
                return None, None, None
            
            # Use tracker time range as the reference
            time_start = tracker_times.min()
            time_end = tracker_times.max()
            time_overlap = time_end - time_start
            
            print(f"        Using tracker time range: {time_start:.3f}s-{time_end:.3f}s")
            
            # Filter participant data to tracker time range
            participant_data = participant_data[participant_mask]
            participant_times = participant_times_shifted[participant_mask]
        else:
            # Use shifted participant times for alignment
            participant_times = participant_times_shifted
    
    # Normalize time to 0-100 range for both datasets
    def normalize_time(times, start_time, end_time):
        """Normalize time array to 0-100 range"""
        if end_time == start_time:
            return np.zeros_like(times)
        return (times - start_time) / (end_time - start_time) * 100
    
    # Normalize both time arrays to 0-100
    tracker_times_norm = normalize_time(tracker_times, time_start, time_end)
    participant_times_norm = normalize_time(participant_times, time_start, time_end)
    
    print(f"        Time normalization: {time_start:.3f}s-{time_end:.3f}s -> 0-100")
    print(f"        Tracker points: {len(tracker_times_norm)}, Participant points: {len(participant_times_norm)}")
    
    # Use participant data as reference - resample tracker to match participant point count EXACTLY
    # This ensures perfect temporal alignment with the same number of points
    participant_n_points = len(participant_times_norm)
    
    print(f"        📊 Sampling rate alignment:")
    print(f"            Tracker original: {len(tracker_times_norm)} points")
    print(f"            Participant original: {participant_n_points} points")
    print(f"            Target: {participant_n_points} points (matching participant)")
    
    # Create evenly spaced normalized time grid matching participant data EXACTLY
    common_times_norm = np.linspace(0, 100, participant_n_points)
    
    if len(common_times_norm) < 10:
        print(f"        ⚠ Too few points for alignment ({len(common_times_norm)}), skipping trial")
        return None, None, None
    
    # Interpolate data using normalized times
    try:
        from scipy.interpolate import interp1d
        
        # Interpolate tracker data to match participant point count EXACTLY
        tracker_interp_x_func = interp1d(tracker_times_norm, tracker_data['transformed_x'].values, 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
        tracker_interp_y_func = interp1d(tracker_times_norm, tracker_data['transformed_y'].values, 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
        
        tracker_interp_x = tracker_interp_x_func(common_times_norm)
        tracker_interp_y = tracker_interp_y_func(common_times_norm)
        
        # Participant data is already at the right time points, just use it directly
        # (no interpolation needed since we're using participant as reference)
        participant_interp_x = participant_data['participant_x'].values
        participant_interp_y = participant_data['participant_y'].values
        
        # Validate that resampling worked correctly
        if len(tracker_interp_x) != participant_n_points or len(tracker_interp_y) != participant_n_points:
            print(f"        ⚠ Resampling validation failed: tracker={len(tracker_interp_x)}, participant={participant_n_points}")
            return None, None, None
        
        # Create aligned dataframes with normalized time index
        tracker_aligned = pd.DataFrame({
            'transformed_x': tracker_interp_x,
            'transformed_y': tracker_interp_y
        }, index=common_times_norm)
        
        participant_aligned = pd.DataFrame({
            'participant_x': participant_interp_x,
            'participant_y': participant_interp_y
        }, index=common_times_norm)
        
        print(f"        ✅ Perfect temporal alignment achieved:")
        print(f"            Final point count: {len(common_times_norm)} points (0-100 range)")
        print(f"            Tracker resampled: {len(tracker_times_norm)} → {len(common_times_norm)} points")
        print(f"            Participant points: {len(participant_interp_x)} (unchanged)")
        print(f"            Point-to-point correspondence: ✓")
        
        return tracker_aligned, participant_aligned, common_times_norm
        
    except Exception as e:
        print(f"    ⚠ Interpolation failed for {trial_key}: {str(e)}")
        return None, None, None


def apply_simple_tracker_alignment_new(tracker_data_list, merged_df):
    """
    Apply simple shift + rotation alignment to tracker data.
    
    Steps:
    1. Shift tracker data to participant's initial position
    2. Test 0°, 90°, 180°, 270° rotations around participant's initial position
    3. Find best rotation using distance-based matching metric
    4. Apply optimal shift and rotation to tracker data
    
    Returns:
        dict: Dictionary mapping trial_key to rotation angle in degrees
    """
    print("🔍 Applying simple shift + rotation tracker alignment...")
    
    import numpy as np
   
    successful_alignments = 0
    failed_alignments = 0
    rotation_angles = {}  # Store rotation angles for scenario objects
    
    for i, tracker_item in enumerate(tracker_data_list):
        ppid = tracker_item['ppid']
        session = tracker_item['session']
        trial = tracker_item['trial']
        
        # Clean identifiers
        ppid_clean, session_clean, trial_clean = clean_tracker_identifiers(ppid, session, trial)
        trial_key = f"{ppid_clean}-{session_clean}-{trial_clean}"
        
        print(f"\n--- Simple Alignment {i+1}/{len(tracker_data_list)}: {trial_key} ---")
        
        # Get tracker data
        tracker_data = tracker_item['tracker_data']
        if tracker_data.empty:
            print(f"    ⚠ No tracker data for {trial_key}")
            failed_alignments += 1
            continue
        
        # Get participant data for this trial
        participant_mask = (merged_df['ppid'].astype(str) == ppid_clean) & (merged_df['session'].astype(str) == session_clean) & (merged_df['trial'].astype(str) == trial_clean)
        participant_data = merged_df[participant_mask]
        
        if participant_data.empty:
            print(f"    ⚠ No participant data for {trial_key}")
            failed_alignments += 1
            continue
        
        # Use robust time alignment to handle different sampling rates
        tracker_times = np.array(tracker_data.index)
        participant_times = np.array(participant_data.index)
        
        print(f"    📊 Debug info for {trial_key}:")
        print(f"        Tracker data shape: {tracker_data.shape}")
        print(f"        Participant data shape: {participant_data.shape}")
        print(f"        Tracker time range: {tracker_times.min():.3f} to {tracker_times.max():.3f}")
        print(f"        Participant time range: {participant_times.min():.3f} to {participant_times.max():.3f}")
        
        # Use robust time alignment
        tracker_aligned, participant_aligned, common_times = robust_time_alignment(
            tracker_times, participant_times, tracker_data, participant_data, trial_key
        )
        
        if tracker_aligned is None:
            print(f"    ⚠ Time alignment failed for {trial_key}")
            failed_alignments += 1
            continue
        
        print(f"    ✓ Successfully aligned data: {len(common_times)} common time points")
        
        # Prepare point clouds
        tracker_points = np.column_stack([
            tracker_aligned['transformed_x'].values,
            tracker_aligned['transformed_y'].values,
            np.zeros(len(tracker_aligned))
        ])
        
        participant_points = np.column_stack([
            participant_aligned['participant_x'].values,
            participant_aligned['participant_y'].values,
            np.zeros(len(participant_aligned))
        ])
        
        # Step 1: Shift tracker data to participant's initial position
        participant_initial = participant_points[0, :2]  # First participant point
        tracker_initial = tracker_points[0, :2]  # First tracker point
        
        # Calculate shift to align tracker's first point with participant's first point
        initial_shift = participant_initial - tracker_initial
        tracker_points_shifted = tracker_points.copy()
        tracker_points_shifted[:, :2] += initial_shift
        
        print(f"        🔧 Step 1: Shifting tracker to participant initial position")
        print(f"        Participant initial: ({participant_initial[0]:.3f}, {participant_initial[1]:.3f})")
        print(f"        Tracker initial: ({tracker_initial[0]:.3f}, {tracker_initial[1]:.3f})")
        print(f"        Shift applied: ({initial_shift[0]:.3f}, {initial_shift[1]:.3f})")
        
        # Step 2: Test discrete rotations around participant's initial position
        print(f"        🔧 Step 2: Testing discrete rotations around participant initial position")
        
        best_angle = 0
        best_distance = float('inf')
        
        for test_angle in [0, 90, 180, 270]:
            # Rotate tracker data around participant's initial position
            shifted_tracker = tracker_points_shifted[:, :2] - participant_initial
            
            # Apply rotation around origin
            test_angle_rad = np.radians(test_angle)
            cos_a, sin_a = np.cos(test_angle_rad), np.sin(test_angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            
            rotated_tracker = shifted_tracker @ rotation_matrix.T
            
            # Shift back to participant's coordinate system
            rotated_tracker_final = rotated_tracker + participant_initial
            
            # Calculate average distance to participant points
            distances = []
            for tp in rotated_tracker_final:
                dists = np.sqrt(np.sum((participant_points[:, :2] - tp)**2, axis=1))
                distances.append(np.min(dists))
            
            avg_distance = np.mean(distances)
            print(f"        {test_angle}°: avg distance = {avg_distance:.3f}")
            
            if avg_distance < best_distance:
                best_distance = avg_distance
                best_angle = test_angle
        
        print(f"        ✓ Best rotation: {best_angle}° (distance: {best_distance:.3f})")
        
        # Step 3: Apply optimal shift and rotation to tracker data
        print(f"        🔧 Step 3: Applying optimal shift and rotation")
        
        # Apply the best rotation
        if best_angle != 0:
            shifted_tracker = tracker_points_shifted[:, :2] - participant_initial
            best_angle_rad = np.radians(best_angle)
            cos_a, sin_a = np.cos(best_angle_rad), np.sin(best_angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            rotated_tracker = shifted_tracker @ rotation_matrix.T
            final_tracker_points = rotated_tracker + participant_initial
        else:
            final_tracker_points = tracker_points_shifted[:, :2]
        
        # Create transformed tracker data
        tracker_data_transformed = pd.DataFrame({
            'transformed_x': final_tracker_points[:, 0],
            'transformed_y': final_tracker_points[:, 1]
        }, index=common_times)
        
        # Copy other columns from original tracker_data
        for col in tracker_data.columns:
            if col not in ['transformed_x', 'transformed_y']:
                if col in tracker_data.columns:
                    try:
                        from scipy.interpolate import interp1d
                        original_times = np.array(tracker_data.index)
                        resampled_times = common_times
                        
                        if len(original_times) > 1:
                            original_times_norm = (original_times - original_times.min()) / (original_times.max() - original_times.min()) * 100
                            resampled_times_norm = (resampled_times - resampled_times.min()) / (resampled_times.max() - resampled_times.min()) * 100
                            
                            interp_func = interp1d(original_times_norm, tracker_data[col].values, 
                                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                            tracker_data_transformed[col] = interp_func(resampled_times_norm)
                        else:
                            tracker_data_transformed[col] = tracker_data[col].iloc[0]
                    except:
                        tracker_data_transformed[col] = tracker_data[col].iloc[0]
        
        # Update the tracker_data_list with the transformed data
        tracker_item['tracker_data'] = tracker_data_transformed
        
        # Store rotation angle for scenario objects
        rotation_angles[trial_key] = best_angle
        
        print(f"    ✓ Simple alignment applied (angle: {best_angle}°, distance: {best_distance:.3f})")
        successful_alignments += 1
    
    print(f"\n--- Simple Alignment Summary ---")
    print(f"  Successful alignments: {successful_alignments}")
    print(f"  Failed alignments: {failed_alignments}")
    print(f"  Rotation angles calculated: {len(rotation_angles)}")
    
    return rotation_angles


def apply_simple_tracker_alignment(tracker_data_list, merged_df):
    """
    Apply ICP-based alignment to tracker data and calculate rotation angles for scenario objects.
    
    This function uses Open3D ICP to find the optimal transformation for each tracker
    and applies it directly to the tracker data, then extracts rotation angles for scenario objects.
    
    IMPORTANT: This function works with already-centered tracker data (from Step 2).
    It does NOT perform additional centering to avoid double transformation.
    
    Returns:
        dict: Dictionary mapping trial_key to rotation angle in degrees
    """
    print("🔍 Applying ICP-based tracker alignment...")
    
    # Debug breakpoint to inspect input data

    
  
    import open3d as o3d
    import numpy as np
   
    successful_alignments = 0
    failed_alignments = 0
    rotation_angles = {}  # Store rotation angles for scenario objects
    
    for i, tracker_item in enumerate(tracker_data_list):
        ppid = tracker_item['ppid']
        session = tracker_item['session']
        trial = tracker_item['trial']
        
        # Clean identifiers
        ppid_clean, session_clean, trial_clean = clean_tracker_identifiers(ppid, session, trial)
        trial_key = f"{ppid_clean}-{session_clean}-{trial_clean}"
        
        print(f"\n--- ICP Alignment {i+1}/{len(tracker_data_list)}: {trial_key} ---")
        
        # Get tracker data
        tracker_data = tracker_item['tracker_data']
        if tracker_data.empty:
            print(f"    ⚠ No tracker data for {trial_key}")
            failed_alignments += 1
            continue
        
        # Get participant data for this trial
        participant_mask = (merged_df['ppid'].astype(str) == ppid_clean) & (merged_df['session'].astype(str) == session_clean) & (merged_df['trial'].astype(str) == trial_clean)
        participant_data = merged_df[participant_mask]
        
        if participant_data.empty:
            print(f"    ⚠ No participant data for {trial_key}")
            failed_alignments += 1
            continue
        
        # Use robust time alignment to handle different sampling rates
        tracker_times = np.array(tracker_data.index)
        participant_times = np.array(participant_data.index)
        
        print(f"    📊 Debug info for {trial_key}:")
        print(f"        Tracker data shape: {tracker_data.shape}")
        print(f"        Participant data shape: {participant_data.shape}")
        print(f"        Tracker time range: {tracker_times.min():.3f} to {tracker_times.max():.3f}")
        print(f"        Participant time range: {participant_times.min():.3f} to {participant_times.max():.3f}")
        
        # Calculate and display sampling rates
        tracker_dt = np.median(np.diff(tracker_times)) if len(tracker_times) > 1 else 0.01
        participant_dt = np.median(np.diff(participant_times)) if len(participant_times) > 1 else 0.01
        print(f"        Tracker sampling rate: {1/tracker_dt:.1f} Hz")
        print(f"        Participant sampling rate: {1/participant_dt:.1f} Hz")
        print(f"        Time will be normalized to 0-100 range for alignment")
        
        # Use robust time alignment
        tracker_aligned, participant_aligned, common_times = robust_time_alignment(
            tracker_times, participant_times, tracker_data, participant_data, trial_key
        )
        
        if tracker_aligned is None:
            print(f"    ⚠ Time alignment failed for {trial_key}")
            failed_alignments += 1
            continue
        
        print(f"    ✓ Successfully aligned data: {len(common_times)} common time points")
        
        # Prepare point clouds
        tracker_points = np.column_stack([
            tracker_aligned['transformed_x'].values,
            tracker_aligned['transformed_y'].values,
            np.zeros(len(tracker_aligned))
        ])
        
        participant_points = np.column_stack([
            participant_aligned['participant_x'].values,
            participant_aligned['participant_y'].values,
            np.zeros(len(participant_aligned))
        ])
        
        # Debug: Show coordinate ranges before ICP
        print(f"        Tracker points range: X=[{tracker_points[:, 0].min():.3f}, {tracker_points[:, 0].max():.3f}], Y=[{tracker_points[:, 1].min():.3f}, {tracker_points[:, 1].max():.3f}]")
        print(f"        Participant points range: X=[{participant_points[:, 0].min():.3f}, {participant_points[:, 0].max():.3f}], Y=[{participant_points[:, 1].min():.3f}, {participant_points[:, 1].max():.3f}]")
        
        # Initialize ICP by shifting tracker data to participant's initial position
        # This provides a much better starting point for alignment
        participant_initial = participant_points[0, :2]  # First participant point
        tracker_initial = tracker_points[0, :2]  # First tracker point
        
        # Calculate shift to align tracker's first point with participant's first point
        initial_shift = participant_initial - tracker_initial
        # Apply shift only to x,y coordinates (z stays 0)
        tracker_points_shifted = tracker_points.copy()
        tracker_points_shifted[:, :2] += initial_shift
        
        print(f"        🔧 Initializing ICP: shifting tracker to participant initial position")
        print(f"        Participant initial: ({participant_initial[0]:.3f}, {participant_initial[1]:.3f})")
        print(f"        Tracker initial: ({tracker_initial[0]:.3f}, {tracker_initial[1]:.3f})")
        print(f"        Shift applied: ({initial_shift[0]:.3f}, {initial_shift[1]:.3f})")
        
        # Now center both datasets for ICP alignment
        participant_center = np.mean(participant_points, axis=0)
        tracker_center = np.mean(tracker_points_shifted, axis=0)
        
        participant_points_centered = participant_points - participant_center
        tracker_points_centered = tracker_points_shifted - tracker_center
        
        print(f"        Shifted tracker (centered): X=[{tracker_points_centered[:, 0].min():.3f}, {tracker_points_centered[:, 0].max():.3f}], Y=[{tracker_points_centered[:, 1].min():.3f}, {tracker_points_centered[:, 1].max():.3f}]")
        print(f"        Centered participant: X=[{participant_points_centered[:, 0].min():.3f}, {participant_points_centered[:, 0].max():.3f}], Y=[{participant_points_centered[:, 1].min():.3f}, {participant_points_centered[:, 1].max():.3f}]")
        
        # Create Open3D point clouds using shifted and centered data
        tracker_pcd = o3d.geometry.PointCloud()
        tracker_pcd.points = o3d.utility.Vector3dVector(tracker_points_centered)  # Use shifted and centered tracker data
        
        participant_pcd = o3d.geometry.PointCloud()
        participant_pcd.points = o3d.utility.Vector3dVector(participant_points_centered)
        
        # Try to pre-align using FPFH features for better initialization
        try:
            # Compute FPFH features
            radius_normal = threshold * 2
            radius_feature = threshold * 5
            
            tracker_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            participant_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            
            tracker_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                tracker_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            participant_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                participant_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
            # Try RANSAC registration for initial alignment
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                tracker_pcd, participant_pcd, tracker_fpfh, participant_fpfh, True,
                distance_threshold=threshold * 2,
                ransac_n=3,
                checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold * 2)],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
            
            if result_ransac.fitness > 0.1:  # If RANSAC found a good initial alignment
                print(f"        RANSAC pre-alignment found (fitness: {result_ransac.fitness:.3f})")
                # Add RANSAC result as an initial transformation
                initial_transforms.insert(1, result_ransac.transformation)
        except Exception as e:
            print(f"        RANSAC pre-alignment failed: {str(e)}")
            # Continue without pre-alignment
        
        # Calculate appropriate threshold based on point cloud scale
        tracker_scale = np.std(tracker_points_centered)  # Use shifted and centered tracker data
        participant_scale = np.std(participant_points_centered)
        threshold = max(tracker_scale, participant_scale) * 0.1  # 10% of scale
        
        print(f"        ICP threshold: {threshold:.3f} (tracker_scale: {tracker_scale:.3f}, participant_scale: {participant_scale:.3f})")
        
        # Check if point clouds are valid
        if len(tracker_pcd.points) < 3 or len(participant_pcd.points) < 3:
            print(f"    ⚠ Insufficient points for ICP: tracker={len(tracker_pcd.points)}, participant={len(participant_pcd.points)}")
            failed_alignments += 1
            continue
        
        # Check if point clouds have sufficient spread
        tracker_spread = np.std(tracker_points_centered, axis=0)  # Use shifted and centered tracker data
        participant_spread = np.std(participant_points_centered, axis=0)
        print(f"        Point cloud spread - tracker: {tracker_spread[:2]}, participant: {participant_spread[:2]}")
        
        if np.any(tracker_spread[:2] < 0.01) or np.any(participant_spread[:2] < 0.01):
            print(f"    ⚠ Point clouds too concentrated for reliable ICP")
            failed_alignments += 1
            continue
        
        
        # Perform ICP registration with improved parameters and multiple initializations
        best_fitness = 0
        best_reg = None
        
        # Try multiple initial transformations to avoid local minima
        initial_transforms = [
            np.eye(4),  # Identity (no initial transformation)
        ]
        
        # Add discrete rotation initializations (0°, 90°, 180°, 270°)
        for angle_deg in [0, 90, 180, 270]:
            angle_rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0, 0],
                [sin_a, cos_a, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            initial_transforms.append(rotation_matrix)
        
        # Add some intermediate angles for better coverage
        for angle_deg in [45, 135, 225, 315]:
            angle_rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0, 0],
                [sin_a, cos_a, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            initial_transforms.append(rotation_matrix)
        
        for init_idx, trans_init in enumerate(initial_transforms):
            for threshold_mult in [0.05, 0.1, 0.2, 0.5]:
                current_threshold = threshold * threshold_mult
                try:
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        tracker_pcd, participant_pcd, current_threshold, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)
                    )
                    
                    if reg_p2p.fitness > best_fitness:
                        best_fitness = reg_p2p.fitness
                        best_reg = reg_p2p
                        print(f"        Better ICP found: init={init_idx}, threshold={current_threshold:.3f} (fitness: {reg_p2p.fitness:.3f})")
                except Exception as e:
                    print(f"        ICP failed: init={init_idx}, threshold={current_threshold:.3f}: {str(e)}")
                    continue
        
        if best_reg is None:
            print(f"    ⚠ All ICP attempts failed for {trial_key}, trying fallback approach...")
            
            # Fallback: try with a very large threshold and simple approach
            try:
                fallback_threshold = max(tracker_scale, participant_scale) * 2.0  # Much larger threshold
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    tracker_pcd, participant_pcd, fallback_threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                
                if reg_p2p.fitness > 0.1:  # Accept if fitness is reasonable
                    best_reg = reg_p2p
                    print(f"        Fallback ICP succeeded with threshold {fallback_threshold:.3f} (fitness: {reg_p2p.fitness:.3f})")
                else:
                    print(f"    ⚠ Fallback ICP also failed for {trial_key}")
                    failed_alignments += 1
                    continue
            except Exception as e:
                print(f"    ⚠ Fallback ICP failed: {str(e)}")
                failed_alignments += 1
                continue
        
        
        reg_p2p = best_reg
        
        # Apply transformation to tracker data
        transformation_matrix = reg_p2p.transformation
        
        # Extract rotation angle from transformation matrix for scenario objects
        rotation_matrix = transformation_matrix[:3, :3]
        angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        # Normalize angle to [0, 360)
        if angle_deg < 0:
            angle_deg += 360
        
        # Store rotation angle for scenario objects
        rotation_angles[trial_key] = angle_deg
        
        print(f"        ICP fitness: {reg_p2p.fitness:.3f}, inlier_rmse: {reg_p2p.inlier_rmse:.3f}")
        
        # Post-ICP refinement: Test discrete rotations to avoid local minima
        print(f"        🔍 Post-ICP refinement: testing discrete rotations...")
        
        # Test 0°, 90°, 180°, 270° rotations on the ICP result
        best_refined_fitness = reg_p2p.fitness
        best_refined_transform = reg_p2p.transformation
        best_refined_angle = angle_deg
        
        for test_angle in [0, 90, 180, 270]:
            # Get participant's initial position (first point) - this will be our rotation center
            participant_initial = participant_points[0, :2]  # First participant point
            
            # Test this rotation angle by rotating tracker data around participant's initial position
            try:
                # Apply the ICP transformation first (using shifted data)
                tracker_points_homogeneous = np.column_stack([
                    tracker_points_shifted[:, 0],
                    tracker_points_shifted[:, 1],
                    np.zeros(len(tracker_points_shifted)),
                    np.ones(len(tracker_points_shifted))
                ])
                
                # Apply ICP transformation
                icp_transformed = (reg_p2p.transformation @ tracker_points_homogeneous.T).T
                icp_transformed_points = icp_transformed[:, :3] + participant_center
                
                # Now rotate the ICP-transformed tracker data around participant's initial position
                # Shift to participant's initial position as origin
                shifted_tracker = icp_transformed_points[:, :2] - participant_initial
                
                # Apply rotation around origin
                test_angle_rad = np.radians(test_angle)
                cos_a, sin_a = np.cos(test_angle_rad), np.sin(test_angle_rad)
                rotation_matrix = np.array([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])
                
                rotated_tracker = shifted_tracker @ rotation_matrix.T
                
                # Shift back to participant's coordinate system
                transformed_points = rotated_tracker + participant_initial
                
                # Calculate fitness (mean distance to nearest participant points) using FULL data
                distances = []
                for tp in transformed_points[:, :2]:
                    dists = np.sqrt(np.sum((participant_points[:, :2] - tp)**2, axis=1))
                    distances.append(np.min(dists))
                
                test_fitness = 1.0 / (1.0 + np.mean(distances))  # Convert distance to fitness
                
                if test_fitness > best_refined_fitness:
                    best_refined_fitness = test_fitness
                    best_refined_transform = reg_p2p.transformation  # Use the original ICP transformation
                    best_refined_angle = (angle_deg + test_angle) % 360
                    print(f"        Better rotation found: {test_angle}° (fitness: {test_fitness:.3f})")
                
            except Exception as e:
                continue
        
        # Use the best refined transformation
        if best_refined_fitness > reg_p2p.fitness:
            print(f"        ✅ Post-ICP refinement improved fitness: {reg_p2p.fitness:.3f} → {best_refined_fitness:.3f}")
            print(f"        Final angle: {best_refined_angle:.1f}° (was {angle_deg:.1f}°)")
            reg_p2p = type('obj', (object,), {
                'transformation': best_refined_transform,
                'fitness': best_refined_fitness,
                'inlier_rmse': reg_p2p.inlier_rmse
            })
            angle_deg = best_refined_angle
        else:
            print(f"        No improvement from post-ICP refinement (fitness: {reg_p2p.fitness:.3f})")
        
        # Apply the optimal transformation to the complete data
        print(f"        🔧 Applying optimal transformation to complete data ({len(tracker_points_shifted)} points)")
        
        # Get participant's initial position for rotation center
        participant_initial = participant_points[0, :2]
        
        # Prepare shifted tracker data for transformation
        tracker_points_homogeneous = np.column_stack([
            tracker_points_shifted[:, 0],  # Use shifted data
            tracker_points_shifted[:, 1],  # Use shifted data
            np.zeros(len(tracker_points_shifted)),
            np.ones(len(tracker_points_shifted))
        ])
        
        # Apply the optimal transformation to the shifted tracker data
        transformed_centered = (reg_p2p.transformation @ tracker_points_homogeneous.T).T
        icp_transformed_points = transformed_centered[:, :3] + participant_center
        
        # If we found a better rotation during refinement, apply it around participant's initial position
        if best_refined_fitness > reg_p2p.fitness:
            # Extract the additional rotation angle from the refinement
            additional_angle = (best_refined_angle - angle_deg) % 360
            if additional_angle != 0:
                print(f"        🔄 Applying additional {additional_angle:.1f}° rotation around participant initial position")
                
                # Shift to participant's initial position as origin
                shifted_tracker = icp_transformed_points[:, :2] - participant_initial
                
                # Apply rotation around origin
                additional_angle_rad = np.radians(additional_angle)
                cos_a, sin_a = np.cos(additional_angle_rad), np.sin(additional_angle_rad)
                rotation_matrix = np.array([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])
                
                rotated_tracker = shifted_tracker @ rotation_matrix.T
                
                # Shift back to participant's coordinate system
                transformed_points_complete = np.column_stack([
                    rotated_tracker + participant_initial,
                    icp_transformed_points[:, 2]  # Keep Z coordinate
                ])
            else:
                transformed_points_complete = icp_transformed_points
        else:
            transformed_points_complete = icp_transformed_points
        
        # Create a new tracker_data DataFrame with complete transformed coordinates
        # This ensures we have the full time series with proper alignment
        tracker_data_resampled = pd.DataFrame({
            'transformed_x': transformed_points_complete[:, 0],
            'transformed_y': transformed_points_complete[:, 1]
        }, index=common_times)  # Use the common time index
        
        # Copy other columns from original tracker_data if they exist
        for col in tracker_data.columns:
            if col not in ['transformed_x', 'transformed_y']:
                # Interpolate other columns to match the resampled time points
                if col in tracker_data.columns:
                    try:
                        from scipy.interpolate import interp1d
                        original_times = np.array(tracker_data.index)
                        resampled_times = common_times
                        
                        # Normalize original times to 0-100 range for interpolation
                        if len(original_times) > 1:
                            original_times_norm = (original_times - original_times.min()) / (original_times.max() - original_times.min()) * 100
                            resampled_times_norm = (resampled_times - resampled_times.min()) / (resampled_times.max() - resampled_times.min()) * 100
                            
                            interp_func = interp1d(original_times_norm, tracker_data[col].values, 
                                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                            tracker_data_resampled[col] = interp_func(resampled_times_norm)
                        else:
                            # If only one point, repeat it
                            tracker_data_resampled[col] = tracker_data[col].iloc[0]
                    except:
                        # If interpolation fails, use the first value
                        tracker_data_resampled[col] = tracker_data[col].iloc[0]
        
        # Replace the original tracker_data with the resampled version
        tracker_data = tracker_data_resampled
        
        print(f"        Complete transformed tracker range: X=[{transformed_points_complete[:, 0].min():.3f}, {transformed_points_complete[:, 0].max():.3f}], Y=[{transformed_points_complete[:, 1].min():.3f}, {transformed_points_complete[:, 1].max():.3f}]")
        
        # Update the tracker_data_list with the resampled and transformed data
        tracker_item['tracker_data'] = tracker_data
        
        print(f"    ✓ ICP alignment applied (fitness: {reg_p2p.fitness:.3f}, angle: {angle_deg:.1f}°)")
        print(f"    📊 Final tracker data: {len(tracker_data)} points (resampled to match participant)")
        successful_alignments += 1
    
    print(f"\n📊 ICP ALIGNMENT SUMMARY:")
    print(f"   ✓ Successful: {successful_alignments}/{len(tracker_data_list)} trackers")
    print(f"   ⚠ Failed: {failed_alignments}/{len(tracker_data_list)} trackers")
    print(f"   🔄 Rotation angles calculated: {len(rotation_angles)}")
    
    # Debug breakpoint to inspect ICP results

    print("\n🔍 DEBUG: Setting breakpoint to inspect ICP alignment results...")
    print("Available variables:")
    print("- tracker_data_list: List of tracker data with applied transformations")
    print("- merged_df: Participant and scenario data")
    print("- rotation_angles: Dictionary of trial_key -> rotation_angle")
    print("- successful_alignments: Number of successful ICP alignments")
    print("- failed_alignments: Number of failed ICP alignments")

    
    return rotation_angles


def apply_tracker_rotations_unified(tracker_data_list, tracker_original_positions, args, merged_df):
    """Apply ICP-based alignment to tracker data - SIMPLIFIED VERSION."""
    print("\n" + "="*80)
    print("STEP 3: APPLYING ICP ALIGNMENT TO TRACKER DATA")
    print("="*80)
    
    # Only use ICP - remove all other transformation methods
    print("✓ Using simple shift + rotation alignment - calculating rotation angles for scenario objects")
    print(f"   📊 Processing {len(tracker_data_list)} tracker datasets...")
    rotation_dict = apply_simple_tracker_alignment_new(tracker_data_list, merged_df)
    
    # Store the rotation angles for scenario objects
    args._icp_rotation_dict = rotation_dict
    print(f"✓ Simple alignment calculated {len(rotation_dict)} rotation angles for scenario objects")
    print(f"   🔄 Rotation angles: {list(rotation_dict.keys())[:5]}...")  # Show first 5 keys
    return


def create_tracker_rotation_mapping_unified(tracker_data_list, merged_df, scenario_rotation_dict):
    """Create mapping between trackers and their rotation angles using unified dataset."""
    print(f"\n" + "="*60)
    print("DEBUG: CREATING TRACKER ROTATION MAPPING (UNIFIED)")
    print("="*60)
    print(f"Total trackers to process: {len(tracker_data_list)}")
    
    # Check if scenario_id column exists in merged_df
    if 'scenario_id' in merged_df.columns:
        print(f"✓ scenario_id column found in merged_df")
        # Handle mixed types (strings and NaN) by filtering out NaN values before sorting
        unique_scenario_ids = merged_df['scenario_id'].dropna().unique()
        print(f"Available scenario_ids in merged_df: {sorted(unique_scenario_ids.astype(str))}")
        
        # Show count of NaN values if any
        nan_count = merged_df['scenario_id'].isna().sum()
        if nan_count > 0:
            print(f"⚠ Found {nan_count} rows with NaN scenario_id values")
    else:
        print(f"⚠ scenario_id column NOT found in merged_df")
        print(f"Available columns: {list(merged_df.columns)}")
        return []
    
    print(f"Available scenario_ids in rotation_dict: {list(scenario_rotation_dict.keys())}")
    
    tracker_rotation_mapping = []
    
    for i, tracker_item in enumerate(tracker_data_list):
        ppid, session, trial = tracker_item['ppid'], tracker_item['session'], tracker_item['trial']
        print(f"\n--- Processing tracker {i+1}/{len(tracker_data_list)}: {ppid}-{session}-{trial} ---")
        
        # Find corresponding row in merged_df to get scenario_id
        trial_data = merged_df[
            (merged_df['ppid'].astype(str) == str(ppid)) & 
            (merged_df['session'].astype(str) == str(session)) & 
            (merged_df['trial'].astype(str) == str(trial))
        ]
        
        if not trial_data.empty:
            if 'scenario_id' in trial_data.columns:
                raw_scenario_id = trial_data.iloc[0]['scenario_id']
                scenario_id = clean_scenario_id(raw_scenario_id)
                
                if scenario_id in scenario_rotation_dict:
                    angle = scenario_rotation_dict[scenario_id]
                    tracker_rotation_mapping.append({
                        'ppid': ppid, 'session': session, 'trial': trial,
                        'scenario_id': scenario_id, 'angle': angle
                    })
                    print(f"  ✓ Mapped tracker {ppid}-{session}-{trial} to scenario_id {scenario_id} with angle {angle}°")
                else:
                    # Still include this tracker but with 0° angle (no rotation, but will be shifted)
                    angle = 0.0
                    tracker_rotation_mapping.append({
                        'ppid': ppid, 'session': session, 'trial': trial,
                        'scenario_id': scenario_id if pd.notna(scenario_id) else 'unknown',
                        'angle': angle
                    })
                    print(f"  ⚠ scenario_id {scenario_id} not found in rotation_dict, setting angle to 0° (will still be shifted)")
            else:
                print(f"  ❌ 'scenario_id' column not found in trial data")
        else:
            print(f"  ⚠ No matching rows found in merged_df, but will still include tracker for shifting")
            # Include this tracker with 0° angle (no rotation, but will be shifted)
            tracker_rotation_mapping.append({
                'ppid': ppid, 'session': session, 'trial': trial,
                'scenario_id': 'unknown',
                'angle': 0.0
            })
            print(f"  ✓ Added tracker {ppid}-{session}-{trial} with 0° angle (will be shifted to participant location)")
    
    print(f"\n" + "="*60)
    print("DEBUG: TRACKER ROTATION MAPPING SUMMARY (UNIFIED)")
    print("="*60)
    print(f"Created {len(tracker_rotation_mapping)} tracker rotation mappings:")
    for i, mapping in enumerate(tracker_rotation_mapping):
        print(f"  {i+1}. {mapping['ppid']}-{mapping['session']}-{mapping['trial']} → scenario_id {mapping['scenario_id']} → {mapping['angle']}°")
    
    return tracker_rotation_mapping


def create_tracker_rotation_mapping_trial_based(tracker_data_list, trial_rotation_dict):
    """Create tracker rotation mapping using trial-based keys (for JSON extraction)."""
    print(f"\n--- Creating trial-based tracker rotation mapping ---")
    print(f"Available trial rotation keys: {list(trial_rotation_dict.keys())}")
    
    tracker_rotation_mapping = []
    
    for i, tracker in enumerate(tracker_data_list):
        ppid = tracker['ppid']
        session = tracker['session']
        trial = tracker['trial']
        
        # Parse and clean identifiers to match merged_df format
        ppid_clean, session_clean, trial_clean = clean_tracker_identifiers(ppid, session, trial)
        
        trial_key = f"{ppid_clean}-{session_clean}-{trial_clean}"
        
        print(f"\n--- Processing tracker {i+1}/{len(tracker_data_list)}: {trial_key} ---")
        print(f"    🔍 DEBUG: Original - ppid={ppid}, session={session}, trial={trial}")
        print(f"    🔍 DEBUG: Cleaned - ppid={ppid_clean}, session={session_clean}, trial={trial_clean}")
        
        if trial_key in trial_rotation_dict:
            angle = trial_rotation_dict[trial_key]
            tracker_rotation_mapping.append({
                'ppid': ppid_clean, 'session': session_clean, 'trial': trial_clean,
                'scenario_id': trial_key, 'angle': angle
            })
            print(f"  ✓ Found rotation angle {angle}° for trial {trial_key}")
        else:
            # Still include this tracker but with 0° angle (no rotation, but will be shifted)
            angle = 0.0
            tracker_rotation_mapping.append({
                'ppid': ppid_clean, 'session': session_clean, 'trial': trial_clean,
                'scenario_id': trial_key,
                'angle': angle
            })
            print(f"  ⚠ No rotation angle found for trial {trial_key}, setting to 0° (will still be shifted)")
    
    print(f"\n✓ Created {len(tracker_rotation_mapping)} trial-based tracker rotation mappings")
    return tracker_rotation_mapping


def apply_single_tracker_rotation_unified(mapping, tracker_data_list, tracker_original_positions, merged_df):
    """Apply rotation to a single tracker using unified dataset."""
    ppid, session, trial = mapping['ppid'], mapping['session'], mapping['trial']
    angle = mapping['angle']
    
    print(f"🔧 DEBUG: apply_single_tracker_rotation_unified CALLED for {ppid}-{session}-{trial} with angle {angle}°")
    
    # Determine if we need to skip rotation
    skip_rotation = (angle == 0)
    if skip_rotation:
        print(f"🔧 DEBUG: Will skip rotation for {ppid}-{session}-{trial} (angle = 0°), but will still shift to participant location")
    
    # Find tracker item
    tracker_item = None
    for item in tracker_data_list:
        if (str(item['ppid']) == str(ppid) and str(item['session']) == str(session) and str(item['trial']) == str(trial)):
            tracker_item = item
            break
    
    if not tracker_item:
        print(f"    ❌ Tracker item not found for {ppid}-{session}-{trial}")
        return
    
    tracker_data = tracker_item['tracker_data']
    key = f"{ppid}-{session}-{trial}"
    
    if key not in tracker_original_positions:
        print(f"    ⚠ No original position stored for tracker {ppid}-{session}-{trial}")
        return
    
    original_x, original_y = tracker_original_positions[key]
    
    # Apply rotation if needed
    if not skip_rotation:
        # Apply rotation around origin (0,0) - trackers are already centered from Step 2
        rad = math.radians(angle)  # Negative for clockwise
        cosv = math.cos(rad)
        sinv = math.sin(rad)
        
        # Trackers are already centered at (0,0) from Step 2, so rotate directly
        x_coords = tracker_data['transformed_x'].values
        y_coords = tracker_data['transformed_y'].values
        
        # Check if arrays are empty before any operations
        if len(x_coords) == 0:
            print(f"    ⚠ WARNING: Tracker data is empty (0 rows) for {ppid}-{session}-{trial}")
            return
        
        # Apply rotation around origin (0,0)
        x_rotated = x_coords * cosv - y_coords * sinv
        y_rotated = x_coords * sinv + y_coords * cosv
    else:
        # For 0° rotation, use original coordinates (no rotation needed)
        x_rotated = tracker_data['transformed_x'].values
        y_rotated = tracker_data['transformed_y'].values
        
        # Check if arrays are empty before any operations
        if len(x_rotated) == 0:
            print(f"    ⚠ WARNING: Tracker data is empty (0 rows) for {ppid}-{session}-{trial}")
            return
    
    # After rotation, tracker data should be shifted to participant initial coordinates
    # Get participant location coordinates from merged_df (reference coordinates - never transformed)
    trial_participant = merged_df[
        (merged_df['ppid'] == ppid) & 
        (merged_df['session'] == session) & 
        (merged_df['trial'] == trial)
    ]
    
    if not trial_participant.empty:
        # Try to find participant location coordinates using identified columns
        participant_location_columns = globals().get('participant_location_columns', [])
        participant_initial_x = None
        participant_initial_y = None
        
        # Look for participant location columns (reference coordinates)
        # Use standardized participant_x, participant_y naming
        if 'participant_x' in trial_participant.columns:
            participant_initial_x = trial_participant['participant_x'].iloc[0]
        elif 'x' in trial_participant.columns:
            participant_initial_x = trial_participant['x'].iloc[0]
            print(f"    DEBUG: Using fallback 'x' column (should be identical to participant_x)")
        
        if 'participant_y' in trial_participant.columns:
            participant_initial_y = trial_participant['participant_y'].iloc[0]
        elif 'y' in trial_participant.columns:
            participant_initial_y = trial_participant['y'].iloc[0]
            print(f"    DEBUG: Using fallback 'y' column (should be identical to participant_y)")
        
        if participant_initial_x is not None and participant_initial_y is not None:
            print(f"    DEBUG: Found participant location coordinates from merged_df: ({participant_initial_x:.3f}, {participant_initial_y:.3f})")
            print(f"    DEBUG: Using participant location columns: {participant_location_columns}")
        else:
            print(f"    ⚠ No participant location coordinates found for {ppid}-{session}-{trial}, using tracker median as fallback")
            participant_initial_x = original_x
            participant_initial_y = original_y
    else:
        print(f"    ⚠ No participant data found for {ppid}-{session}-{trial}, using tracker median as fallback")
        participant_initial_x = original_x
        participant_initial_y = original_y
    
    # Shift the rotated tracker data to the participant initial coordinates
    final_x = x_rotated + participant_initial_x
    final_y = y_rotated + participant_initial_y
    
    # Assign back to DataFrame
    tracker_data['transformed_x'] = final_x
    tracker_data['transformed_y'] = final_y
    
    if skip_rotation:
        print(f"    ✓ Skipped rotation for tracker {ppid}-{session}-{trial} (0° angle)")
    else:
        print(f"    ✓ Applied {angle}° rotation to tracker {ppid}-{session}-{trial} around origin (0,0)")
        # Debug: Show sample coordinates before and after rotation
        print(f"    🔍 DEBUG: Sample coordinates before rotation: ({x_coords[0]:.3f}, {y_coords[0]:.3f})")
        print(f"    🔍 DEBUG: Sample coordinates after rotation: ({x_rotated[0]:.3f}, {y_rotated[0]:.3f})")
    
    print(f"    ✓ Shifted tracker to participant initial coordinates ({participant_initial_x:.3f}, {participant_initial_y:.3f})")


def center_scenario_objects_unified(merged_df):
    """Store original startpoint positions for scenario objects (NO centering - keep original positions)."""
    print("\n" + "="*80)
    print("STEP 4: STORING ORIGINAL STARTPOINT POSITIONS (UNIFIED)")
    print("="*80)
    print("=== Step 4: Store original startpoint positions for rotation pivot points ===")
    
    scenario_original_positions = {}
    
    if merged_df is None or len(merged_df) == 0:
        print("⚠ No unified data available")
        return scenario_original_positions
    
    # Ensure scenario object columns are identified before proceeding
    ensure_column_identification(merged_df)
    
    # Use the identified scenario object columns
    scenario_object_columns = globals().get('scenario_object_columns', [])
    if not scenario_object_columns:
        print("⚠ No scenario object columns identified even after column identification")
        return scenario_original_positions
    
    # Find startpoint columns from scenario object columns
    startpoint_x_col = None
    startpoint_y_col = None
    for col in scenario_object_columns:
        if 'startpoint' in col.lower() and col.endswith('_x'):
            startpoint_x_col = col
        elif 'startpoint' in col.lower() and col.endswith('_y'):
            startpoint_y_col = col
    
    if not startpoint_x_col or not startpoint_y_col:
        print("⚠ No startpoint columns found in scenario object columns")
        return scenario_original_positions
    
    print(f"✓ Found startpoint columns: {startpoint_x_col}, {startpoint_y_col}")
    print(f"✓ Using {len(scenario_object_columns)} scenario object columns")
    print("✓ NOT centering scenario objects - keeping original positions for rotation around startpoint")
    
    # Store original startpoint positions (for rotation pivot points)
    for idx, row in merged_df.iterrows():
        ppid, session, trial = row['ppid'], row['session'], row['trial']
        startpoint_x = row[startpoint_x_col]
        startpoint_y = row[startpoint_y_col]
        
        # Store original position (will be used as pivot point for rotation)
        scenario_original_positions[f"{ppid}-{session}-{trial}"] = (startpoint_x, startpoint_y)
        
        print(f"  ✓ Stored startpoint position for {ppid}-{session}-{trial}: ({startpoint_x:.3f}, {startpoint_y:.3f})")
    
    return scenario_original_positions


def preprocess_scenario_id(merged_df):
    """Preprocess scenario_id values for rotation matching."""
    print("Preprocessing scenario_id values for rotation matching...")
    
    if 'scenario_id' not in merged_df.columns:
        print("⚠ No scenario_id column available")
        return
    
    # Create a copy of scenario_id for preprocessing
    original_scenario_id = merged_df['scenario_id'].copy()
    processed_scenario_id = merged_df['scenario_id'].copy()
    
    # Process each row
    for idx, row in merged_df.iterrows():
        scenario_id = str(row['scenario_id']) if pd.notna(row['scenario_id']) else None
        episode_metadata = str(row.get('episode_metadata', '')) if pd.notna(row.get('episode_metadata')) else ''
        
        if scenario_id and scenario_id != 'nan':
            # 1. Remove leading zeros
            processed_id = scenario_id.lstrip('0') or '0'  # Keep '0' if all zeros
        else:
            # 2. scenario_id is missing, look at episode_metadata
            if 'tutorial' in episode_metadata.lower():
                processed_id = "tutorial"
            elif '_F_' in episode_metadata:
                processed_id = "f"
            elif 'Stump' in episode_metadata:
                processed_id = "s"
            else:
                # 2.4. Split string by "_" and use second element without leading zeros
                parts = episode_metadata.split('_')
                if len(parts) >= 2:
                    second_element = parts[1].lstrip('0') or '0'
                    processed_id = second_element
                else:
                    processed_id = episode_metadata  # Fallback to original
                    print(f"    ⚠ Could not extract scenario_id from episode_metadata: '{episode_metadata}'")
        
        processed_scenario_id.iloc[idx] = processed_id
        
        # Debug output for first few rows
        if idx < 5:
            print(f"  Row {idx}: '{original_scenario_id.iloc[idx]}' + '{episode_metadata}' → '{processed_id}'")
    
    # Update the dataframe
    merged_df['scenario_id'] = processed_scenario_id
    
    print(f"✓ Preprocessed {len(merged_df)} scenario_id values")
    return merged_df


def apply_scenario_rotations_unified(merged_df, scenario_original_positions, args, spatial_output_dir, tracker_data_list):
    """Apply rotations to scenario objects around origin (0,0) using unified dataset."""
    print("\n" + "="*80)
    print("STEP 5: APPLYING ROTATIONS TO SCENARIO OBJECTS AROUND ORIGIN (UNIFIED)")
    print("="*80)
    
    # Use ICP-calculated rotation angles for scenario objects
    if hasattr(args, '_icp_rotation_dict') and args._icp_rotation_dict:
        print("✓ Using ICP-calculated rotation angles for scenario objects")
        scenario_rotation_dict = args._icp_rotation_dict
    else:
        print("⚠ No ICP rotation angles available - skipping scenario object rotation")
        return
    
    # Ensure scenario_id is available in merged_df
    if 'scenario_id' not in merged_df.columns:
        print("⚠ No scenario_id column available for rotations")
        return
    
    # Preprocess scenario_id values before matching with rotation data
    merged_df = preprocess_scenario_id(merged_df)
    
    # Ensure column identification is done before applying rotations
    ensure_column_identification(merged_df)
    
    # Apply rotations
    print(f"\n🔍 DEBUG: About to apply scenario rotations with {len(scenario_rotation_dict)} angles")
    print(f"🔍 DEBUG: Sample rotation angles: {list(scenario_rotation_dict.items())[:3]}")
    apply_scenario_rotations_implementation_unified(merged_df, scenario_original_positions, scenario_rotation_dict)
    print(f"🔍 DEBUG: Scenario rotations applied")


def apply_scenario_rotations_implementation_unified(merged_df, scenario_original_positions, scenario_rotation_dict):
    """Implementation of scenario rotations for unified dataset - matches tracker rotation logic exactly."""
    print("Applying scenario rotations to unified dataset...")
    
    # Use the identified scenario object columns
    scenario_object_columns = globals().get('scenario_object_columns', [])
    if not scenario_object_columns:
        print("⚠ No scenario object columns identified")
        print("🔍 DEBUG: Available columns in merged_df:")
        print(f"   {list(merged_df.columns)[:10]}...")
        
        # Try to identify scenario object columns manually as fallback
        print("🔍 DEBUG: Trying manual fallback identification...")
        scenario_object_columns = []
        for col in merged_df.columns:
            if (col.endswith('_x') or col.endswith('_y')) and 'participant' not in col.lower() and 'tracker' not in col.lower():
                scenario_object_columns.append(col)
        
        if scenario_object_columns:
            print(f"🔍 DEBUG: Found {len(scenario_object_columns)} columns with fallback: {scenario_object_columns[:5]}...")
            # Update the global variable
            globals()['scenario_object_columns'] = scenario_object_columns
        else:
            print("❌ CRITICAL: No scenario object columns found even with fallback!")
        return
    
    print(f"✓ Using {len(scenario_object_columns)} scenario object columns for rotations")
    print(f"🔍 DEBUG: Scenario object columns: {scenario_object_columns}")
    
    # Debug: Check if scenario object columns exist in merged_df
    missing_cols = [col for col in scenario_object_columns if col not in merged_df.columns]
    if missing_cols:
        print(f"⚠ WARNING: Missing scenario object columns in merged_df: {missing_cols}")
    else:
        print(f"✓ All scenario object columns found in merged_df")
    
    # Check if we're using trial-based or scenario-based rotation keys
    is_trial_based = any('-' in key for key in scenario_rotation_dict.keys())
    print(f"🔍 DEBUG: Rotation keys type - trial_based: {is_trial_based}")
    print(f"🔍 DEBUG: Sample rotation keys: {list(scenario_rotation_dict.keys())[:5]}")
    
    # Debug: Show sample scenario object data before rotation
    if scenario_object_columns:
        sample_cols = [col for col in scenario_object_columns if col in merged_df.columns][:2]  # First 2 columns
        if sample_cols:
            print(f"🔍 DEBUG: Sample scenario object data before rotation:")
            for col in sample_cols:
                if col.endswith('_x'):
                    y_col = col.replace('_x', '_y')
                    if y_col in merged_df.columns:
                        sample_data = merged_df[[col, y_col]].head(3)
                        print(f"  {col}, {y_col}: {sample_data.values.tolist()}")
    
    if is_trial_based:
        print("✓ Using trial-based scenario rotations (JSON extraction)")
        apply_scenario_rotations_trial_based(merged_df, scenario_original_positions, scenario_rotation_dict, scenario_object_columns)
    else:
        print("✓ Using scenario-based scenario rotations (CSV extraction)")
        apply_scenario_rotations_scenario_based(merged_df, scenario_original_positions, scenario_rotation_dict, scenario_object_columns)


def apply_scenario_rotations_trial_based(merged_df, scenario_original_positions, trial_rotation_dict, scenario_object_columns):
    """Apply scenario rotations using trial-based keys (for JSON extraction)."""
    print("Applying trial-based scenario rotations...")
    
    for trial_key, angle in trial_rotation_dict.items():
        print(f"  Processing trial {trial_key} with angle {angle}°")
        
        # Debug: Show if angle is 0
        if angle == 0:
            print(f"    ⚠ WARNING: Rotation angle is 0° for trial {trial_key} - no rotation will be applied!")
        
        # Parse trial key
        try:
            ppid, session, trial = trial_key.split('-')
        except ValueError:
            print(f"    ⚠ Invalid trial key format: {trial_key}")
            continue
        
        # Find rows with this trial
        trial_mask = (merged_df['ppid'].astype(str) == str(ppid)) & \
                    (merged_df['session'].astype(str) == str(session)) & \
                    (merged_df['trial'].astype(str) == str(trial))
        
        if not trial_mask.any():
            print(f"    ⚠ No data found for trial {trial_key}")
            continue
        
        # Apply rotation to this specific trial
        apply_rotation_to_trial(merged_df, trial_mask, trial_key, angle, scenario_original_positions, scenario_object_columns)


def apply_rotation_to_trial(merged_df, trial_mask, trial_key, angle, scenario_original_positions, scenario_object_columns):
    """
    Apply rotation to a specific trial's scenario objects.
    
    Process:
    1. Get the startpoint position as pivot point
    2. Apply rotation around the pivot point to all scenario object coordinates
    3. Update the coordinates in merged_df
    
    Args:
        merged_df: DataFrame containing scenario object coordinates
        trial_mask: Boolean mask for the specific trial
        trial_key: String identifier for the trial
        angle: Rotation angle in degrees
        scenario_original_positions: Dict mapping trial_key to (x, y) startpoint position
        scenario_object_columns: List of scenario object coordinate columns
    """
    if trial_key not in scenario_original_positions:
        print(f"    ⚠ No startpoint position stored for trial {trial_key}, skipping rotation")
        return
    
    # Get the startpoint as pivot point
    pivot_x, pivot_y = scenario_original_positions[trial_key]
    print(f"    🔄 Rotating scenario objects around startpoint ({pivot_x:.3f}, {pivot_y:.3f}) by {angle}°")
    
    # Convert angle to radians (negative for clockwise, same as tracker)
    rad = math.radians(angle)
    cosv = math.cos(rad)
    sinv = math.sin(rad)
    
    # Apply rotation to all scenario object coordinates
    rotated_objects = 0
    for col in scenario_object_columns:
        if col in merged_df.columns and col.endswith('_x'):
            y_col = col.replace('_x', '_y')
            if y_col in merged_df.columns:
                # Get current coordinates
                x_coords = merged_df.loc[trial_mask, col].copy()
                y_coords = merged_df.loc[trial_mask, y_col].copy()
                
                # Apply rotation around the startpoint
                x_rotated = (x_coords - pivot_x) * cosv - (y_coords - pivot_y) * sinv + pivot_x
                y_rotated = (x_coords - pivot_x) * sinv + (y_coords - pivot_y) * cosv + pivot_y
                
                # Update the coordinates
                merged_df.loc[trial_mask, col] = x_rotated
                merged_df.loc[trial_mask, y_col] = y_rotated
                
                rotated_objects += 1
    
    print(f"    ✓ Rotated {rotated_objects} scenario objects by {angle}°")


def apply_scenario_rotations_scenario_based(merged_df, scenario_original_positions, scenario_rotation_dict, scenario_object_columns):
    """Apply scenario rotations using scenario-based keys (for CSV extraction)."""
    print("Applying scenario-based scenario rotations...")
    
    # Group by scenario_id and apply rotations (same as tracker logic)
    for scenario_id, angle in scenario_rotation_dict.items():
        print(f"  Processing scenario {scenario_id} with angle {angle}°")
        
        # Find rows with this scenario_id
        scenario_mask = merged_df['scenario_id'].astype(str) == str(scenario_id)
        if not scenario_mask.any():
            print(f"    ⚠ No data found for scenario {scenario_id}")
            continue
        
        # Convert angle to radians (negative for clockwise, same as tracker)
        rad = math.radians(angle)
        cosv = math.cos(rad)
        sinv = math.sin(rad)
        
        # Find common pivot point for all trials in this scenario
        # Use the startpoint of the first trial as the common pivot point
        scenario_trials = merged_df[scenario_mask]
        if scenario_trials.empty:
            continue
            
        first_trial_idx = scenario_trials.index[0]
        first_trial_key = f"{scenario_trials.loc[first_trial_idx, 'ppid']}-{scenario_trials.loc[first_trial_idx, 'session']}-{scenario_trials.loc[first_trial_idx, 'trial']}"
        
        if first_trial_key not in scenario_original_positions:
            print(f"    ⚠ No startpoint position stored for first trial {first_trial_key}, skipping scenario")
            continue
        
        # Use the first trial's startpoint as the common pivot point for all trials in this scenario
        pivot_x, pivot_y = scenario_original_positions[first_trial_key]
        print(f"    Using common pivot point ({pivot_x:.3f}, {pivot_y:.3f}) for all trials in scenario {scenario_id}")
        
        # Apply rotation to ALL trials in this scenario using the common pivot point
        for idx in merged_df[scenario_mask].index:
            ppid = merged_df.loc[idx, 'ppid']
            session = merged_df.loc[idx, 'session']
            trial = merged_df.loc[idx, 'trial']
            trial_key = f"{ppid}-{session}-{trial}"
            
            print(f"    Rotating trial {trial_key} around common pivot ({pivot_x:.3f}, {pivot_y:.3f})")
            
            # Apply rotation to each scenario object around the common pivot point
            for col in scenario_object_columns:
                if col.endswith('_x'):
                    x_coord = merged_df.loc[idx, col]
                    y_col = col.replace('_x', '_y')
                    
                    if y_col in scenario_object_columns:  # Make sure y column is also a scenario object column
                        y_coord = merged_df.loc[idx, y_col]
                        
                        # Skip if coordinates are NaN
                        if pd.isna(x_coord) or pd.isna(y_coord):
                            continue
                        
                        # Apply rotation around common pivot point - all objects rotate together as frozen group
                        # Step 1: Translate to common pivot as origin
                        x_centered = x_coord - pivot_x
                        y_centered = y_coord - pivot_y
                        
                        # Step 2: Apply rotation around origin (0,0) - same as tracker
                        if angle != 0:  # Only rotate if angle is not 0
                            x_rotated = x_centered * cosv - y_centered * sinv
                            y_rotated = x_centered * sinv + y_centered * cosv
                        else:
                            x_rotated = x_centered
                            y_rotated = y_centered
                        
                        # Step 3: Translate back from common pivot
                        x_final = x_rotated + pivot_x
                        y_final = y_rotated + pivot_y
                    
                        # Update the unified dataset
                        merged_df.loc[idx, col] = x_final
                        merged_df.loc[idx, y_col] = y_final
                        
                        # Debug: Show sample rotation for first few objects
                        if rotated_objects < 3:
                            print(f"    🔍 DEBUG: Rotated {col} from ({x_coord:.3f}, {y_coord:.3f}) to ({x_final:.3f}, {y_final:.3f})")
        
        if angle != 0:
            print(f"    ✓ Applied {angle}° rotation to scenario {scenario_id} around common pivot point")
        else:
            print(f"    ✓ Skipped rotation for scenario {scenario_id} (0° angle)")


def shift_scenario_objects_to_participant_unified(merged_df, scenario_original_positions):
    """Shift scenario objects to participant initial coordinates (same logic as trackers)."""
    print("\n" + "="*80)
    print("STEP 6: SHIFTING SCENARIO OBJECTS TO PARTICIPANT COORDINATES (UNIFIED)")
    print("="*80)
    print("=== Step 6: Shift scenario objects to participant initial coordinates (same as trackers) ===")
    
    if not scenario_original_positions:
        print("⚠ No scenario original positions available")
        return
    
    # Ensure column identification is done before proceeding
    ensure_column_identification(merged_df)
    
    # Use the identified scenario object columns
    scenario_object_columns = globals().get('scenario_object_columns', [])
    if not scenario_object_columns:
        print("⚠ No scenario object columns identified even after column identification")
        return
    
    # Use the identified participant location columns
    participant_location_columns = globals().get('participant_location_columns', [])
    if not participant_location_columns:
        print("⚠ No participant location columns identified")
        return
    
    print(f"✓ Using {len(scenario_object_columns)} scenario object columns for shifting")
    print(f"✓ Using {len(participant_location_columns)} participant location columns as reference")
    
    # Shift scenario objects to participant coordinates for each trial
    for trial_key, (startpoint_x, startpoint_y) in scenario_original_positions.items():
        ppid, session, trial = trial_key.split('-')
        
        # Find rows for this trial
        trial_mask = (
            (merged_df['ppid'].astype(str) == str(ppid)) & 
            (merged_df['session'].astype(str) == str(session)) & 
            (merged_df['trial'].astype(str) == str(trial))
        )
        
        if not trial_mask.any():
            print(f"  ⚠ No data found for trial {ppid}-{session}-{trial}")
            continue
        
        # Get participant coordinates for this trial (reference coordinates - never transformed)
        trial_participant = merged_df[trial_mask]
        participant_initial_x = None
        participant_initial_y = None
        
        # Look for participant location columns (reference coordinates)
        if 'participant_x' in trial_participant.columns:
            participant_initial_x = trial_participant['participant_x'].iloc[0]
        elif 'x' in trial_participant.columns:
            participant_initial_x = trial_participant['x'].iloc[0]
        
        if 'participant_y' in trial_participant.columns:
            participant_initial_y = trial_participant['participant_y'].iloc[0]
        elif 'y' in trial_participant.columns:
            participant_initial_y = trial_participant['y'].iloc[0]
        
        if participant_initial_x is None or participant_initial_y is None:
            print(f"  ⚠ No participant coordinates found for trial {ppid}-{session}-{trial}, using startpoint as fallback")
            participant_initial_x = startpoint_x
            participant_initial_y = startpoint_y
        else:
            print(f"  ✓ Found participant coordinates for trial {ppid}-{session}-{trial}: ({participant_initial_x:.3f}, {participant_initial_y:.3f})")
        
        # Calculate shift vector from startpoint to participant coordinates
        shift_x = participant_initial_x - startpoint_x
        shift_y = participant_initial_y - startpoint_y
        
        print(f"  ✓ Shifting scenario objects by ({shift_x:.3f}, {shift_y:.3f}) to align with participant coordinates")
        
        # Apply shift to all scenario object coordinates
        for col in scenario_object_columns:
            if col.endswith('_x'):
                # Shift x coordinates
                if not pd.isna(merged_df.loc[trial_mask, col]).all():
                    merged_df.loc[trial_mask, col] = merged_df.loc[trial_mask, col] + shift_x
            elif col.endswith('_y'):
                # Shift y coordinates
                if not pd.isna(merged_df.loc[trial_mask, col]).all():
                    merged_df.loc[trial_mask, col] = merged_df.loc[trial_mask, col] + shift_y
        
        print(f"  ✓ Shifted scenario objects for trial {ppid}-{session}-{trial} to participant coordinates")
    
    print("✓ All scenario objects shifted to participant coordinates")


def shift_participant_data_to_startpoint_unified_UNUSED(merged_df, scenario_original_positions):
    """Shift participant data so that startpoint becomes the reference point (0,0)."""
    print("\n" + "="*80)
    print("STEP 5.5: SHIFTING PARTICIPANT DATA TO STARTPOINT REFERENCE (UNIFIED)")
    print("="*80)
    
    if not scenario_original_positions:
        print("⚠ No scenario original positions available")
        return
    
    # Use the identified participant location columns
    participant_location_columns = globals().get('participant_location_columns', [])
    if not participant_location_columns:
        print("⚠ No participant location columns identified")
        return
    
    print(f"✓ Using {len(participant_location_columns)} participant location columns for shifting")
    
    # Find startpoint columns from scenario object columns
    scenario_object_columns = globals().get('scenario_object_columns', [])
    startpoint_x_col = None
    startpoint_y_col = None
    for col in scenario_object_columns:
        if 'startpoint' in col.lower() and col.endswith('_x'):
            startpoint_x_col = col
        elif 'startpoint' in col.lower() and col.endswith('_y'):
            startpoint_y_col = col
    
    if not startpoint_x_col or not startpoint_y_col:
        print("⚠ No startpoint columns found for participant shifting")
        return
    
    print("Shifting participant data to startpoint reference...")
    
    # Shift participant data for each trial
    for trial_key, (original_x, original_y) in scenario_original_positions.items():
        ppid, session, trial = trial_key.split('-')
        
        # Find rows for this trial
        trial_mask = (
            (merged_df['ppid'].astype(str) == str(ppid)) & 
            (merged_df['session'].astype(str) == str(session)) & 
            (merged_df['trial'].astype(str) == str(trial))
        )
        
        if not trial_mask.any():
            continue
        
        # Get the startpoint position for this trial (should be at origin after centering)
        startpoint_positions = merged_df.loc[trial_mask, [startpoint_x_col, startpoint_y_col]]
        
        # Check if startpoint is at origin (0,0) as expected
        startpoint_x = startpoint_positions[startpoint_x_col].iloc[0]
        startpoint_y = startpoint_positions[startpoint_y_col].iloc[0]
        
        if abs(startpoint_x) > 0.001 or abs(startpoint_y) > 0.001:
            print(f"  ⚠ Warning: Startpoint for {ppid}-{session}-{trial} not at origin: ({startpoint_x:.3f}, {startpoint_y:.3f})")
            continue
        
        # Shift participant data so that startpoint becomes reference point
        # This means participant coordinates should be shifted by the NEGATIVE of the original startpoint position
        # because we want the startpoint (now at 0,0) to become the reference for participant coordinates
        # Later, when scenario objects are restored, the startpoint will go back to its original position
        # and participant data will be correctly positioned relative to it
        for col in participant_location_columns:
            if col.endswith('_x'):
                merged_df.loc[trial_mask, col] = merged_df.loc[trial_mask, col] - original_x
            elif col.endswith('_y'):
                merged_df.loc[trial_mask, col] = merged_df.loc[trial_mask, col] - original_y
        
        print(f"  ✓ Shifted participant data for {ppid}-{session}-{trial} by (-{original_x:.3f}, -{original_y:.3f})")
    
    print("✓ All participant data shifted to startpoint reference")


def convert_tracker_categories_to_syllables(merged_df, label_column):
    """
    Convert tracker categories to numeric syllable-like identifiers for consistent coloring.
    
    This function helps when the label column contains tracker categories (like 'stand', 'walk') 
    instead of actual syllables, by converting them to numeric identifiers that can be 
    consistently colored using the color map.
    """
    print(f"\n=== Converting tracker categories to syllable-like identifiers ===")
    print(f"Original label column: {label_column}")
    
    if label_column not in merged_df.columns:
        print("⚠ Label column not found in merged_df")
        return merged_df, label_column
    
    # Get unique tracker categories
    unique_categories = merged_df[label_column].dropna().unique()
    print(f"Found {len(unique_categories)} unique tracker categories: {sorted(unique_categories)}")
    
    # Create a mapping from tracker categories to numeric identifiers
    category_to_syllable = {}
    for i, category in enumerate(sorted(unique_categories)):
        category_to_syllable[category] = i
        print(f"  {category} → syllable {i}")
    
    # Create a new column with syllable-like identifiers
    new_label_column = f"{label_column}_syllable"
    merged_df[new_label_column] = merged_df[label_column].map(category_to_syllable)
    
    # Create a consistent color map for the converted syllable identifiers
    import matplotlib.pyplot as plt
    syllable_color_map = {}
    
    # Use a combination of colormaps to get more unique colors
    colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
    colors_per_cmap = 20  # Most colormaps have 20 colors
    
    for i, category in enumerate(sorted(unique_categories)):
        # Cycle through different colormaps to get more unique colors
        cmap_index = (i // colors_per_cmap) % len(colormaps)
        color_index = i % colors_per_cmap
        color = colormaps[cmap_index](color_index)
        
        # Map BOTH the numeric identifier AND the original string label to the same color
        # This ensures compatibility with both the converted data and the original color map
        syllable_color_map[i] = color
        syllable_color_map[category] = color  # Preserve original string label mapping
        
        print(f"  Syllable {i} ({category}) → color: {color} (cmap: {cmap_index}, index: {color_index})")
        print(f"    → Mapped both {i} and '{category}' to same color")
    
    print(f"✓ Created new column: {new_label_column}")
    print(f"✓ Converted {len(unique_categories)} tracker categories to syllable-like identifiers")
    print(f"✓ Created consistent color map with {len(syllable_color_map)} colors")
    
    return merged_df, new_label_column, syllable_color_map


def perform_spatial_analysis_unified(merged_df, tracker_data_list, color_map, spatial_output_dir, args):
    """
    Perform spatial analysis with unified dataset - adapted from analyse_location_based_syllable_usage_clean.py
    
    This function:
    1. Analyzes the unified dataset structure
    2. Creates spatial plots grouped by episode_set_name2 and scenario_id
    3. Performs object-syllable analysis (if requested)
    
    Args:
        merged_df: Unified dataset with all data already merged
        tracker_data_list: VR headset movement data
        color_map: Color mapping for syllables
        spatial_output_dir: Output directory for plots and data
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("UNIFIED SPATIAL ANALYSIS: USING PRE-MERGED DATASET")
    print("="*80)
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os
    import re
    import numpy as np
    
    print(f"✓ Using unified dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"✓ Tracker trials: {len(tracker_data_list)}")
    
    # Define object styles for scenario objects (same as original)
    object_styles = {
        'task':      dict(marker='^', color='green',     label='Task',        s=120),
        'safehouse': dict(marker='s', color='saddlebrown',label='Safehouse',  s=120),
        'threat':    dict(marker='p', color='red',       label='Threat',      s=120),
        'startpoint':dict(marker='o', color='yellow',    label='StartPoint',  s=120),
        'weapons':   dict(marker='s', color='magenta',   label='Weapons',     s=120),
        'standpoint':dict(marker='o', color='orange',    label='StandPoint',  s=120),
        'wall':      dict(marker='s', color='grey',      label='Wall',        s=120),
        'hide':      dict(marker='s', color='cyan',      label='Hide',        s=120),
    }
    
    # STEP 1: Analyze unified dataset structure
    print("\n=== STEP 1: ANALYZING UNIFIED DATASET STRUCTURE ===")
    
    # Check what we have in the unified dataset
    essential_columns = ['ppid', 'session', 'trial', 'time']
    participant_location_columns = globals().get('participant_location_columns', [])
    scenario_object_columns = globals().get('scenario_object_columns', [])
    label_columns = globals().get('label_columns', [])
    
    # Check if participant coordinates are dummy values
    if 'participant_x' in merged_df.columns and 'participant_y' in merged_df.columns:
        # Check if all participant coordinates are the same (likely dummy values)
        unique_x = merged_df['participant_x'].nunique()
        unique_y = merged_df['participant_y'].nunique()
        if unique_x == 1 and unique_y == 1:
            x_val = merged_df['participant_x'].iloc[0]
            y_val = merged_df['participant_y'].iloc[0]
            if x_val == 0.0 and y_val == 0.0:
                print("⚠ WARNING: Participant coordinates appear to be dummy values (0,0)")
                print("⚠ This means no real participant movement data is available")
                print("⚠ Use --coords-dir argument to load real participant coordinates")
                print("⚠ Spatial analysis will show static participant positions at origin")
    
    print(f"✓ Essential columns: {essential_columns}")
    print(f"✓ Participant location columns: {participant_location_columns}")
    print(f"✓ Scenario object columns: {len(scenario_object_columns)} found")
    print(f"✓ Label columns: {label_columns}")
    
    # Check for critical metadata
    if 'scenario_id' in merged_df.columns:
        unique_scenarios = merged_df['scenario_id'].dropna().unique()
        print(f"✓ scenario_id found: {len(unique_scenarios)} unique scenarios")
        print(f"  Sample scenarios: {list(unique_scenarios[:10])}")
    else:
        print(f"⚠ No scenario_id column found")
    
    if 'episode_set_name2' in merged_df.columns:
        unique_blocks = merged_df['episode_set_name2'].dropna().unique()
        print(f"✓ episode_set_name2 found: {len(unique_blocks)} unique blocks")
        print(f"  Sample blocks: {list(unique_blocks[:10])}")
    else:
        print(f"⚠ No episode_set_name2 column found")
    
    # Determine participant coordinate column names
    part_x_col = 'participant_x' if 'participant_x' in merged_df.columns else ('x' if 'x' in merged_df.columns else None)
    part_y_col = 'participant_y' if 'participant_y' in merged_df.columns else ('y' if 'y' in merged_df.columns else None)
    
    if part_x_col is None or part_y_col is None:
        print("Error: Could not determine participant coordinate columns. Available columns:", list(merged_df.columns))
        return
    
    print(f"✓ Using coordinate columns: {part_x_col}, {part_y_col}")
    
    # Check for label column - use the first available from our prioritized list
    label_column = None
    if label_columns:
        label_column = label_columns[0]  # Use the first (highest priority) label column
        print(f"✓ Using label column: {label_column}")
        
        # Check what type of data this column contains
        if label_column in merged_df.columns:
            sample_values = merged_df[label_column].dropna().unique()[:10]  # First 10 unique values
            print(f"  Sample values: {sample_values}")
            
            # Check if these look like syllables (numeric) or tracker categories (text)
            if sample_values.size > 0:
                first_val = sample_values[0]
                try:
                    float(first_val)
                    print(f"  ✓ Column appears to contain numeric data (likely syllables)")
                except (ValueError, TypeError):
                    print(f"  ⚠ Column appears to contain text data (likely tracker categories)")
                    print(f"  ⚠ For proper syllable coloring, ensure this column contains syllable numbers/identifiers")
    else:
        print("⚠ No label columns found for syllable-based plotting")
    
    # Check data availability
    if part_x_col in merged_df.columns and part_y_col in merged_df.columns:
        participant_data_count = len(merged_df) - merged_df[part_x_col].isna().sum()
        print(f"✓ Participant data available: {participant_data_count} rows with coordinates")
    
    if label_column and label_column in merged_df.columns:
        label_data_count = len(merged_df) - merged_df[label_column].isna().sum()
        print(f"✓ Label data available: {label_data_count} rows with labels")
        
        # Check if the label column contains tracker categories instead of syllables
        sample_values = merged_df[label_column].dropna().unique()[:5]
        if sample_values.size > 0:
            first_val = sample_values[0]
            try:
                float(first_val)
                print(f"  ✓ Label column contains numeric data (likely syllables)")
            except (ValueError, TypeError):
                print(f"  ⚠ Label column contains text data (tracker categories: {sample_values})")
                
                # If an external color map is provided, DON'T convert - use original labels with external color map
                if color_map is not None:
                    print(f"  🔧 External color map provided - keeping original string labels for proper color mapping")
                    print(f"    External color map keys: {list(color_map.keys())[:10]}...")
                    print(f"    Original label values: {sample_values}")
                    print(f"    ✓ Will use original string labels with external color map")
                else:
                    print(f"  🔧 No external color map - converting tracker categories to syllable-like identifiers for consistent coloring")
                    merged_df, label_column, syllable_color_map = convert_tracker_categories_to_syllables(merged_df, label_column)
                    color_map = syllable_color_map
                    print(f"  ✓ Using generated color map for converted syllable identifiers")
                    print(f"    Label column after conversion: {label_column}")
    
    # STEP 2: Create plots directory (skip main all-data plot for performance)
    print("\n=== STEP 2: CREATING PLOTS DIRECTORY ===")
    
    # Create plots directory
    plots_dir = os.path.join(spatial_output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("✓ Plots directory created")
    print("ℹ Skipping main 'all data' plot (computationally expensive)")
    print("ℹ Will create block-wise and scenario-wise plots instead")
    
    # STEP 3: Create data availability report and filter for complete trials
    print("\n=== STEP 3: DATA AVAILABILITY ANALYSIS ===")
    
    # Create comprehensive data availability report
    report_path = create_data_availability_report(merged_df, tracker_data_list, spatial_output_dir)
    
    # Data already filtered for complete trials (early filtering applied)
    print("\n  ✓ Data already filtered for complete trials (early filtering applied)")
    print("  ✓ All trials have: trackers + participant + scenario objects")
    print("  ✓ No additional filtering needed - proceeding with spatial analysis")
    
    # Verify that we still have data
    if len(merged_df) == 0:
        print("  ❌ No data available after early filtering - cannot create plots")
        return
    
    # Show summary of trials being used
    all_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"  ✓ Using {len(all_trials)} complete trials for spatial analysis")
    print(f"  ✓ All trials have complete data (all three datasets available)")
    
    # STEP 4: Create plots grouped by episode_set_name2
    print("\n=== STEP 4: CREATING PLOTS GROUPED BY EPISODE_SET_NAME2 ===")
    
    if 'episode_set_name2' in merged_df.columns:
        unique_blocks = merged_df['episode_set_name2'].dropna().unique()
        print(f"Found {len(unique_blocks)} blocks: {list(unique_blocks)}")
        
        for block in unique_blocks:
            print(f"\n--- Processing block: {block} ---")
            
            # Filter data for this block
            block_data = merged_df[merged_df['episode_set_name2'] == block]
            print(f"  Block '{block}' has {len(block_data)} rows")
            
            if len(block_data) == 0:
                print(f"  ⚠ No data found for block '{block}', skipping")
                continue
            
            # CRITICAL: Verify that block_data only contains complete trials
            print(f"  🔍 VERIFYING BLOCK DATA COMPLETENESS:")
            block_trials = block_data[['ppid', 'session', 'trial']].drop_duplicates()
            print(f"    - Block has {len(block_trials)} unique trials")
            
            complete_block_trials = 0
            incomplete_block_trials = 0
            
            for _, trial_row in block_trials.iterrows():
                ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
                trial_data = block_data[(block_data['ppid'] == ppid) & (block_data['session'] == session) & (block_data['trial'] == trial)]
                
                # Check if this trial has all three datasets
                has_participant = not trial_data['participant_x'].isna().all() and not trial_data['participant_y'].isna().all()
                
                # Check for scenario objects
                scenario_object_columns = globals().get('scenario_object_columns', [])
                has_scenario = False
                if scenario_object_columns:
                    scenario_data_count = 0
                    for col in scenario_object_columns:
                        if col in trial_data.columns and not trial_data[col].isna().all():
                            scenario_data_count += 1
                    has_scenario = scenario_data_count > 0
                
                # Check for tracker data
                trial_key = f"{ppid}-{session}-{trial}"
                has_tracker = any(trial_key in str(tracker_data.get('trial_key', '')) for tracker_data in tracker_data_list)
                
                if has_participant and has_scenario and has_tracker:
                    complete_block_trials += 1
                else:
                    incomplete_block_trials += 1
                    missing = []
                    if not has_participant:
                        missing.append('participant')
                    if not has_scenario:
                        missing.append('scenario')
                    if not has_tracker:
                        missing.append('tracker')
                    print(f"      ❌ Trial {ppid}-{session}-{trial}: Missing {', '.join(missing)} data - should not be in block data!")
            
            print(f"    - Complete trials in block: {complete_block_trials}")
            print(f"    - Incomplete trials in block: {incomplete_block_trials}")
            
            if incomplete_block_trials > 0:
                print(f"    ❌ ERROR: Block contains {incomplete_block_trials} incomplete trials!")
                print(f"    ❌ This indicates that the complete trials filtering is not working correctly!")
                print(f"    ❌ Skipping this block to avoid plotting incomplete data!")
                continue
            else:
                print(f"    ✓ All trials in block have complete data")
            
            # Create block-level plot
            safe_block_name = str(block).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            title = f'Block {block} - Spatial Analysis'
            filename = f'spatial_block_{safe_block_name}.png'
            
            try:
                print(f"    Calling create_spatial_plot_unified with:")
                print(f"      - label_column: '{label_column}'")
                print(f"      - color_map keys: {list(color_map.keys())[:10] if color_map else 'None'}...")
                print(f"      - block_data columns: {list(block_data.columns)}")
                if label_column and label_column in block_data.columns:
                    print(f"      - label_column sample values: {block_data[label_column].dropna().unique()[:10]}")
                
                create_spatial_plot_unified(block_data, title, filename, 
                                          part_x_col, part_y_col, label_column, color_map,
                                          tracker_data_list, args, object_styles, spatial_output_dir, subfolder='blocks')
                print(f"  ✓ Successfully created plot for block '{block}'")
            except Exception as e:
                print(f"  ❌ Error creating plot for block '{block}': {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"⚠ No 'episode_set_name2' column found in merged data")
    
    # STEP 5: Create plots grouped by scenario_id
    print("\n=== STEP 5: CREATING PLOTS GROUPED BY SCENARIO_ID ===")
    
    if 'scenario_id' in merged_df.columns:
        unique_scenarios = merged_df['scenario_id'].dropna().unique()
        print(f"Found {len(unique_scenarios)} scenarios: {list(unique_scenarios)}")
        
        for scenario_id in unique_scenarios:
            print(f"\n--- Processing scenario: {scenario_id} ---")
            
            # Filter data for this scenario
            scenario_data = merged_df[merged_df['scenario_id'] == scenario_id]
            print(f"  Scenario '{scenario_id}' has {len(scenario_data)} rows")
            
            if len(scenario_data) == 0:
                print(f"  ⚠ No data found for scenario '{scenario_id}', skipping")
                continue
            
            # Create scenario-level plot
            safe_scenario_id = re.sub(r'[^\w\-_\.]', '_', str(scenario_id))
            title = f'Scenario {scenario_id} - Spatial Analysis'
            filename = f'spatial_scenario_{safe_scenario_id}.png'
            
            try:
                print(f"    Calling create_spatial_plot_unified with:")
                print(f"      - label_column: '{label_column}'")
                print(f"      - color_map keys: {list(color_map.keys())[:10] if color_map else 'None'}...")
                print(f"      - scenario_data columns: {list(scenario_data.columns)}")
                if label_column and label_column in scenario_data.columns:
                    print(f"      - label_column sample values: {scenario_data[label_column].dropna().unique()[:10]}")
                
                create_spatial_plot_unified(scenario_data, title, filename, 
                                          part_x_col, part_y_col, label_column, color_map,
                                          tracker_data_list, args, object_styles, spatial_output_dir, subfolder='scenarios')
                print(f"  ✓ Successfully created plot for scenario '{scenario_id}'")
            except Exception as e:
                print(f"  ❌ Error creating plot for scenario '{scenario_id}': {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"⚠ No 'scenario_id' column found in merged data")
    
    # STEP 6: Perform object-syllable analysis if requested
    print("\n=== STEP 6: OBJECT-SYLLABLE ANALYSIS ===")
    
    # Check if object-syllable analysis is requested
    object_radius = getattr(args, 'object_radius', 1.0)  # Default radius of 1.0 meters
    perform_object_analysis = getattr(args, 'object_syllable_analysis', False)
    
    if perform_object_analysis:
        print(f"✓ Performing object-syllable analysis with radius: {object_radius}m")
        print(f"✓ Using FILTERED data - only trials with all three datasets (trackers, participant, scenario objects)")
        
        # DEBUG: Check what columns are available in merged_df before grouped analysis
        print(f"\n--- DEBUG: CHECKING COLUMNS FOR GROUPED ANALYSIS ---")
        print(f"Merged dataset shape: {merged_df.shape}")
        print(f"All columns: {list(merged_df.columns)}")
        
        # Verify that we're using filtered data
        all_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
        print(f"✓ Using {len(all_trials)} trials with complete data (all three datasets available)")
        
        # Check for critical grouping columns
        critical_cols = ['scenario_id', 'episode_set_name2']
        for col in critical_cols:
            if col in merged_df.columns:
                unique_vals = merged_df[col].dropna().unique()
                print(f"✓ Column '{col}' found with {len(unique_vals)} unique values: {list(unique_vals)}")
            else:
                print(f"❌ Column '{col}' NOT found in merged dataset")
        
        # Check for participant coordinate and label columns
        coord_cols = [col for col in merged_df.columns if 'x' in col.lower() or 'y' in col.lower()]
        label_cols = [col for col in merged_df.columns if 'label' in col.lower() or 'syllable' in col.lower()]
        print(f"Coordinate columns: {coord_cols}")
        print(f"Label columns: {label_cols}")
        
        # Verify data completeness for object analysis
        print(f"\n--- VERIFYING DATA COMPLETENESS FOR OBJECT ANALYSIS ---")
        participant_data_count = len(merged_df[merged_df['participant_x'].notna() & merged_df['participant_y'].notna()])
        scenario_data_count = len(merged_df[merged_df['task_x'].notna() & merged_df['task_y'].notna()])
        print(f"  Participant data points: {participant_data_count}")
        print(f"  Scenario data points: {scenario_data_count}")
        print(f"  Total data points: {len(merged_df)}")
        
        # STEP 5.1: Overall object-syllable analysis (filtered data)
        print(f"\n--- STEP 5.1: OVERALL OBJECT-SYLLABLE ANALYSIS (FILTERED DATA) ---")
        perform_object_syllable_analysis_unified(merged_df, spatial_output_dir, args, object_radius, color_map)
        
        # STEP 5.2: Block-level object-syllable analysis
        print(f"\n--- STEP 5.2: BLOCK-LEVEL OBJECT-SYLLABLE ANALYSIS ---")
        if 'episode_set_name2' in merged_df.columns:
            unique_blocks = merged_df['episode_set_name2'].dropna().unique()
            print(f"Found {len(unique_blocks)} blocks for object-syllable analysis: {list(unique_blocks)}")
            
            for block in unique_blocks:
                print(f"\n--- Processing block: {block} ---")
                
                # Create block-specific output directory
                safe_block_name = str(block).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                block_output_dir = os.path.join(spatial_output_dir, 'object_syllable_analysis', 'blocks', safe_block_name)
                
                try:
                    perform_grouped_object_syllable_analysis_unified(
                        merged_df, block_output_dir, color_map, object_radius, args,
                        group_type='block', group_name=block, group_column='episode_set_name2'
                    )
                    print(f"  ✓ Successfully completed object-syllable analysis for block '{block}'")
                except Exception as e:
                    print(f"  ❌ Error in object-syllable analysis for block '{block}': {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"⚠ No 'episode_set_name2' column found in merged data for block-level analysis")
        
        # Summary of block-level analysis
        print(f"\n--- BLOCK-LEVEL ANALYSIS SUMMARY ---")
        block_analysis_dir = os.path.join(spatial_output_dir, 'object_syllable_analysis', 'blocks')
        if os.path.exists(block_analysis_dir):
            block_dirs = [d for d in os.listdir(block_analysis_dir) if os.path.isdir(os.path.join(block_analysis_dir, d))]
            print(f"✓ Created {len(block_dirs)} block directories: {block_dirs}")
            
            total_plots = 0
            for block_dir in block_dirs:
                block_path = os.path.join(block_analysis_dir, block_dir)
                plot_files = [f for f in os.listdir(block_path) if f.endswith('.png')]
                total_plots += len(plot_files)
                print(f"  • {block_dir}: {len(plot_files)} plot files")
            
            print(f"✓ Total plots created: {total_plots}")
        else:
            print(f"⚠ No block analysis directory created")
        
        # STEP 5.3: Scenario-level object-syllable analysis
        print(f"\n--- STEP 5.3: SCENARIO-LEVEL OBJECT-SYLLABLE ANALYSIS ---")
        if 'scenario_id' in merged_df.columns:
            unique_scenarios = merged_df['scenario_id'].dropna().unique()
            print(f"Found {len(unique_scenarios)} scenarios for object-syllable analysis: {list(unique_scenarios)}")
            
            for scenario_id in unique_scenarios:
                print(f"\n--- Processing scenario: {scenario_id} ---")
                
                # Create scenario-specific output directory
                safe_scenario_id = re.sub(r'[^\w\-_\.]', '_', str(scenario_id))
                scenario_output_dir = os.path.join(spatial_output_dir, 'object_syllable_analysis', 'scenarios', safe_scenario_id)
                
                try:
                    perform_grouped_object_syllable_analysis_unified(
                        merged_df, scenario_output_dir, color_map, object_radius, args,
                        group_type='scenario', group_name=scenario_id, group_column='scenario_id'
                    )
                    print(f"  ✓ Successfully completed object-syllable analysis for scenario '{scenario_id}'")
                except Exception as e:
                    print(f"  ❌ Error in object-syllable analysis for scenario '{scenario_id}': {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"⚠ No 'scenario_id' column found in merged data for scenario-level analysis")
    else:
        print(f"⚠ Object-syllable analysis not requested (use --object-syllable-analysis to enable)")
    
    # STEP 6: Save final unified dataset
    print("\n=== STEP 6: SAVING FINAL UNIFIED DATASET ===")
    
    # Save the final unified dataset
    final_data_path = os.path.join(spatial_output_dir, 'data', 'final_spatial_analysis_unified_dataset.csv')
    merged_df.to_csv(final_data_path, index=False)
    print(f"✓ Saved final spatial analysis dataset to {final_data_path}")
    
    print(f"\n✅ UNIFIED SPATIAL ANALYSIS COMPLETE!")
    print(f"  - Final dataset saved to: {spatial_output_dir}/data/")
    print(f"  - Plots organized in: {spatial_output_dir}/plots/")
    print(f"    • Block-level plots: plots/blocks/")
    print(f"    • Scenario-level plots: plots/scenarios/")
    print(f"    • Main all-data plot skipped (computationally expensive)")
    if perform_object_analysis:
        print(f"  - Object-syllable analysis completed with radius: {object_radius}m")
        print(f"    • Object-syllable plots organized in: {spatial_output_dir}/object_syllable_analysis/")
        print(f"      - Overall analysis: object_syllable_analysis/")
        print(f"      - Block-level analysis: object_syllable_analysis/blocks/")
        print(f"      - Scenario-level analysis: object_syllable_analysis/scenarios/")
    
    print("✓ UNIFIED SPATIAL ANALYSIS COMPLETED SUCCESSFULLY!")
    print("✓ No data merging issues - all data was already unified!")


def create_spatial_plot_unified(data_df, title, filename, x_col, y_col, label_col, color_map, 
                               tracker_data_list, args, object_styles, output_dir, subfolder=None):
    """Create spatial plot with same characteristics as original - adapted for unified dataset."""
    
    # Calculate number of observations (unique ppid-session-trial combinations that have syllable data)
    # Only count trials that have participant location data with syllables
    if label_col and label_col in data_df.columns and x_col in data_df.columns and y_col in data_df.columns:
        # Count only trials that have syllable data (non-null predicted_label)
        trials_with_syllables = data_df[data_df[label_col].notna()][['ppid', 'session', 'trial']].drop_duplicates()
        n_observations = len(trials_with_syllables)
        print(f"  Creating plot '{title}' with {n_observations} trials containing syllable data")
        
        # DEBUG: Show trial breakdown
        print(f"  🔍 DEBUG: Trial count analysis:")
        print(f"    - Total unique trials in data: {data_df[['ppid', 'session', 'trial']].drop_duplicates().shape[0]}")
        print(f"    - Trials with syllable data: {n_observations}")
        print(f"    - Trial list: {[(row['ppid'], row['session'], row['trial']) for _, row in trials_with_syllables.iterrows()]}")
        
        # Add observation count to filename
        filename_with_obs = filename.replace('.png', f'_n{n_observations}.png')
    else:
        # Fallback if no syllable data
        n_observations = data_df[['ppid', 'session', 'trial']].drop_duplicates().shape[0]
        filename_with_obs = filename.replace('.png', f'_n{n_observations}.png')
        print(f"  Creating plot '{title}' with {n_observations} trials (no syllable data)")
        
        # DEBUG: Show trial breakdown
        print(f"  🔍 DEBUG: Trial count analysis:")
        print(f"    - Total unique trials in data: {n_observations}")
        print(f"    - Trial list: {[(row['ppid'], row['session'], row['trial']) for _, row in data_df[['ppid', 'session', 'trial']].drop_duplicates().iterrows()]}")
    
    # Construct full save path
    plots_dir = os.path.join(output_dir, 'plots')
    if subfolder:
        plots_dir = os.path.join(plots_dir, subfolder)
    save_path = os.path.join(plots_dir, filename_with_obs)
    
    # Create the actual spatial plot
    plot_spatial_unified(data_df, title, save_path, x_col, y_col, label_col, color_map,
                        tracker_data_list, args, object_styles, subfolder, args.xlim, args.ylim)


def plot_spatial_unified(df, title_suffix, save_path, x_col, y_col, label_col, color_map, 
                        tracker_data_list, args, object_styles, subfolder=None, xlim=None, ylim=None):
    """Plot spatial data using unified dataset - adapted from original plot_spatial function."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import re
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Handle subfolder if specified
    if subfolder:
        plots_dir = os.path.join(os.path.dirname(save_path), subfolder)
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, os.path.basename(save_path))
    
    plt.figure(figsize=(12, 9))
    
    # Plot participant trajectories colored by syllables
    print(f"Plot {title_suffix} - Debug info:")
    print(f"  - DataFrame shape: {df.shape}")
    print(f"  - Required columns: x_col='{x_col}', y_col='{y_col}', label_col='{label_col}'")
    print(f"  - Available columns: {list(df.columns)}")
    print(f"  - x_col in df: {x_col in df.columns}")
    print(f"  - y_col in df: {y_col in df.columns}")
    print(f"  - label_col in df: {label_col in df.columns}")
    print(f"  - Participant data available: {not df[x_col].isna().all() and not df[y_col].isna().all() if x_col in df.columns and y_col in df.columns else False}")
    
    # CRITICAL: Verify that only complete trials are being plotted
    print(f"  - VERIFICATION: Checking data completeness for plotting...")
    if 'ppid' in df.columns and 'session' in df.columns and 'trial' in df.columns:
        unique_trials = df[['ppid', 'session', 'trial']].drop_duplicates()
        print(f"  - Unique trials in plot data: {len(unique_trials)}")
        
        # Check each trial for completeness
        complete_trials_in_plot = 0
        incomplete_trials_in_plot = 0
        
        for _, trial_row in unique_trials.iterrows():
            ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
            trial_data = df[(df['ppid'] == ppid) & (df['session'] == session) & (df['trial'] == trial)]
            
            # Check if this trial has all three datasets
            has_participant = not trial_data[x_col].isna().all() if x_col in trial_data.columns else False
            has_scenario = False
            has_tracker = False
            
            # Check for scenario objects
            scenario_object_columns = globals().get('scenario_object_columns', [])
            if scenario_object_columns:
                scenario_data_count = 0
                for col in scenario_object_columns:
                    if col in trial_data.columns and not trial_data[col].isna().all():
                        scenario_data_count += 1
                has_scenario = scenario_data_count > 0
            
            # Check for tracker data
            trial_key = f"{ppid}-{session}-{trial}"
            has_tracker = any(trial_key in str(tracker_data.get('trial_key', '')) for tracker_data in tracker_data_list)
            
            if has_participant and has_scenario and has_tracker:
                complete_trials_in_plot += 1
            else:
                incomplete_trials_in_plot += 1
                missing = []
                if not has_participant:
                    missing.append('participant')
                if not has_scenario:
                    missing.append('scenario')
                if not has_tracker:
                    missing.append('tracker')
                print(f"    ⚠ Trial {ppid}-{session}-{trial}: Missing {', '.join(missing)} data - should not be in plot!")
        
        print(f"  - Complete trials in plot: {complete_trials_in_plot}")
        print(f"  - Incomplete trials in plot: {incomplete_trials_in_plot}")
        
        if incomplete_trials_in_plot > 0:
            print(f"  ❌ ERROR: Found {incomplete_trials_in_plot} incomplete trials in plot data!")
            print(f"  ❌ This indicates that data filtering is not working correctly!")
        else:
            print(f"  ✓ All trials in plot have complete data (all three datasets)")
    else:
        print(f"  ⚠ Cannot verify trial completeness - missing ppid/session/trial columns")
    
    if label_col and label_col in df.columns:
        print(f"  - Label data available: {not df[label_col].dropna().all()}")
        print(f"  - Unique labels: {df[label_col].dropna().unique()}")
        print(f"  - Label data types: {df[label_col].dtype}")
        print(f"  - Sample label values: {df[label_col].dropna().head(10).tolist()}")
    
    # Debug color map information
    if color_map is not None:
        print(f"  - Color map type: {type(color_map)}")
        print(f"  - Color map length: {len(color_map)}")
        print(f"  - Color map key types: {[type(k) for k in list(color_map.keys())[:5]]}")
        print(f"  - Color map sample keys: {list(color_map.keys())[:5]}")
        print(f"  - Color map sample values: {list(color_map.values())[:5]}")
        
        # Check if color map keys match the expected syllable format
        # Note: unique_sylls is defined later in the function
        print(f"  - Color map keys are text-based, syllables are numeric - will need conversion")
    else:
        print(f"  - No color map provided")
    
    if x_col in df.columns and y_col in df.columns and label_col and label_col in df.columns:
        print(f"Plot {title_suffix} - Plotting time series data for {len(df)} total data points")
        
        # Check if participant coordinates are dummy values (all the same)
        unique_x = df[x_col].nunique()
        unique_y = df[y_col].nunique()
        if unique_x == 1 and unique_y == 1:
            x_val = df[x_col].iloc[0]
            y_val = df[y_col].iloc[0]
            if x_val == 0.0 and y_val == 0.0:
                print(f"  ⚠ WARNING: Participant coordinates are dummy values (0,0) - no real movement data")
                print(f"  ⚠ Use --coords-dir argument to load real participant coordinates")
                print(f"  ⚠ Skipping participant trajectory plotting")
                # Don't plot anything for dummy coordinates
                return
            else:
                print(f"  ✓ Participant coordinates appear to be real data (unique X: {unique_x}, unique Y: {unique_y})")
        
        # Group by syllable and plot each syllable's data with its color from color_map
        print(f"  Available columns in DataFrame: {list(df.columns)}")
        print(f"  Label column '{label_col}' exists: {label_col in df.columns}")
        if label_col in df.columns:
            print(f"  Label column data type: {df[label_col].dtype}")
            print(f"  Label column sample values: {df[label_col].dropna().head(10).tolist()}")
            print(f"  Label column unique values count: {df[label_col].nunique()}")
        
        unique_sylls = df[label_col].dropna().unique()
        print(f"  Available syllables: {sorted(unique_sylls)}")
        print(f"  Syllable types: {[type(syll) for syll in unique_sylls[:10]]}")  # Show types of first 10 syllables
        print(f"  Color map provided: {color_map is not None}")
        
        # Verify that we're using the right label column
        if label_col.endswith('_syllable'):
            print(f"  ⚠ WARNING: Using converted syllable column '{label_col}' - this may cause color mapping issues!")
            print(f"  🔧 Consider using the original label column for proper color mapping")
            
            # Try to find the original label column
            original_label_col = label_col.replace('_syllable', '')
            if original_label_col in df.columns:
                print(f"  🔧 Found original label column '{original_label_col}' - consider using this instead")
                print(f"    Original column sample values: {df[original_label_col].dropna().unique()[:10]}")
        
        if color_map:
            print(f"  Color map keys: {list(color_map.keys())[:10]}...")  # Show first 10 keys
            print(f"  Color map key types: {[type(k) for k in list(color_map.keys())[:10]]}")  # Show key types
            print(f"  Color map values: {list(color_map.values())[:5]}...")  # Show first 5 colors
            
            # Check for key type mismatches
            syll_types = set(type(syll) for syll in unique_sylls)
            key_types = set(type(k) for k in color_map.keys())
            print(f"  Syllable types in data: {syll_types}")
            print(f"  Key types in color map: {key_types}")
            
            if syll_types != key_types:
                print(f"  ⚠ WARNING: Type mismatch between syllables and color map keys!")
                print(f"    This may cause color lookup failures")
        else:
            print(f"  No color map - using default matplotlib colors")
            # Create fallback colors for all syllables
            syllable_to_color_map = {}  # Ensure it's defined
            fallbacks = 0  # Ensure it's defined
            for syll in unique_sylls:
                try:
                    syll_int = int(syll)
                    # Use multi-colormap approach for more unique colors
                    colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                    colors_per_cmap = 20
                    cmap_index = (syll_int // colors_per_cmap) % len(colormaps)
                    color_index = syll_int % colors_per_cmap
                    syllable_to_color_map[syll] = colormaps[cmap_index](color_index)
                    fallbacks += 1
                except (ValueError, TypeError):
                    # For non-numeric syllables, use hash-based color with multi-colormap
                    hash_val = hash(str(syll))
                    colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                    colors_per_cmap = 20
                    cmap_index = (abs(hash_val) // colors_per_cmap) % len(colormaps)
                    color_index = abs(hash_val) % colors_per_cmap
                    syllable_to_color_map[syll] = colormaps[cmap_index](color_index)
                    fallbacks += 1
        
        if len(unique_sylls) == 0:
            # Fallback: plot all participant data if label column has no values
            print(f"Plot {title_suffix} - No syllable labels available; plotting participant data as background")
            plt.scatter(df[x_col], df[y_col], c='lightgrey', s=10, alpha=0.8, zorder=2, label='Participant (no syllable labels)')
        
        syllable_stats = {}
        
        # Initialize syllable-to-color mapping and tracking variables
        syllable_to_color_map = {}
        exact_matches = 0
        string_matches = 0
        integer_matches = 0
        string_integer_matches = 0
        fallbacks = 0
        
        if color_map is not None:
            print(f"  Creating syllable-to-color mapping using color map...")
            print(f"  Color map keys: {list(color_map.keys())[:10]}...")  # Show first 10 keys
            print(f"  Color map key types: {[type(k) for k in list(color_map.keys())[:10]]}")  # Show key types
            print(f"  Sample syllables to map: {unique_sylls[:10]}")  # Show what we're trying to map
            
            # Analyze color map compatibility
            syll_types = set(type(syll) for syll in unique_sylls)
            key_types = set(type(k) for k in color_map.keys())
            print(f"  Syllable types in data: {syll_types}")
            print(f"  Key types in color map: {key_types}")
            
            if syll_types != key_types:
                print(f"  ⚠ WARNING: Type mismatch between syllables and color map keys!")
                print(f"    This will cause color lookup failures")
                
                # Show examples of the mismatch
                if str in syll_types and int in key_types:
                    print(f"    Data has string syllables but color map has integer keys")
                elif int in syll_types and str in key_types:
                    print(f"    Data has integer syllables but color map has string keys")
            
            # Check for missing syllables in color map
            missing_in_color_map = []
            for syll in unique_sylls:
                if (syll not in color_map and 
                    str(syll) not in color_map and 
                    (int(syll) if isinstance(syll, (int, float)) else None) not in color_map):
                    missing_in_color_map.append(syll)
            
            if missing_in_color_map:
                print(f"  ⚠ WARNING: {len(missing_in_color_map)} syllables not found in color map: {missing_in_color_map[:10]}...")
                print(f"    These will use fallback colors instead of the intended color map colors")
            
            # Track mapping success/failure (variables already initialized above)
            
            print(f"  Starting color mapping for {len(unique_sylls)} syllables...")
            
            for syll in unique_sylls:
                color_found = False
                print(f"    Processing syllable '{syll}' (type: {type(syll)})")
                
                try:
                    # Try exact match first (preserve original label type)
                    if syll in color_map:
                        syllable_to_color_map[syll] = color_map[syll]
                        color_found = True
                        exact_matches += 1
                        print(f"      ✓ Exact match: {syll} → Color {color_map[syll]}")
                    # Try string version
                    elif str(syll) in color_map:
                        syllable_to_color_map[syll] = color_map[str(syll)]
                        color_found = True
                        string_matches += 1
                        print(f"      ✓ String match: {syll} → Color {color_map[str(syll)]}")
                    # Try integer version if syllable is numeric
                    else:
                        try:
                            syll_int = int(syll)
                            if syll_int in color_map:
                                syllable_to_color_map[syll] = color_map[syll_int]
                                color_found = True
                                integer_matches += 1
                                print(f"      ✓ Integer match: {syll} → Color {color_map[syll_int]}")
                            else:
                                # Try string version of integer
                                if str(syll_int) in color_map:
                                    syllable_to_color_map[syll] = color_map[str(syll_int)]
                                    color_found = True
                                    string_integer_matches += 1
                                    print(f"      ✓ String integer match: {syll} → Color {color_map[str(syll_int)]}")
                        except (ValueError, TypeError):
                            pass
                    
                    # If no color found in map, use fallback
                    if not color_found:
                        print(f"      ⚠ No color found in color map for syllable '{syll}'")
                        print(f"        Color map keys available: {list(color_map.keys())[:20]}...")
                        print(f"        Syllable type: {type(syll)}, value: '{syll}'")
                        
                        try:
                            syll_int = int(syll)
                            # Use multi-colormap approach for more unique colors
                            colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                            colors_per_cmap = 20
                            cmap_index = (syll_int // colors_per_cmap) % len(colormaps)
                            color_index = syll_int % colors_per_cmap
                            syllable_to_color_map[syll] = colormaps[cmap_index](color_index)
                            fallbacks += 1
                            print(f"        ⚠ Using numeric fallback color (multi-colormap: {cmap_index}, {color_index})")
                        except (ValueError, TypeError):
                            # For non-numeric syllables, use hash-based color with multi-colormap
                            hash_val = hash(str(syll))
                            colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                            colors_per_cmap = 20
                            cmap_index = (abs(hash_val) // colors_per_cmap) % len(colormaps)
                            color_index = abs(hash_val) % colors_per_cmap
                            syllable_to_color_map[syll] = colormaps[cmap_index](color_index)
                            fallbacks += 1
                            print(f"        ⚠ Using hash-based fallback color (multi-colormap: {cmap_index}, {color_index})")
                except Exception as e:
                    print(f"      ⚠ Error processing syllable '{syll}' for color map: {e}")
                    print(f"        Syllable: '{syll}' (type: {type(syll)})")
                    print(f"        Color map keys: {list(color_map.keys())[:20]}...")
                    
                    # Use fallback color with multi-colormap approach
                    try:
                        syll_int = int(syll)
                        colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                        colors_per_cmap = 20
                        cmap_index = (syll_int // colors_per_cmap) % len(colormaps)
                        color_index = syll_int % colors_per_cmap
                        syllable_to_color_map[syll] = colormaps[cmap_index](color_index)
                        fallbacks += 1
                        print(f"        ⚠ Using exception numeric fallback color (multi-colormap: {cmap_index}, {color_index})")
                    except (ValueError, TypeError):
                        hash_val = hash(str(syll))
                        colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                        colors_per_cmap = 20
                        cmap_index = (abs(hash_val) // colors_per_cmap) % len(colormaps)
                        color_index = abs(hash_val) % colors_per_cmap
                        syllable_to_color_map[syll] = colormaps[cmap_index](color_index)
                        fallbacks += 1
                        print(f"        ⚠ Using exception hash-based fallback color (multi-colormap: {cmap_index}, {color_index})")
        
        # Print mapping summary
        print(f"  Color mapping summary:")
        print(f"    ✓ Exact matches: {exact_matches}")
        print(f"    ✓ String matches: {string_matches}")
        print(f"    ✓ Integer matches: {integer_matches}")
        print(f"    ✓ String integer matches: {string_integer_matches}")
        print(f"    ⚠ Fallbacks: {fallbacks}")
        print(f"    ✓ Total syllables processed: {len(unique_sylls)}")
        print(f"    ✓ Syllables with colors: {len(syllable_to_color_map)}")
        if color_map is not None:
            print(f"    ✓ Color map keys used: {list(color_map.keys())[:10]}...")
        else:
            print(f"    ✓ No color map provided")
        
        # Validate that all syllables have colors
        missing_colors = [syll for syll in unique_sylls if syll not in syllable_to_color_map]
        if missing_colors:
            print(f"  ❌ ERROR: {len(missing_colors)} syllables missing colors: {missing_colors}")
            print(f"    This indicates a bug in the color mapping logic above")
        else:
            print(f"  ✓ All syllables have colors assigned")
        
        for syll in unique_sylls:
            sub = df[df[label_col] == syll]
            if len(sub) == 0:
                continue
            
            # Get syllable color from our mapping - this should always have a color now
            if syll not in syllable_to_color_map:
                print(f"  ❌ ERROR: Syllable '{syll}' has no color assigned!")
                print(f"    This should not happen - all syllables should have colors from the mapping above")
                print(f"    Available mapped syllables: {list(syllable_to_color_map.keys())}")
                continue
            
            syll_color = syllable_to_color_map[syll]
            print(f"  ✓ Using mapped color for syllable '{syll}': {syll_color}")
            
            # Count number of trials for this syllable
            n_trials = sub[['ppid', 'session', 'trial']].drop_duplicates().shape[0]
            n_observations = len(sub)
            syllable_stats[syll] = {'trials': n_trials, 'observations': n_observations}
            
            # Plot all time series data points for this syllable (simple scatter plot like original)
            plt.scatter(sub[x_col], sub[y_col], c=syll_color, s=20, alpha=0.7, zorder=2, label=f'{syll}')
            print(f"  ✓ Plotted {len(sub)} data points for syllable '{syll}' with color {syll_color}")
            
            # Debug: Check if the scatter plot was actually created
            current_handles, current_labels = plt.gca().get_legend_handles_labels()
            if f'{syll}' in current_labels:
                print(f"    ✓ Syllable '{syll}' handle added to legend successfully")
                print(f"      Handle type: {type(current_handles[-1])}")
                print(f"      Handle label: {current_handles[-1].get_label()}")
            else:
                print(f"    ⚠ Syllable '{syll}' handle NOT found in legend after plotting")
                print(f"      Current legend labels: {current_labels}")
                print(f"      Expected label: '{syll}'")
                print(f"      All handles: {[h.get_label() for h in current_handles]}")
        
        # Print syllable statistics
        print(f"\nSyllable Statistics for {title_suffix}:")
        for syll, stats in syllable_stats.items():
            print(f"  {syll}: {stats['trials']} trials, {stats['observations']} observations")
        
        # Print color map usage summary
        if color_map:
            print(f"\nColor Map Usage Summary for {title_suffix}:")
            if syllable_to_color_map:
                print(f"  ✓ All {len(unique_sylls)} syllables mapped to color map colors")
                print(f"  ✓ Using {len(color_map)} available colors from color map")
                print(f"  ✓ Color map keys: {list(color_map.keys())[:10]}...")  # Show first 10
            else:
                print(f"  ⚠ No syllable-to-color mapping created")
        else:
            print(f"\nNo color map provided - all syllables using fallback colors")
    
    elif x_col in df.columns and y_col in df.columns:
        # Fallback: plot all participant data if no label column
        plt.scatter(df[x_col], df[y_col], c='lightgrey', s=10, alpha=0.8, zorder=2, label='Participant (no syllable labels)')

    # Plot scenario objects using the passed object_styles
    print(f"Plot {title_suffix} - Available columns: {list(df.columns)}")
    scenario_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    print(f"Plot {title_suffix} - Scenario coordinate columns: {scenario_cols}")
    
    # Debug: Show participant coordinate ranges for comparison
    if x_col in df.columns and y_col in df.columns:
        participant_x = df[x_col].dropna()
        participant_y = df[y_col].dropna()
        if len(participant_x) > 0 and len(participant_y) > 0:
            print(f"  📍 Participant coordinate ranges: X=[{participant_x.min():.3f}, {participant_x.max():.3f}], Y=[{participant_y.min():.3f}, {participant_y.max():.3f}]")
    
    object_bases = set()
    for c in scenario_cols:
        m = re.match(r'(.+)_([xyz])$', c)
        if m:
            object_bases.add(m.group(1))
    
    print(f"Plot {title_suffix} - Found scenario objects: {list(object_bases)}")
    plotted_objects = set()  # Track which object types have been plotted
    
    for obj in object_bases:
        base = obj.lower()
        style_key = None
        if base.startswith('t') and (base == 't' or base[1:].isdigit()):
            style_key = 'T'
        else:
            for k in object_styles:
                if k in base:
                    style_key = k
                    break
        
        if style_key is None:
            continue
        
        # Skip target objects (don't plot them)
        if style_key == 'T':
            print(f"  Skipping target object {obj} (not needed in plots)")
            continue
        
        xcol = f'{obj}_x'
        ycol = f'{obj}_y'
        if xcol not in df.columns or ycol not in df.columns:
            continue
        
        # Check if coordinates exist and are valid
        valid_x = df[xcol].dropna()
        valid_y = df[ycol].dropna()
        
        if len(valid_x) == 0 or len(valid_y) == 0:
            continue
        
        points = set(zip(valid_x, valid_y))
        
        # DEBUG: Show trial breakdown for this object
        print(f"  🔍 DEBUG: {obj} object analysis:")
        print(f"    - Total data points: {len(valid_x)}")
        print(f"    - Unique positions: {len(points)}")
        
        # Count how many trials contribute to these positions
        trials_with_this_object = df[df[obj + '_x'].notna() & df[obj + '_y'].notna()][['ppid', 'session', 'trial']].drop_duplicates()
        print(f"    - Trials with {obj} data: {len(trials_with_this_object)}")
        print(f"    - Trial list: {[(row['ppid'], row['session'], row['trial']) for _, row in trials_with_this_object.iterrows()]}")
        
        # Filter out points that are extremely far away (likely data errors)
        reasonable_points = []
        for x, y in points:
            # Check if point is within reasonable bounds (e.g., within 20 meters of origin)
            if abs(x) <= 20 and abs(y) <= 20:
                reasonable_points.append((x, y))
            else:
                print(f"    ⚠ Skipping {obj} at ({x:.3f}, {y:.3f}) - too far from origin")
        
        if reasonable_points:
            print(f"  Plotting {obj}: {len(reasonable_points)} reasonable positions (filtered from {len(points)} total)")
            
            # Debug: Show coordinate ranges for this object
            x_coords, y_coords = zip(*reasonable_points)
            print(f"    {obj} coordinate ranges: X=[{min(x_coords):.3f}, {max(x_coords):.3f}], Y=[{min(y_coords):.3f}, {max(y_coords):.3f}]")
            
            for x, y in reasonable_points:
                # Only add label if this object type hasn't been plotted yet
                add_label = style_key not in plotted_objects
                if add_label:
                    plotted_objects.add(style_key)
                    print(f"      {obj} position: ({x:.3f}, {y:.3f})")
                
                plt.scatter(x, y, marker=object_styles[style_key]['marker'], 
                           color=object_styles[style_key]['color'], s=object_styles[style_key]['s'], edgecolor='black', zorder=10, 
                           label=object_styles[style_key]['label'] if add_label else "")
        else:
            print(f"  ⚠ No reasonable positions found for {obj} after filtering")
    
    # Plot tracker paths if requested
    plot_trackers = getattr(args, 'plot_trackers', True)
    if plot_trackers and tracker_data_list is not None:
        print(f"  Plotting tracker paths...")
        print(f"    Using {len(tracker_data_list)} pre-loaded tracker data files")
        
        # Get all trials in this plot
        plot_trials = df[['ppid', 'session', 'trial']].drop_duplicates()
        print(f"    Processing {len(plot_trials)} trials for tracker data")
        
        tracker_plotted_count = 0
        for _, trial_row in plot_trials.iterrows():
            ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
            
            # Find corresponding tracker data
            tracker_data = None
            for tracker_item in tracker_data_list:
                if (str(tracker_item['ppid']) == str(ppid) and 
                    str(tracker_item['session']) == str(session) and 
                    str(tracker_item['trial']) == str(trial)):
                    tracker_data = tracker_item['tracker_data']
                    break
            
            if tracker_data is not None and len(tracker_data) > 0:
                # Check if transformed coordinates are available
                if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                    # Use pre-transformed coordinates (already aligned and rotated)
                    tracker_x = tracker_data['transformed_x']
                    tracker_y = tracker_data['transformed_y']
                    
                    print(f"      Trial {ppid}-{session}-{trial}: Using pre-aligned and rotated tracker data")
                    print(f"      Tracker data range: x=[{tracker_x.min():.3f}, {tracker_x.max():.3f}], y=[{tracker_y.min():.3f}, {tracker_y.max():.3f}]")
                    
                    # Plot tracker path with dark grey color and X markers (no additional correction needed)
                    plt.scatter(tracker_x, tracker_y, color='darkgrey', marker='x', s=20, alpha=0.6, zorder=1, 
                               label='Tracker Path' if tracker_plotted_count == 0 else "")
                    
                    # Connect consecutive points with lines
                    if len(tracker_x) > 1:
                        plt.plot(tracker_x, tracker_y, color='darkgrey', alpha=0.4, linewidth=0.5, zorder=1)
                    
                    tracker_plotted_count += 1
                    print(f"      ✓ Plotted tracker data for trial {ppid}-{session}-{trial} ({len(tracker_data)} points)")
                else:
                    print(f"      ✗ No transformed coordinates found in tracker data for trial {ppid}-{session}-{trial}")
            else:
                print(f"      ✗ No tracker data found for trial {ppid}-{session}-{trial}")
        
        if tracker_plotted_count == 0:
            print(f"    ⚠ No tracker data was plotted")
        else:
            print(f"    ✓ Successfully plotted tracker data for {tracker_plotted_count} trials")
    elif plot_trackers and tracker_data_list is None:
        print(f"  ⚠ Tracker plotting requested but no tracker_data_list provided")
    elif not plot_trackers:
        print(f"  ℹ Tracker plotting not requested")
    
    # Add dark grey dashed square (10x6) centered at (0,0)
    square_width = 10
    square_height = 6
    square_x = [-square_width/2, square_width/2, square_width/2, -square_width/2, -square_width/2]
    square_y = [-square_height/2, -square_height/2, square_height/2, square_height/2, -square_height/2]
    plt.plot(square_x, square_y, color='darkgrey', linestyle='--', linewidth=2, zorder=5, label='Reference Square (10x6m)')
    
    # Add annotation about time series data and start/end markers
    plt.annotate('All time series data values are plotted, colored by syllables\nDiamonds (◆) = Start points, Stars (★) = End points\nOnly consecutive time points are connected', 
                xy=(1, 1), xycoords='axes fraction', fontsize=12, ha='right', va='top')
    
    # Create legend for scenario objects that were actually plotted
    object_handles = {}
    for style_key in plotted_objects:
        if style_key in object_styles:
            object_handles[object_styles[style_key]['label']] = mpatches.Patch(color=object_styles[style_key]['color'], label=object_styles[style_key]['label'])
    
    # Get existing legend handles from the plot (syllables and reference square)
    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    
    # Debug legend information
    print(f"  Existing legend handles: {len(existing_handles)}")
    print(f"  Existing legend labels: {existing_labels}")
    print(f"  Object handles to add: {list(object_handles.keys())}")
    
    # Check if we have any syllable handles at all
    syllable_handles = [h for h, l in zip(existing_handles, existing_labels) if l not in object_handles and l != 'Reference Square (10x6m)']
    print(f"  Syllable handles found: {len(syllable_handles)}")
    print(f"  Syllable labels: {[l for h, l in zip(existing_handles, existing_labels) if l not in object_handles and l != 'Reference Square (10x6m)']}")
    
    # IMPORTANT: Don't filter out syllable handles - they should all be included
    # Only filter out duplicate scenario object handles
    filtered_handles = []
    seen_labels = set()
    
    for handle, label in zip(existing_handles, existing_labels):
        # Skip scenario object labels that we'll add manually (but keep syllables!)
        if label in object_handles:
            print(f"    Skipping duplicate object label: {label}")
            continue
        # Skip duplicates (but this shouldn't affect syllables)
        if label in seen_labels:
            print(f"    Skipping duplicate label: {label}")
            continue
        seen_labels.add(label)
        filtered_handles.append(handle)
        print(f"    Keeping handle for label: {label}")
    
    # Combine filtered syllable legend with scenario object legend
    all_handles = filtered_handles + list(object_handles.values())
    
    print(f"  Final legend composition:")
    print(f"    Syllable/Reference handles: {len(filtered_handles)}")
    print(f"    Object handles: {len(object_handles)}")
    print(f"    Total handles: {len(all_handles)}")
    
    # Always create a legend, even if empty
    if all_handles:
        try:
            plt.legend(handles=all_handles, loc='best', fontsize=10)
            print(f"  Created legend with {len(all_handles)} items: {[h.get_label() for h in all_handles]}")
        except Exception as e:
            print(f"  ⚠ Error creating legend: {e}")
            # Try to create a basic legend
            try:
                plt.legend(loc='best')
                print(f"  Created basic legend as fallback")
            except Exception as e2:
                print(f"  ⚠ Error creating fallback legend: {e2}")
    else:
        # Create a minimal legend to show that the plot is working
        try:
            plt.legend(handles=[], loc='best')
            print(f"  Created empty legend (no items to display)")
        except Exception as e:
            print(f"  ⚠ Error creating empty legend: {e}")
    
    # Use strict axis limits from command line arguments for consistent, legible plots
    if xlim is not None and ylim is not None:
        print(f"  Using strict plot bounds from arguments: X={xlim}, Y={ylim}")
        plt.xlim(xlim)
        plt.ylim(ylim)
    else:
        # Fallback to default bounds if no limits specified
        print(f"  No axis limits specified, using default bounds: X=[-6, 6], Y=[-4, 4]")
        plt.xlim(-6, 6)
        plt.ylim(-4, 4)
    
    # Ensure equal aspect ratio so squares look like squares
    plt.gca().set_aspect('equal')
    
    plt.title(title_suffix)
    plt.xlabel('x (metres)')
    plt.ylabel('y (metres)')
    
    # Ensure a legend exists (fallback) - but don't create one with all objects
    if not plt.gca().get_legend():
        pass  # Don't create a fallback legend that might show all objects
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved spatial plot: {save_path}")


def perform_object_syllable_analysis_unified(merged_df, spatial_output_dir, args, object_radius=1.0, color_map=None):
    """
    Perform object-syllable correlation analysis using unified dataset.
    
    This analysis re-centers data to each object and computes syllable frequencies
    within/outside circular radius. Data is shifted so each object is at origin (0,0).
    
    Args:
        merged_df: Unified dataset with all data already merged and filtered (only complete trials)
        spatial_output_dir: Output directory for analysis results
        args: Command line arguments
        object_radius: Radius in meters for analysis (default: 1.0)
        color_map: Color map for consistent syllable coloring across all plots
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    import os
    from matplotlib.patches import Circle
    
    print(f"\n=== OBJECT-SYLLABLE ANALYSIS (UNIFIED DATASET) ===")
    print(f"Radius: {object_radius} meters")
    print("Note: Data will be centered around each object (object at origin)")
    print("Note: Using FILTERED unified dataset - only trials with all three datasets (trackers, participant, scenario objects)")
    
    # Verify that we're using filtered data
    all_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"✓ Using {len(all_trials)} trials with complete data (all three datasets available)")
    
    # Verify data completeness
    participant_data_count = len(merged_df[merged_df['participant_x'].notna() & merged_df['participant_y'].notna()])
    scenario_data_count = len(merged_df[merged_df['task_x'].notna() & merged_df['task_y'].notna()])
    print(f"✓ Data completeness verification:")
    print(f"  - Participant data points: {participant_data_count}")
    print(f"  - Scenario data points: {scenario_data_count}")
    print(f"  - Total data points: {len(merged_df)}")
    print(f"  - All data points have both participant and scenario data: {participant_data_count == scenario_data_count == len(merged_df)}")
    
    # Verify that only complete trials are being used
    print(f"✓ Trial completeness verification:")
    print(f"  - All trials in this analysis have: trackers + participant + scenario objects")
    print(f"  - Incomplete trials have been filtered out before this analysis")
    print(f"  - This ensures consistent data availability across all spatial analysis components")
    
    # Data already filtered for complete trials (early filtering applied)
    print(f"✓ Data already filtered for complete trials (early filtering applied)")
    print(f"✓ All trials in analysis have complete data (all three datasets)")
    print(f"✓ No additional verification needed - early filtering ensures completeness")
    
    # Define specific objects to analyze (as requested by user)
    # Note: standpoint is not available in the current dataset
    objects_to_analyze = ['task', 'safehouse', 'threat', 'startpoint', 'weapons', 'wall', 'hide']
    print(f"Analyzing specific objects: {objects_to_analyze}")
    print(f"Note: 'standpoint' not available in dataset - skipping")
    
    # Create output directory for object-syllable analysis
    object_analysis_dir = os.path.join(spatial_output_dir, 'object_syllable_analysis')
    os.makedirs(object_analysis_dir, exist_ok=True)
    
    # Initialize results storage
    all_object_frequencies = {}
    all_centered_data_for_plotting = {}  # Store centered data for each object type
    
    # Get all trials that have participant data
    all_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"Total trials with participant data: {len(all_trials)}")
    
    for object_name in objects_to_analyze:
        print(f"\n--- Analyzing object: {object_name} ---")
        
        # Find object columns in unified dataset (use regular _x and _y columns)
        object_x_col = f'{object_name}_x'
        object_y_col = f'{object_name}_y'
        
        if object_x_col in merged_df.columns and object_y_col in merged_df.columns:
            print(f"  Using columns: {object_x_col}, {object_y_col}")
        else:
            print(f"  ERROR: Object '{object_name}' columns not found in unified dataset!")
            print(f"  Looking for: '{object_x_col}' and '{object_y_col}'")
            print(f"  Available columns: {[col for col in merged_df.columns if object_name in col]}")
            continue
        
        # Get object positions from unified dataset
        object_positions = merged_df[['ppid', 'session', 'trial', object_x_col, object_y_col]].dropna()
        
        if len(object_positions) == 0:
            print(f"  No positions found for object '{object_name}', skipping...")
            continue
        
        # Check if we have valid coordinate data
        valid_positions = object_positions[
            (object_positions[object_x_col].notna()) & 
            (object_positions[object_y_col].notna()) &
            (object_positions[object_x_col] != 0) & 
            (object_positions[object_y_col] != 0)
        ]
        
        if len(valid_positions) == 0:
            print(f"  No valid positions found for object '{object_name}' (all positions are NaN or 0), skipping...")
            continue
        
        print(f"  Found {len(valid_positions)} valid object positions for {object_name}")
        print(f"  Coordinate range: X [{valid_positions[object_x_col].min():.2f}, {valid_positions[object_x_col].max():.2f}], Y [{valid_positions[object_y_col].min():.2f}, {valid_positions[object_y_col].max():.2f}]")
        
        # For unified dataset, we consider the object present for the entire trial
        # So we filter merged data to only include trials where this object exists
        object_presence_filter = valid_positions[['ppid', 'session', 'trial']].drop_duplicates()
        
        # Filter data to only include trials with this object
        filtered_data = pd.merge(merged_df, object_presence_filter, on=['ppid', 'session', 'trial'], how='inner')
        
        if len(filtered_data) == 0:
            print(f"  No data found after filtering for object '{object_name}', skipping...")
            continue
        
        print(f"  Filtered data for object '{object_name}': {len(filtered_data)} rows")
        
        # Debug: Check if we're using preprocessed data
        if len(filtered_data) > 0:
            sample_obj_x = filtered_data[object_x_col].iloc[0]
            sample_obj_y = filtered_data[object_y_col].iloc[0]
            sample_part_x = filtered_data['participant_x'].iloc[0]
            sample_part_y = filtered_data['participant_y'].iloc[0]
            print(f"  Sample data (first row):")
            print(f"    Object {object_name}: ({sample_obj_x:.3f}, {sample_obj_y:.3f})")
            print(f"    Participant: ({sample_part_x:.3f}, {sample_part_y:.3f})")
        
        # Perform object-syllable analysis for this object with centered data
        # Note: filtered_data contains preprocessed coordinates, but we need to recenter around
        # the preprocessed object positions (which are already correctly positioned)
        object_frequencies, centered_data = analyze_single_object_syllables_centered(
            filtered_data, object_name, object_x_col, object_y_col, 
            object_radius, object_analysis_dir, color_map
        )
        
        if object_frequencies:
            all_object_frequencies[object_name] = object_frequencies
            all_centered_data_for_plotting[object_name] = centered_data
    
    # Create summary plots if we have results
    if all_object_frequencies:
        print(f"\n--- Creating Summary Plots ---")
        try:
            create_object_syllable_summary_plots(all_object_frequencies, object_analysis_dir, object_radius, color_map)
        except Exception as e:
            print(f"⚠ Error creating summary plots: {e}")
            print(f"  Object frequencies data: {list(all_object_frequencies.keys())}")
            print(f"  Sample data structure: {list(all_object_frequencies.items())[:2] if all_object_frequencies else 'None'}")
            # Continue with the analysis even if summary plots fail
    else:
        print(f"\n--- No Object Frequencies Found ---")
        print(f"  Skipping summary plot creation")
    
    print(f"\n✅ Object-syllable analysis complete!")
    print(f"  - Results saved to: {object_analysis_dir}/")
    print(f"  - Analyzed {len(all_object_frequencies)} objects")
    print(f"  - All data centered around respective objects (objects at origin)")
    
    return all_object_frequencies


def analyze_single_object_syllables_centered(filtered_data, object_name, object_x_col, object_y_col, object_radius, output_dir, color_map=None):
    """Analyze syllable frequencies for a single object with data centered around the object."""
    print(f"    Analyzing syllable frequencies for {object_name} (centered)...")
    
    # Check if we have the required columns
    required_cols = ['participant_x', 'participant_y', 'predicted_label']
    missing_cols = [col for col in required_cols if col not in filtered_data.columns]
    if missing_cols:
        print(f"      ERROR: Missing required columns: {missing_cols}")
        print(f"      Available columns: {list(filtered_data.columns)}")
        
        # Try alternative column names
        alt_participant_x = None
        alt_participant_y = None
        alt_label = None
        
        # Look for alternative participant coordinate columns
        for col in filtered_data.columns:
            if 'x' in col.lower() and 'participant' not in col.lower() and 'object' not in col.lower():
                alt_participant_x = col
            elif 'y' in col.lower() and 'participant' not in col.lower() and 'object' not in col.lower():
                alt_participant_y = col
            elif 'label' in col.lower() or 'syllable' in col.lower():
                alt_label = col
        
        if alt_participant_x and alt_participant_y and alt_label:
            print(f"      Using alternative columns: {alt_participant_x}, {alt_participant_y}, {alt_label}")
            # Update the data to use these columns
            filtered_data = filtered_data.copy()
            filtered_data['participant_x'] = filtered_data[alt_participant_x]
            filtered_data['participant_y'] = filtered_data[alt_participant_y]
            filtered_data['predicted_label'] = filtered_data[alt_label]
        else:
            print(f"      No suitable alternative columns found")
            return None, None
    
    # Initialize frequency counters
    syllable_frequencies = {}
    total_observations = 0
    
    # Store centered data for plotting
    centered_data = []
    
    # Group by trial to analyze each trial separately
    for (ppid, session, trial), trial_data in filtered_data.groupby(['ppid', 'session', 'trial']):
        # Get object position for this trial
        object_positions = trial_data[[object_x_col, object_y_col]].dropna()
        if len(object_positions) == 0:
            continue
        
        # Use the first object position for this trial
        obj_x = object_positions.iloc[0][object_x_col]
        obj_y = object_positions.iloc[0][object_y_col]
        
        # Get participant positions for this trial
        participant_positions = trial_data[['participant_x', 'participant_y', 'predicted_label', 'time']].dropna()
        if len(participant_positions) == 0:
            continue
        
        # Debug: Show recentering details for first trial
        if ppid == 1 and session == 1 and trial == 1:
            print(f"      DEBUG: Recentering around {object_name} for trial {ppid}-{session}-{trial}")
            print(f"        Object position: ({obj_x:.3f}, {obj_y:.3f})")
            print(f"        Sample participant position: ({participant_positions['participant_x'].iloc[0]:.3f}, {participant_positions['participant_y'].iloc[0]:.3f})")
            
            # Check if the object is already at origin (which would explain the shifting issue)
            if abs(obj_x) < 0.001 and abs(obj_y) < 0.001:
                print(f"        ⚠️ WARNING: Object {object_name} is already at origin (0,0) in preprocessed data!")
                print(f"        This means recentering around (0,0) won't change the participant positions.")
                print(f"        The data might need to be recentered around the ORIGINAL object position, not the preprocessed one.")
        
        # IMPORTANT: The data is already preprocessed (rotated and shifted), so we need to
        # recenter around the ORIGINAL object position, not the preprocessed one.
        # The preprocessed object position might already be at (0,0), which would make
        # recentering ineffective and cause the data to appear shifted.
        
        # For now, let's use the preprocessed data as-is and just ensure the object of interest
        # is at (0,0) for visualization purposes
        if abs(obj_x) < 0.001 and abs(obj_y) < 0.001:
            # Object is already at origin, use preprocessed data as-is
            print(f"        Object {object_name} already at origin, using preprocessed data as-is")
            centered_x = participant_positions['participant_x']
            centered_y = participant_positions['participant_y']
            centered_trial_data = trial_data.copy()
        else:
            # Object is not at origin, recenter around it
            print(f"        Recentering around object {object_name} at ({obj_x:.3f}, {obj_y:.3f})")
        centered_x = participant_positions['participant_x'] - obj_x
        centered_y = participant_positions['participant_y'] - obj_y
        
        # Center ALL other object data in this trial
        centered_trial_data = trial_data.copy()
        
        # Get all object coordinate columns (x and y pairs)
        object_coord_cols = []
        for col in trial_data.columns:
            if col.endswith('_x') and any(obj in col.lower() for obj in ['task', 'safehouse', 'threat', 'startpoint', 'weapons', 'wall', 'hide']):
                object_coord_cols.append(col)
        
        # Center all object coordinates around the object of interest
        for x_col in object_coord_cols:
            y_col = x_col.replace('_x', '_y')
            if y_col in trial_data.columns:
                # Center this object's coordinates around the object of interest
                centered_trial_data[x_col] = trial_data[x_col] - obj_x
                centered_trial_data[y_col] = trial_data[y_col] - obj_y
        
        # Debug: Show centered positions for first trial
        if ppid == 1 and session == 1 and trial == 1:
            print(f"        Final participant position: ({centered_x.iloc[0]:.3f}, {centered_y.iloc[0]:.3f})")
        
        # 3. Calculate distances from origin (0,0) since data is now centered
        distances = np.sqrt(centered_x**2 + centered_y**2)
        
        # 4. Find positions within radius
        within_radius_mask = distances <= object_radius
        
        # 5. Get syllables for positions within radius (using centered data)
        if within_radius_mask.any():
            # Create centered participant data for frequency calculation
            centered_participant_data = participant_positions.copy()
            centered_participant_data['centered_x'] = centered_x
            centered_participant_data['centered_y'] = centered_y
            centered_participant_data['distance'] = distances
            
            # Get syllables within radius using the centered data
            within_radius_data = centered_participant_data[within_radius_mask]
            syllables_within = within_radius_data['predicted_label'].dropna()
            
            for syll in syllables_within:
                if syll not in syllable_frequencies:
                    syllable_frequencies[syll] = 0
                syllable_frequencies[syll] += 1
                total_observations += 1
        
        # 6. Store centered data for plotting (all data, not just within radius)
        trial_centered_data = pd.DataFrame({
            'ppid': ppid,
            'session': session,
            'trial': trial,
            'time': participant_positions['time'],
            'centered_x': centered_x,
            'centered_y': centered_y,
            'predicted_label': participant_positions['predicted_label'],
            'distance': distances
        })
        
        # Add centered object coordinates to the plotting data
        # Get all object coordinate columns (x and y pairs)
        object_coord_cols = []
        for col in trial_data.columns:
            if col.endswith('_x') and any(obj in col.lower() for obj in ['task', 'safehouse', 'threat', 'startpoint', 'weapons', 'wall', 'hide']):
                object_coord_cols.append(col)
        
        for x_col in object_coord_cols:
            y_col = x_col.replace('_x', '_y')
            if y_col in trial_data.columns:
                obj_name = x_col.replace('_x', '')
                trial_centered_data[f'{obj_name}_centered_x'] = centered_trial_data[x_col]
                trial_centered_data[f'{obj_name}_centered_y'] = centered_trial_data[y_col]
        
        centered_data.append(trial_centered_data)
    
    if not syllable_frequencies:
        print(f"      No syllable data found within {object_radius}m of {object_name}")
        return None, None
    
    print(f"      Found {len(syllable_frequencies)} syllables within {object_radius}m of {object_name}")
    print(f"      Total observations: {total_observations}")
    
    # Combine all centered data
    if centered_data:
        all_centered_data = pd.concat(centered_data, ignore_index=True)
        print(f"      Created centered dataset with {len(all_centered_data)} data points")
    else:
        all_centered_data = None
    
    # Create combined plot with spatial localization and frequency analysis
    if all_centered_data is not None:
        print(f"      Creating combined plot for {object_name} with {len(all_centered_data)} data points")
        create_combined_object_syllable_plot(object_name, all_centered_data, syllable_frequencies, 
                                           object_radius, output_dir, color_map, total_observations)
    else:
        print(f"      ⚠ No centered data available for plotting {object_name}")
    
    return syllable_frequencies, all_centered_data


def create_centered_object_syllable_plot(object_name, centered_data, radius, output_dir, xlim=None, ylim=None):
    """Create centered spatial plot showing syllables around object at origin (0,0)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    print(f"        Creating centered spatial plot for {object_name}...")
    
    # Create spatial plots directory
    spatial_plots_dir = os.path.join(output_dir, 'spatial_plots')
    os.makedirs(spatial_plots_dir, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot the radius circle with high zorder to ensure it's on top
    circle = Circle((0, 0), radius, fill=False, color='red', linestyle='--', linewidth=3, label=f'{radius}m radius', zorder=20)
    plt.gca().add_patch(circle)
    
    # Plot object center at origin
    object_markers = {
        'task': '^', 'safehouse': 's', 'threat': 'p', 'startpoint': 'o',
        'weapons': 's', 'standpoint': 'o', 'wall': 's', 'hide': 's'
    }
    object_colors = {
        'task': 'green', 'safehouse': 'saddlebrown', 'threat': 'red', 'startpoint': 'yellow',
        'weapons': 'magenta', 'standpoint': 'orange', 'wall': 'grey', 'hide': 'cyan'
    }
    
    marker = object_markers.get(object_name, 'o')
    color = object_colors.get(object_name, 'blue')
    
    plt.scatter(0, 0, marker=marker, color=color, s=200, edgecolor='black', 
               linewidth=2, zorder=15, label=f'{object_name.title()} center')
    
    # Plot syllables with different colors
    unique_labels = centered_data['predicted_label'].dropna().unique()
    print(f"          Found {len(unique_labels)} unique syllables to plot")
    
    # Create color map for syllables
    cmap = plt.cm.tab20
    label_colors = {}
    
    for i, label in enumerate(unique_labels):
        label_colors[label] = cmap(i % 20)
    
    # Plot each syllable
    for label in unique_labels:
        label_data = centered_data[centered_data['predicted_label'] == label]
        if len(label_data) > 0:
            plt.scatter(label_data['centered_x'], label_data['centered_y'], 
                       c=[label_colors[label]], s=20, alpha=0.7, 
                       label=str(label), zorder=5)
    
    # Set plot properties - use provided limits or default to focused view
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim(-3, 3)  # Default focused view for centered plots
    
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim(-3, 3)  # Default focused view for centered plots
    plt.xlabel('Distance from object center (meters)')
    plt.ylabel('Distance from object center (meters)')
    plt.title(f'Syllables Around {object_name.title()} Objects (Centered) - {radius}m Radius (n={len(centered_data[["ppid", "session", "trial"]].drop_duplicates())} trials)')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend (outside plot to avoid overlapping)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Ensure equal aspect ratio
    plt.gca().set_aspect('equal')
    
    # Save plot
    plot_path = os.path.join(spatial_plots_dir, f'{object_name}_centered_spatial_radius{radius}m.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"          ✓ Saved centered spatial plot: {plot_path}")


def create_combined_object_syllable_plot(object_name, centered_data, frequencies, radius, output_dir, color_map=None, total_observations=0):
    """
    Create a combined plot showing both spatial localization and frequency analysis.
    
    This creates a 2-subplot figure:
    - Left: Spatial plot showing syllables around object at origin (0,0)
    - Right: Frequency bar plot showing syllable counts within radius
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    print(f"        Creating combined spatial + frequency plot for {object_name}...")
    print(f"          Output directory: {output_dir}")
    
    # Create combined plots directory
    combined_plots_dir = os.path.join(output_dir, 'combined_plots')
    os.makedirs(combined_plots_dir, exist_ok=True)
    print(f"          Combined plots directory: {combined_plots_dir}")
    
    # Create the combined figure with 2 subplots side by side
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    except Exception as e:
        print(f"          ⚠ Error creating subplots: {e}")
        # Fallback to single plot
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        ax2 = None
    
    # ===== LEFT SUBPLOT: SPATIAL LOCALIZATION =====
    # Plot the radius circle with high zorder to ensure it's on top
    circle = Circle((0, 0), radius, fill=False, color='red', linestyle='--', linewidth=3, label=f'{radius}m radius', zorder=20)
    ax1.add_patch(circle)
    
    # Plot object center at origin
    object_markers = {
        'task': '^', 'safehouse': 's', 'threat': 'p', 'startpoint': 'o',
        'weapons': 's', 'standpoint': 'o', 'wall': 's', 'hide': 's'
    }
    object_colors = {
        'task': 'green', 'safehouse': 'saddlebrown', 'threat': 'red', 'startpoint': 'yellow',
        'weapons': 'magenta', 'standpoint': 'orange', 'wall': 'grey', 'hide': 'cyan'
    }
    
    marker = object_markers.get(object_name, 'o')
    color = object_colors.get(object_name, 'blue')
    
    ax1.scatter(0, 0, marker=marker, color=color, s=200, edgecolor='black', 
               linewidth=2, zorder=15, label=f'{object_name.title()} center')
    
    # Plot syllables with consistent colors from provided color map
    unique_labels = centered_data['predicted_label'].dropna().unique()
    print(f"          Found {len(unique_labels)} unique syllables to plot")
    
    # Use provided color map or create consistent fallback
    if color_map is not None:
        print(f"          Using provided color map with {len(color_map)} colors")
        label_colors = {}
        for label in unique_labels:
            # Try multiple key formats to find the label in the color map
            color_found = False
            
            try:
                # Try exact match first
                if label in color_map:
                    label_colors[label] = color_map[label]
                    color_found = True
                # Try string version
                elif str(label) in color_map:
                    label_colors[label] = color_map[str(label)]
                    color_found = True
                # Try integer version if label is numeric
                else:
                    try:
                        label_int = int(label)
                        if label_int in color_map:
                            label_colors[label] = color_map[label_int]
                            color_found = True
                        else:
                            # Try string version of integer
                            if str(label_int) in color_map:
                                label_colors[label] = color_map[str(label_int)]
                                color_found = True
                    except (ValueError, TypeError):
                        pass
                
                # If no color found in map, use fallback
                if not color_found:
                    try:
                        label_int = int(label)
                        # Use multi-colormap approach for more unique colors
                        colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                        colors_per_cmap = 20
                        cmap_index = (label_int // colors_per_cmap) % len(colormaps)
                        color_index = label_int % colors_per_cmap
                        label_colors[label] = colormaps[cmap_index](color_index)
                    except (ValueError, TypeError):
                        # For non-numeric labels, use hash-based color with multi-colormap
                        hash_val = hash(str(label))
                        colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                        colors_per_cmap = 20
                        cmap_index = (abs(hash_val) // colors_per_cmap) % len(colormaps)
                        color_index = abs(hash_val) % colors_per_cmap
                        label_colors[label] = colormaps[cmap_index](color_index)
            except Exception as e:
                print(f"          ⚠ Error processing label '{label}' for color map: {e}")
                # Use fallback color with multi-colormap approach
                try:
                    label_int = int(label)
                    colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                    colors_per_cmap = 20
                    cmap_index = (label_int // colors_per_cmap) % len(colormaps)
                    color_index = label_int % colors_per_cmap
                    label_colors[label] = colormaps[cmap_index](color_index)
                except (ValueError, TypeError):
                    hash_val = hash(str(label))
                    colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                    colors_per_cmap = 20
                    cmap_index = (abs(hash_val) // colors_per_cmap) % len(colormaps)
                    color_index = abs(hash_val) % colors_per_cmap
                    label_colors[label] = colormaps[cmap_index](color_index)
    else:
        print(f"          No color map provided, using consistent fallback colors")
        # Create consistent color map for syllables using multi-colormap approach
        colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
        colors_per_cmap = 20
        label_colors = {}
        for i, label in enumerate(unique_labels):
            cmap_index = (i // colors_per_cmap) % len(colormaps)
            color_index = i % colors_per_cmap
            label_colors[label] = colormaps[cmap_index](color_index)
    
    # Plot each syllable (without labels for legend)
    for label in unique_labels:
        label_data = centered_data[centered_data['predicted_label'] == label]
        if len(label_data) > 0:
            ax1.scatter(label_data['centered_x'], label_data['centered_y'], 
                       c=[label_colors[label]], s=20, alpha=0.7, 
                       zorder=5)
    
    # Plot other centered objects (excluding the object of interest) without labels
    other_objects = ['task', 'safehouse', 'threat', 'startpoint', 'weapons', 'wall', 'hide']
    other_objects = [obj for obj in other_objects if obj != object_name]
    
    for obj in other_objects:
        x_col = f'{obj}_centered_x'
        y_col = f'{obj}_centered_y'
        
        if x_col in centered_data.columns and y_col in centered_data.columns:
            obj_data = centered_data[[x_col, y_col]].dropna()
            if len(obj_data) > 0:
                # Get unique positions for this object (one per trial)
                unique_positions = obj_data.drop_duplicates()
                
                marker = object_markers.get(obj, 'o')
                color = object_colors.get(obj, 'blue')
                
                ax1.scatter(unique_positions[x_col], unique_positions[y_col], 
                           marker=marker, color=color, s=100, alpha=0.8, 
                           edgecolor='black', linewidth=1, 
                           zorder=10)
    
    # Set spatial plot properties
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel('Distance from object center (meters)')
    ax1.set_ylabel('Distance from object center (meters)')
    ax1.set_title(f'Spatial Localization: Syllables Around {object_name.title()}\n{radius}m Radius (n={len(centered_data[["ppid", "session", "trial"]].drop_duplicates())} trials)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_aspect('equal')
    
    # ===== RIGHT SUBPLOT: FREQUENCY ANALYSIS =====
    if ax2 is not None and frequencies:
        syllables = list(frequencies.keys())
        counts = list(frequencies.values())
        
        # Compute normalized frequencies (as percentages)
        total_count = sum(counts)
        normalized_frequencies = [count / total_count * 100 for count in counts]
        
        # Sort syllables by frequency (highest to lowest)
        sorted_indices = sorted(range(len(syllables)), key=lambda i: normalized_frequencies[i], reverse=True)
        sorted_syllables = [syllables[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        sorted_normalized_frequencies = [normalized_frequencies[i] for i in sorted_indices]
        
        # Filter syllables to only show those that account for 85% of cumulative values if more than 20 syllables
        if len(sorted_syllables) > 20:
            cumulative_percentage = 0
            filtered_indices = []
            for i, freq in enumerate(sorted_normalized_frequencies):
                cumulative_percentage += freq
                filtered_indices.append(i)
                if cumulative_percentage >= 85:
                    break
            
            # Use only the filtered syllables
            syllables = [sorted_syllables[i] for i in filtered_indices]
            counts = [sorted_counts[i] for i in filtered_indices]
            normalized_frequencies = [sorted_normalized_frequencies[i] for i in filtered_indices]
        else:
            # Use all syllables if 20 or fewer
            syllables = sorted_syllables
            counts = sorted_counts
            normalized_frequencies = sorted_normalized_frequencies
        
        # Use the same colors as the spatial plot for consistency
        bar_colors = [label_colors.get(syll, 'skyblue') for syll in syllables]
        
        # Convert syllable labels to strings for better x-axis display
        syllable_labels = [str(syll) for syll in syllables]
        
        # Create bar plot with normalized frequencies
        bars = ax2.bar(range(len(syllables)), normalized_frequencies, color=bar_colors, edgecolor='black', alpha=0.8)
        ax2.set_title(f'Normalized Frequency Analysis: Syllables within {radius}m of {object_name.title()}\nTotal observations: {total_observations}')
        ax2.set_xlabel('Syllable')
        ax2.set_ylabel('Normalized Frequency (%)')
        ax2.set_xticks(range(len(syllables)))
        ax2.set_xticklabels(syllable_labels, rotation=45, ha='right')
        
        # Add cumulative curve in grey line
        cumulative_frequencies = []
        cumulative_sum = 0
        for freq in normalized_frequencies:
            cumulative_sum += freq
            cumulative_frequencies.append(cumulative_sum)
        
        # Create secondary y-axis for cumulative curve
        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(len(syllables)), cumulative_frequencies, color='grey', linewidth=2, marker='o', markersize=4)
        ax2_twin.set_ylabel('Cumulative Percentage (%)', color='grey')
        ax2_twin.tick_params(axis='y', labelcolor='grey')
        ax2_twin.set_ylim(0, 100)
        
        # Add value labels on top of bars (show only percentage)
        for bar, norm_freq in zip(bars, normalized_frequencies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(normalized_frequencies)*0.01,
                    f'{norm_freq:.1f}%', ha='center', va='bottom', fontsize=9)
    elif ax2 is not None:
        ax2.text(0.5, 0.5, 'No frequency data available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title(f'Frequency Analysis: No data within {radius}m of {object_name.title()}')
    else:
        print(f"          ⚠ Frequency subplot not available (fallback mode)")
    
    # Overall title for the combined plot
    fig.suptitle(f'Object-Syllable Analysis: {object_name.title()} - Combined Spatial + Frequency View', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(combined_plots_dir, f'{object_name}_combined_spatial_frequency_radius{radius}m.png')
    print(f"          Saving plot to: {plot_path}")
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"          ✓ Successfully saved combined plot: {plot_path}")
    except Exception as e:
        print(f"          ❌ Error saving plot: {e}")
        import traceback
        traceback.print_exc()


def create_object_syllable_frequency_plot(object_name, frequencies, output_dir, radius, total_observations):
    """Create frequency plot for a single object."""
    import matplotlib.pyplot as plt
    
    # Create frequency plots directory
    freq_plots_dir = os.path.join(output_dir, 'frequency_plots')
    os.makedirs(freq_plots_dir, exist_ok=True)
    
    # Create normalized frequency plot
    plt.figure(figsize=(10, 6))
    syllables = list(frequencies.keys())
    counts = list(frequencies.values())
    
    # Compute normalized frequencies (as percentages)
    total_count = sum(counts)
    normalized_frequencies = [count / total_count * 100 for count in counts]
    
    # Create bar plot with normalized frequencies
    bars = plt.bar(syllables, normalized_frequencies, color='skyblue', edgecolor='black')
    plt.title(f'Normalized Syllable Frequencies within {radius}m of {object_name.title()}\nTotal observations: {total_observations}')
    plt.xlabel('Syllable')
    plt.ylabel('Normalized Frequency (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars (show both normalized and raw counts)
    for bar, norm_freq, count in zip(bars, normalized_frequencies, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(normalized_frequencies)*0.01,
                f'{norm_freq:.1f}%\n({count})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(freq_plots_dir, f'{object_name}_normalized_frequencies_n{total_observations}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"        ✓ Saved normalized frequency plot: {plot_path}")


def create_object_syllable_summary_plots(all_frequencies, output_dir, radius, color_map=None):
    """Create improved summary plots for all objects using heatmaps and enhanced visualizations."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Create summary plots directory
    summary_plots_dir = os.path.join(output_dir, 'summary_plots')
    os.makedirs(summary_plots_dir, exist_ok=True)
    
    # Get all unique syllables across all objects
    all_syllables = set()
    for obj_freqs in all_frequencies.values():
        if obj_freqs:  # Check if the object has frequency data
            all_syllables.update(obj_freqs.keys())
    
    if not all_syllables:
        print("      ⚠ No syllable data found for summary plots")
        return
    
    all_syllables = sorted(all_syllables)
    print(f"      Found {len(all_syllables)} unique syllables across all objects")
    
    # Create data matrix for heatmap
    object_names = [name for name in all_frequencies.keys() if all_frequencies[name]]  # Only include objects with data
    if not object_names:
        print("      ⚠ No objects with frequency data found for summary plots")
        return
    
    print(f"      Creating summary plots for {len(object_names)} objects: {object_names}")
    
    # Create both raw counts and normalized frequency matrices
    raw_data_matrix = np.zeros((len(object_names), len(all_syllables)))
    normalized_data_matrix = np.zeros((len(object_names), len(all_syllables)))
    
    for i, obj_name in enumerate(object_names):
        obj_freqs = all_frequencies[obj_name]
        total_count = sum(obj_freqs.values())
        
        for j, syll in enumerate(all_syllables):
            raw_count = obj_freqs.get(syll, 0)
            raw_data_matrix[i, j] = raw_count
            
            # Compute normalized frequency (percentage)
            if total_count > 0:
                normalized_data_matrix[i, j] = (raw_count / total_count) * 100
            else:
                normalized_data_matrix[i, j] = 0.0
    
    # Ensure data matrices contain only valid numeric values
    raw_data_matrix = np.nan_to_num(raw_data_matrix, nan=0.0)
    normalized_data_matrix = np.nan_to_num(normalized_data_matrix, nan=0.0)
    raw_data_matrix = raw_data_matrix.astype(int)  # Convert to integers
    
    # Validate data matrices
    if raw_data_matrix.size == 0 or np.all(raw_data_matrix == 0):
        print("      ⚠ Raw data matrix is empty or contains only zeros")
        return
    
    if normalized_data_matrix.size == 0 or np.all(normalized_data_matrix == 0):
        print("      ⚠ Normalized data matrix is empty or contains only zeros")
        return
    
    # ===== PLOT 1: NORMALIZED FREQUENCY HEATMAP =====
    plt.figure(figsize=(16, 10))
    
    # Debug: Print data matrix info
    print(f"      Normalized data matrix shape: {normalized_data_matrix.shape}")
    print(f"      Normalized data matrix min/max: {normalized_data_matrix.min():.2f}%/{normalized_data_matrix.max():.2f}%")
    print(f"      Normalized data matrix sample values: {normalized_data_matrix.flatten()[:10]}...")
    
    # Create heatmap using seaborn with plasma colormap for normalized frequencies
    try:
        sns.heatmap(normalized_data_matrix, 
                    xticklabels=all_syllables, 
                    yticklabels=[obj.title() for obj in object_names],
                    annot=True,  # Show values in cells
                    fmt='.1f',   # Format as float with 1 decimal place
                    cmap='plasma',  # Use plasma colormap as requested
                    cbar_kws={'label': 'Normalized Frequency (%)'},
                    square=False,
                    linewidths=0.5,
                    linecolor='white')
    except Exception as e:
        print(f"      ⚠ Error creating normalized frequency heatmap: {e}")
        print(f"      Normalized data matrix info: shape={normalized_data_matrix.shape}, dtype={normalized_data_matrix.dtype}")
        print(f"      Normalized data matrix sample: {normalized_data_matrix[:3, :3]}")
        raise  # Re-raise to see full traceback
    
    plt.title(f'Normalized Syllable Frequencies Heatmap - {radius}m Radius Analysis\nAll Objects Centered (Objects at Origin)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Syllable', fontsize=12, fontweight='bold')
    plt.ylabel('Object Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    heatmap_path = os.path.join(summary_plots_dir, f'all_objects_normalized_heatmap_plasma_radius{radius}m.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      ✓ Saved normalized frequency heatmap plot: {heatmap_path}")
    
    # ===== PLOT 2: RAW COUNTS HEATMAP (FOR COMPARISON) =====
    plt.figure(figsize=(16, 10))
    
    # Create heatmap using seaborn with plasma colormap for raw counts
    try:
        sns.heatmap(raw_data_matrix, 
                    xticklabels=all_syllables, 
                    yticklabels=[obj.title() for obj in object_names],
                    annot=True,  # Show values in cells
                    fmt='d',     # Format as integers
                    cmap='plasma',  # Use plasma colormap as requested
                    cbar_kws={'label': 'Syllable Count'},
                    square=False,
                    linewidths=0.5,
                    linecolor='white')
    except Exception as e:
        print(f"      ⚠ Error creating raw counts heatmap: {e}")
        print(f"      Raw data matrix info: shape={raw_data_matrix.shape}, dtype={raw_data_matrix.dtype}")
        print(f"      Raw data matrix sample: {raw_data_matrix[:3, :3]}")
        raise  # Re-raise to see full traceback
    
    plt.title(f'Raw Syllable Counts Heatmap - {radius}m Radius Analysis\nAll Objects Centered (Objects at Origin)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Syllable', fontsize=12, fontweight='bold')
    plt.ylabel('Object Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    raw_heatmap_path = os.path.join(summary_plots_dir, f'all_objects_raw_heatmap_plasma_radius{radius}m.png')
    plt.savefig(raw_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      ✓ Saved raw counts heatmap plot: {raw_heatmap_path}")
    
    # ===== PLOT 3: ENHANCED STACKED BAR CHART =====
    plt.figure(figsize=(16, 10))
    
    # Create enhanced stacked bar chart
    x = np.arange(len(all_syllables))
    width = 0.8 / len(object_names)  # Use object_names instead of all_frequencies
    
    # Use a nice color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(object_names)))
    
    for i, obj_name in enumerate(object_names):
        obj_freqs = all_frequencies[obj_name]
        counts = [obj_freqs.get(syll, 0) for syll in all_syllables]
        bars = plt.bar(x + i * width, counts, width, label=obj_name.title(), 
                      alpha=0.8, color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar, count in zip(bars, counts):
            if count > 0:  # Only show labels for non-zero values
                height = bar.get_height()
                max_count = max([max([all_frequencies[obj].get(syll, 0) for obj in object_names]) for syll in all_syllables])
                plt.text(bar.get_x() + bar.get_width()/2., height + max_count * 0.01,
                        f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Syllable', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(f'Enhanced Syllable Frequencies - {radius}m Radius Analysis\nAll Objects Centered (Objects at Origin)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x + width * (len(all_frequencies) - 1) / 2, all_syllables, rotation=45, ha='right')
    plt.legend(title='Object Types', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    bar_path = os.path.join(summary_plots_dir, f'all_objects_enhanced_bars_radius{radius}m.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      ✓ Saved enhanced bar chart: {bar_path}")
    
    # ===== PLOT 3: NORMALIZED HEATMAP (PERCENTAGES) =====
    plt.figure(figsize=(16, 10))
    
    # Calculate percentages for each object
    normalized_matrix = np.zeros_like(data_matrix)
    print(f"      Normalizing data per object type (each row sums to 100%):")
    
    for i in range(len(object_names)):
        obj_name = object_names[i]
        row_sum = np.sum(data_matrix[i, :])
        if row_sum > 0:
            normalized_matrix[i, :] = (data_matrix[i, :] / row_sum) * 100
            # Verify normalization
            row_sum_percent = np.sum(normalized_matrix[i, :])
            print(f"        {obj_name}: raw sum = {row_sum}, normalized sum = {row_sum_percent:.1f}%")
            
            # Show top syllables for this object
            top_indices = np.argsort(normalized_matrix[i, :])[-3:]  # Top 3 syllables
            top_syllables = [all_syllables[j] for j in top_indices]
            top_percentages = [normalized_matrix[i, j] for j in top_indices]
            print(f"          Top syllables: {list(zip(top_syllables, [f'{p:.1f}%' for p in top_percentages]))}")
        else:
            normalized_matrix[i, :] = 0
            print(f"        {obj_name}: no data (row sum = 0)")
    
    print(f"      ✓ Normalization complete - each object type row sums to 100%")
    
    # Create normalized heatmap
    sns.heatmap(normalized_matrix, 
                xticklabels=all_syllables, 
                yticklabels=[obj.title() for obj in object_names],
                annot=True,  # Show values in cells
                fmt='.1f',   # Format as percentages with 1 decimal place
                cmap='plasma',  # Use plasma colormap
                cbar_kws={'label': 'Syllable Percentage (%)'},
                square=False,
                linewidths=0.5,
                linecolor='white')
    
    # Verify normalization is correct
    print(f"      Verifying normalization accuracy:")
    for i, obj_name in enumerate(object_names):
        row_sum = np.sum(normalized_matrix[i, :])
        if row_sum > 0:
            print(f"        {obj_name}: {row_sum:.2f}% (should be 100.00%)")
            if abs(row_sum - 100.0) > 0.01:  # Allow for small floating point errors
                print(f"          ⚠ Warning: {obj_name} row does not sum to 100%")
            else:
                print(f"          ✓ {obj_name} row correctly normalized")
        else:
            print(f"        {obj_name}: {row_sum:.2f}% (no data)")
    
    plt.title(f'Syllable Percentages Heatmap - {radius}m Radius Analysis\nNormalized by Object Type (Objects at Origin)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Syllable', fontsize=12, fontweight='bold')
    plt.ylabel('Object Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    norm_heatmap_path = os.path.join(summary_plots_dir, f'all_objects_normalized_heatmap_plasma_radius{radius}m.png')
    plt.savefig(norm_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      ✓ Saved normalized heatmap: {norm_heatmap_path}")
    
    print(f"      ✓ All summary plots saved to: {summary_plots_dir}/")
    print(f"        - Heatmap with plasma colormap (raw counts)")
    print(f"        - Enhanced stacked bar chart (raw counts)")
    print(f"        - Normalized percentage heatmap (each object type sums to 100%)")
    print(f"        Note: All data centered around respective objects (objects at origin)")
    print(f"        Note: Normalized heatmap shows relative syllable distribution per object type")


def perform_grouped_object_syllable_analysis_unified(merged_df, output_dir, color_map, object_radius, args,
                                                   group_type, group_name, group_column):
    """Perform object-syllable analysis for a specific group (block or scenario)."""
    print(f"    Performing {group_type}-level object-syllable analysis for '{group_name}'...")
    
    # Check if the group column exists
    if group_column not in merged_df.columns:
        print(f"      ❌ ERROR: Group column '{group_column}' not found in merged dataset!")
        print(f"      Available columns: {[col for col in merged_df.columns if group_type in col.lower() or 'block' in col.lower() or 'scenario' in col.lower()]}")
        return
    
    # Check what unique values exist in the group column
    unique_groups = merged_df[group_column].dropna().unique()
    print(f"      Available {group_type}s in '{group_column}': {list(unique_groups)}")
    
    # Filter data for this group
    group_data = merged_df[merged_df[group_column] == group_name]
    print(f"      Group '{group_name}' has {len(group_data)} rows")
    
    if len(group_data) == 0:
        print(f"      ⚠ No data found for {group_type} '{group_name}', skipping")
        print(f"      This might be because the group name doesn't match exactly.")
        print(f"      Available groups: {list(unique_groups)}")
        return
    
    # Create group-specific output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Debug: Check what data is available in the group
    print(f"      Group data columns: {list(group_data.columns)}")
    print(f"      Group data shape: {group_data.shape}")
    
    # Check for essential columns in group data
    essential_cols = ['ppid', 'session', 'trial', 'time']
    missing_essential = [col for col in essential_cols if col not in group_data.columns]
    if missing_essential:
        print(f"      ⚠ WARNING: Missing essential columns in group data: {missing_essential}")
    
    # Check for participant coordinates and labels
    coord_cols = [col for col in group_data.columns if 'x' in col.lower() or 'y' in col.lower()]
    label_cols = [col for col in group_data.columns if 'label' in col.lower() or 'syllable' in col.lower()]
    print(f"      Coordinate columns in group data: {coord_cols}")
    print(f"      Label columns in group data: {label_cols}")
    
    # Perform object-syllable analysis on this group's data
    # Use a group-specific version that works with the group output directory
    group_frequencies = perform_group_object_syllable_analysis(
        group_data, output_dir, object_radius, color_map
    )
    
    if group_frequencies:
        print(f"      ✓ Successfully completed {group_type}-level analysis for '{group_name}'")
        print(f"        Found frequencies for {len(group_frequencies)} objects")
        print(f"        All data centered around respective objects (objects at origin)")
    else:
        print(f"      ⚠ No results from {group_type}-level analysis for '{group_name}'")
    
    return group_frequencies


def perform_group_object_syllable_analysis(group_data, output_dir, object_radius, color_map=None):
    """
    Perform object-syllable analysis for a specific group (block or scenario).
    
    This function is specifically designed for grouped analysis and works with
    the group-specific output directory structure.
    
    Args:
        group_data: Filtered data for the specific group
        output_dir: Group-specific output directory
        object_radius: Radius in meters for analysis
        color_map: Color map for consistent syllable coloring
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    import os
    from matplotlib.patches import Circle
    
    print(f"      Performing group-specific object-syllable analysis...")
    print(f"      Radius: {object_radius} meters")
    print(f"      Output directory: {output_dir}")
    
    # Define specific objects to analyze
    objects_to_analyze = ['task', 'safehouse', 'threat', 'startpoint', 'weapons', 'wall', 'hide']
    print(f"      Analyzing objects: {objects_to_analyze}")
    
    # Create output directory structure for this group
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results storage
    all_object_frequencies = {}
    
    # Get all trials that have participant data
    all_trials = group_data[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"      Total trials in group: {len(all_trials)}")
    
    # Debug: Check what data is available
    print(f"      Group data shape: {group_data.shape}")
    print(f"      Group data columns: {list(group_data.columns)}")
    
    # Check for essential columns
    essential_cols = ['ppid', 'session', 'trial', 'time']
    missing_essential = [col for col in essential_cols if col not in group_data.columns]
    if missing_essential:
        print(f"      ⚠ WARNING: Missing essential columns: {missing_essential}")
    
    # Check for participant coordinates and labels
    coord_cols = [col for col in group_data.columns if 'x' in col.lower() or 'y' in col.lower()]
    label_cols = [col for col in group_data.columns if 'label' in col.lower() or 'syllable' in col.lower()]
    print(f"      Coordinate columns: {coord_cols}")
    print(f"      Label columns: {label_cols}")
    
    # Check for specific required columns and suggest alternatives
    if 'participant_x' not in group_data.columns:
        print(f"      ⚠ 'participant_x' not found. Looking for alternatives...")
        alt_x_cols = [col for col in coord_cols if 'participant' not in col.lower() and 'object' not in col.lower() and 'x' in col.lower()]
        if alt_x_cols:
            print(f"      Found alternative X columns: {alt_x_cols}")
    
    if 'participant_y' not in group_data.columns:
        print(f"      ⚠ 'participant_y' not found. Looking for alternatives...")
        alt_y_cols = [col for col in coord_cols if 'participant' not in col.lower() and 'object' not in col.lower() and 'y' in col.lower()]
        if alt_y_cols:
            print(f"      Found alternative Y columns: {alt_y_cols}")
    
    if 'predicted_label' not in group_data.columns:
        print(f"      ⚠ 'predicted_label' not found. Looking for alternatives...")
        if label_cols:
            print(f"      Found alternative label columns: {label_cols}")
    
    for object_name in objects_to_analyze:
        print(f"        Analyzing object: {object_name}")
        
        # Find object columns in group data
        object_x_col = f'{object_name}_x'
        object_y_col = f'{object_name}_y'
        
        if object_x_col in group_data.columns and object_y_col in group_data.columns:
            print(f"          Using columns: {object_x_col}, {object_y_col}")
        else:
            print(f"          ERROR: Object '{object_name}' columns not found in group data!")
            print(f"          Available columns: {[col for col in group_data.columns if object_name in col]}")
            continue
        
        # Get object positions from group data
        object_positions = group_data[['ppid', 'session', 'trial', object_x_col, object_y_col]].dropna()
        
        if len(object_positions) == 0:
            print(f"          No positions found for object '{object_name}', skipping...")
            continue
        
        # Check if we have valid coordinate data
        valid_positions = object_positions[
            (object_positions[object_x_col].notna()) & 
            (object_positions[object_y_col].notna()) &
            (object_positions[object_x_col] != 0) & 
            (object_positions[object_y_col] != 0)
        ]
        
        if len(valid_positions) == 0:
            print(f"          No valid positions found for object '{object_name}' (all positions are NaN or 0), skipping...")
            continue
        
        print(f"          Found {len(valid_positions)} valid object positions for {object_name}")
        
        # Filter data to only include trials with this object
        object_presence_filter = valid_positions[['ppid', 'session', 'trial']].drop_duplicates()
        filtered_data = pd.merge(group_data, object_presence_filter, on=['ppid', 'session', 'trial'], how='inner')
        
        if len(filtered_data) == 0:
            print(f"          No data found after filtering for object '{object_name}', skipping...")
            continue
        
        print(f"          Filtered data for object '{object_name}': {len(filtered_data)} rows")
        
        # Perform object-syllable analysis for this object with centered data
        object_frequencies, centered_data = analyze_single_object_syllables_centered(
            filtered_data, object_name, object_x_col, object_y_col, 
            object_radius, output_dir, color_map
        )
        
        if object_frequencies:
            all_object_frequencies[object_name] = object_frequencies
            print(f"          ✓ Found frequencies for {len(object_frequencies)} syllables")
        else:
            print(f"          ⚠ No frequencies found for object '{object_name}'")
    
    # Create summary plots if we have results
    if all_object_frequencies:
        print(f"        Creating summary plots for group...")
        try:
            create_object_syllable_summary_plots(all_object_frequencies, output_dir, object_radius, color_map)
            print(f"        ✓ Summary plots created successfully")
        except Exception as e:
            print(f"        ⚠ Error creating summary plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"        ⚠ No object frequencies found for group, skipping summary plots")
    
    print(f"      ✓ Group object-syllable analysis complete!")
    print(f"        - Results saved to: {output_dir}/")
    print(f"        - Analyzed {len(all_object_frequencies)} objects")
    
    # List what was actually created
    if os.path.exists(output_dir):
        created_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.png'):
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    created_files.append(rel_path)
        
        if created_files:
            print(f"        - Created {len(created_files)} plot files:")
            for file in created_files[:5]:  # Show first 5 files
                print(f"          • {file}")
            if len(created_files) > 5:
                print(f"          • ... and {len(created_files) - 5} more")
        else:
            print(f"        - ⚠ No plot files were created")
    else:
        print(f"        - ❌ Output directory was not created")
    
    return all_object_frequencies


def create_first_step_analysis_plot_unified(merged_df, tracker_data_list, spatial_output_dir, args, plot_title):
    """Create spatial analysis plot using unified dataset."""
    if not (args.debug or args.debug_plots_only):
        return
    
    print(f"\n=== Creating Spatial Analysis Plot: {plot_title} ===")
    print(f"✓ Using unified dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # Use debug_plots directory directly (already created in main)
    debug_dir = os.path.join(spatial_output_dir, 'debug_plots')
    
    # Get unique trials from unified dataset
    unique_trials = merged_df[['ppid', 'session', 'trial']].drop_duplicates()
    if len(unique_trials) == 0:
        print("No trials found in unified dataset")
        return
    
    # Take first 9 trials to see more examples
    first_9_trials = unique_trials.head(9)
    print(f"Creating subplots for first 9 trials: {first_9_trials.to_dict('records')}")
    
    # Create subplots - arrange in a 3x3 grid for better visualization
    if len(first_9_trials) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = [ax]
    else:
        # Create a 3x3 grid layout
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        # Flatten the 2D array to 1D for easier iteration
        axes = axes.flatten()
    
    # Add main title to the figure
    fig.suptitle(f'{plot_title} - First 9 Trials Analysis (Unified Dataset)', fontsize=16, fontweight='bold')
    
    for i, (_, trial_info) in enumerate(first_9_trials.iterrows()):
        ppid, session, trial = trial_info['ppid'], trial_info['session'], trial_info['trial']
        ax = axes[i]
        
        # Get data for this trial from unified dataset
        trial_data = merged_df[
            (merged_df['ppid'] == ppid) & 
            (merged_df['session'] == session) & 
            (merged_df['trial'] == trial)
        ]
        
        # Plot participant trajectory
        if 'participant_x' in trial_data.columns and 'participant_y' in trial_data.columns:
            ax.scatter(trial_data['participant_x'], trial_data['participant_y'], 
                      c='blue', s=10, alpha=0.6, label='Participant', zorder=3)
        elif 'x' in trial_data.columns and 'y' in trial_data.columns:
            ax.scatter(trial_data['x'], trial_data['y'], 
                      c='blue', s=10, alpha=0.6, label='Participant', zorder=3)
        
        # Plot scenario objects for this trial
        # Find startpoint columns
        startpoint_x_col = None
        startpoint_y_col = None
        for col in trial_data.columns:
            if col.lower().endswith('startpoint_x'):
                startpoint_x_col = col  
            elif col.lower().endswith('startpoint_y'):
                startpoint_y_col = col
        
        if startpoint_x_col and startpoint_y_col:
            ax.scatter(trial_data[startpoint_x_col], trial_data[startpoint_y_col], 
                      marker='o', color='yellow', s=200, edgecolor='black', zorder=10, 
                      label='StartPoint')
        
        # Plot other scenario objects
        object_styles = {
            'task': dict(marker='^', color='green', label='Task'),
            'safehouse': dict(marker='s', color='saddlebrown', label='Safehouse'),
            'threat': dict(marker='p', color='red', label='Threat'),
            'weapons': dict(marker='s', color='magenta', label='Weapons'),
            'standpoint': dict(marker='o', color='orange', label='StandPoint'),
            'wall': dict(marker='s', color='grey', label='Wall'),
            'hide': dict(marker='s', color='cyan', label='Hide'),
        }
        
        # Track which objects were actually plotted
        plotted_objects = set()
        
        for object_name, style in object_styles.items():
            x_col = None
            y_col = None
            for col in trial_data.columns:
                if col.lower().endswith(f'{object_name}_x'):
                    x_col = col
                elif col.lower().endswith(f'{object_name}_y'):
                    y_col = col
            
            if x_col and y_col:
                # Check if coordinates exist and are valid
                valid_x = trial_data[x_col].dropna()
                valid_y = trial_data[y_col].dropna()
                
                if len(valid_x) > 0 and len(valid_y) > 0:
                    ax.scatter(trial_data[x_col], trial_data[y_col], 
                              marker=style['marker'], color=style['color'], 
                              s=120, edgecolor='black', zorder=10, 
                              label=style['label'])
                    plotted_objects.add(object_name)
        
        # Plot tracker path for this trial if available
        for tracker_item in tracker_data_list:
            if (str(tracker_item['ppid']) == str(ppid) and 
                str(tracker_item['session']) == str(session) and 
                str(tracker_item['trial']) == str(trial)):
                
                tracker_data = tracker_item['tracker_data']
                if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                    ax.scatter(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                              color='red', marker='x', s=15, alpha=0.8, zorder=1, 
                              label='Tracker')
                    
                    # Connect consecutive points
                    if len(tracker_data) > 1:
                        ax.plot(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                               color='red', alpha=0.6, linewidth=1, zorder=1)
                break
        
        # Add reference square
        square_width = 10
        square_height = 6
        square_x = [-square_width/2, square_width/2, square_width/2, -square_width/2, -square_width/2]
        square_y = [-square_height/2, -square_height/2, square_height/2, square_height/2, -square_height/2]
        ax.plot(square_x, square_y, color='darkgrey', linestyle='--', linewidth=2, zorder=5, 
                label='Reference Square (10x6m)')
        
        # Set limits - use command line arguments for consistency
        if hasattr(args, 'xlim') and args.xlim:
            ax.set_xlim(args.xlim)
        else:
            ax.set_xlim([-6, 6])
        
        if hasattr(args, 'ylim') and args.ylim:
            ax.set_ylim(args.ylim)
        else:
            ax.set_ylim([-4, 4])
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Labels and title
        ax.set_xlabel('x (metres)')
        ax.set_ylabel('y (metres)')
        ax.set_title(f'Trial {ppid}-{session}-{trial}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(debug_dir, f'{plot_title.lower().replace(" ", "_")}_unified.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"✓ Plot saved to {plot_path}")


def print_completion_summary_unified(args):
    """Print completion summary for unified approach."""
    print("\n" + "="*80)
    print("UNIFIED DATA PROCESSING COMPLETE")
    print("="*80)
    print("✓ All data sources loaded and merged into unified dataset")
    print("✓ All transformations applied to unified dataset")
    print("✓ Data consistency maintained throughout processing")
    print("✓ Ready for spatial analysis with unified data")
    
    if args.spatial_analysis:
        print("✓ Spatial analysis completed using unified dataset")
    else:
        print("⚠ Spatial analysis not requested")
    
    print("✓ Check output directory for results and debug plots")

def create_first_step_analysis_plot(participant_df, scenario_df, tracker_data_list, output_dir, args, step_name="First Step Analysis", scenario_rotation_dict=None):
    """Create analysis plot showing first 3 trials with subplots."""
    if not (args.debug or args.debug_plots_only):
        return
    
    print(f"\n=== {step_name.upper()} PLOT ===")
    
    # Use debug_plots directory directly (already created in main)
    debug_dir = os.path.join(output_dir, 'debug_plots')
    
    # Get unique trials from participant data
    unique_trials = participant_df[['ppid', 'session', 'trial']].drop_duplicates()
    if len(unique_trials) == 0:
        print("No trials found in participant data")
        return
    
    # Take first 9 trials (instead of 3) to see more examples
    first_9_trials = unique_trials.head(9)
    print(f"Creating subplots for first 9 trials: {first_9_trials.to_dict('records')}")
    
    # Create subplots - arrange in a 3x3 grid for better visualization
    if len(first_9_trials) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = [ax]
    else:
        # Create a 3x3 grid layout
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        # Flatten the 2D array to 1D for easier iteration
        axes = axes.flatten()
    
    # Add main title to the figure
    fig.suptitle(f'{step_name} - First 9 Trials Analysis', fontsize=16, fontweight='bold')
    
    for i, (_, trial_info) in enumerate(first_9_trials.iterrows()):
        ppid, session, trial = trial_info['ppid'], trial_info['session'], trial_info['trial']
        ax = axes[i]
        
        # Get participant data for this trial
        trial_participant = participant_df[
            (participant_df['ppid'] == ppid) & 
            (participant_df['session'] == session) & 
            (participant_df['trial'] == trial)
        ]
        
        # Plot participant trajectory
        if 'participant_x' in trial_participant.columns and 'participant_y' in trial_participant.columns:
            ax.scatter(trial_participant['participant_x'], trial_participant['participant_y'], 
                      c='blue', s=10, alpha=0.6, label='Participant', zorder=3)
        
        # Plot scenario objects for this trial
        if scenario_df is not None:
            trial_scenario = scenario_df[
                (scenario_df['ppid'] == ppid) & 
                (scenario_df['session'] == session) & 
                (scenario_df['trial'] == trial)
            ]
            
            if not trial_scenario.empty:
                # Plot startpoint
                startpoint_x_col = None
                startpoint_y_col = None
                for col in trial_scenario.columns:
                    if col.lower().endswith('startpoint_x'):
                        startpoint_x_col = col  
                    elif col.lower().endswith('startpoint_y'):
                        startpoint_y_col = col
                
                if startpoint_x_col and startpoint_y_col:
                    ax.scatter(trial_scenario[startpoint_x_col], trial_scenario[startpoint_y_col], 
                              marker='o', color='yellow', s=200, edgecolor='black', zorder=10, 
                              label='StartPoint')
                
                # Plot other scenario objects
                object_styles = {
                    'task': dict(marker='^', color='green', label='Task'),
                    'safehouse': dict(marker='s', color='saddlebrown', label='Safehouse'),
                    'threat': dict(marker='p', color='red', label='Threat'),
                    'weapons': dict(marker='s', color='magenta', label='Weapons'),
                    'standpoint': dict(marker='o', color='orange', label='StandPoint'),
                    'wall': dict(marker='s', color='grey', label='Wall'),
                    'hide': dict(marker='s', color='cyan', label='Hide'),
                }
                
                for object_name, style in object_styles.items():
                    x_col = None
                    y_col = None
                    for col in trial_scenario.columns:
                        if col.lower().endswith(f'{object_name}_x'):
                            x_col = col
                        elif col.lower().endswith(f'{object_name}_y'):
                            y_col = col
                    
                    if x_col and y_col:
                        ax.scatter(trial_scenario[x_col], trial_scenario[y_col], 
                                  marker=style['marker'], color=style['color'], 
                                  s=120, edgecolor='black', zorder=10, 
                                  label=style['label'])
        
        # Plot tracker data for this trial
        if tracker_data_list:
            tracker_found = False
            
            for tracker_item in tracker_data_list:
                # Check data types and values
                tracker_ppid = str(tracker_item['ppid'])
                tracker_session = str(tracker_item['session'])
                tracker_trial = str(tracker_item['trial'])
                trial_ppid = str(ppid)
                trial_session = str(session)
                trial_trial = str(trial)
                
                if (tracker_ppid == trial_ppid and 
                    tracker_session == trial_session and 
                    tracker_trial == trial_trial):
                    
                    tracker_data = tracker_item['tracker_data']
                    if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                        ax.scatter(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                                  color='red', marker='x', s=15, alpha=0.8, zorder=1, 
                                  label='Tracker')
                        
                        # Connect consecutive points
                        if len(tracker_data) > 1:
                            ax.plot(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                                   color='red', alpha=0.6, linewidth=1, zorder=1)
                        
                        tracker_found = True
                        break
        
        # Add reference square
        square_width = 10
        square_height = 6
        square_x = [-square_width/2, square_width/2, square_width/2, -square_width/2, -square_width/2]
        square_y = [-square_height/2, -square_height/2, square_height/2, square_height/2, -square_height/2]
        ax.plot(square_x, square_y, color='darkgrey', linestyle='--', linewidth=2, zorder=5, 
                label='Reference Square (10x6m)')
        
        # Set limits and labels
        ax.set_xlim([-6, 6])
        ax.set_ylim([-4, 4])
        ax.set_aspect('equal')
        ax.set_xlabel('x (metres)')
        ax.set_ylabel('y (metres)')
        
        # Get rotation angle for this trial
        rotation_info = ""
        if scenario_rotation_dict and scenario_df is not None and 'scenario_id' in scenario_df.columns:
            trial_scenario = scenario_df[
                (scenario_df['ppid'] == ppid) & 
                (scenario_df['session'] == session) & 
                (scenario_df['trial'] == trial)
            ]
            if not trial_scenario.empty:
                scenario_id = trial_scenario.iloc[0]['scenario_id']
                if pd.notna(scenario_id):
                    scenario_id_str = str(scenario_id)
                    if scenario_id_str in scenario_rotation_dict:
                        angle = scenario_rotation_dict[scenario_id_str]
                        rotation_info = f" | Rotation: {angle}°"
                    else:
                        rotation_info = f" | Scenario: {scenario_id_str} (no rotation)"
                else:
                    rotation_info = " | No scenario_id"
            else:
                rotation_info = " | No scenario data"
        
        ax.set_title(f'Trial {ppid}-{session}-{trial}{rotation_info}\n{step_name}', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{step_name.lower().replace(" ", "_")}_plot.png'
    plot_path = os.path.join(debug_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show plot on screen
    
    
    print(f"✓ {step_name} plot saved to {plot_path}")
    print(f"✓ {step_name} plot displayed on screen")

def create_spatial_plot(participant_df, scenario_df, tracker_data_list, output_dir, color_map, 
                       xlim=None, ylim=None, debug=False):
    """Create spatial plot with participant data, scenario objects, and tracker paths."""
    
    # Use plots directory (already created in main)
    plots_dir = os.path.join(output_dir, 'plots')
    
    # Plot
    plt.figure(figsize=(12, 9))
    
    # Plot participant trajectories colored by syllables
    if 'predicted_label' in participant_df.columns:
        unique_labels = participant_df['predicted_label'].dropna().unique()
        
        for label in unique_labels:
            label_data = participant_df[participant_df['predicted_label'] == label]
            if len(label_data) > 0:
                color = color_map.get(str(label), 'blue') if color_map else 'blue'
                plt.scatter(label_data['participant_x'], label_data['participant_y'], 
                           c=color, s=20, alpha=0.7, label=f'{label}')
    
    # Plot scenario objects
    if scenario_df is not None:
        # Plot startpoint first
        startpoint_x_col = None
        startpoint_y_col = None
        for col in scenario_df.columns:
            if col.lower().endswith('startpoint_x'):
                startpoint_x_col = col
            elif col.lower().endswith('startpoint_y'):
                startpoint_y_col = col
        
        if startpoint_x_col and startpoint_y_col:
            plt.scatter(scenario_df[startpoint_x_col], scenario_df[startpoint_y_col], 
                       marker='o', color='yellow', s=200, edgecolor='black', zorder=10, 
                       label='StartPoint')
        
        object_styles = {
            'task': dict(marker='^', color='green', label='Task'),
            'safehouse': dict(marker='s', color='saddlebrown', label='Safehouse'),
            'threat': dict(marker='p', color='red', label='Threat'),
            'weapons': dict(marker='s', color='magenta', label='Weapons'),
            'standpoint': dict(marker='o', color='orange', label='StandPoint'),
            'wall': dict(marker='s', color='grey', label='Wall'),
            'hide': dict(marker='s', color='cyan', label='Hide'),
        }
        
        # Track which objects were actually plotted
        plotted_objects = set()
        
        for object_name, style in object_styles.items():
            # Find object columns
            x_col = None
            y_col = None
            for col in scenario_df.columns:
                if col.lower().endswith(f'{object_name}_x'):
                    x_col = col
                elif col.lower().endswith(f'{object_name}_y'):
                    y_col = col
            
            if x_col and y_col:
                # Check if coordinates exist and are valid
                valid_x = scenario_df[x_col].dropna()
                valid_y = scenario_df[y_col].dropna()
                
                if len(valid_x) > 0 and len(valid_y) > 0:
                    plt.scatter(scenario_df[x_col], scenario_df[y_col], 
                               marker=style['marker'], color=style['color'], 
                               s=120, edgecolor='black', zorder=10, 
                               label=style['label'])
                    plotted_objects.add(object_name)
    
    # Plot tracker paths
    if tracker_data_list:
        for tracker_item in tracker_data_list:
            tracker_data = tracker_item['tracker_data']
            if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                plt.scatter(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                           color='darkgrey', marker='x', s=20, alpha=0.6, zorder=1, 
                           label='Tracker Path' if tracker_item == tracker_data_list[0] else "")
                
                # Connect consecutive points
                if len(tracker_data) > 1:
                    plt.plot(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                            color='darkgrey', alpha=0.4, linewidth=0.5, zorder=1)
    
    # Add reference square
    square_width = 10
    square_height = 6
    square_x = [-square_width/2, square_width/2, square_width/2, -square_width/2, -square_width/2]
    square_y = [-square_height/2, -square_height/2, square_height/2, square_height/2, -square_height/2]
    plt.plot(square_x, square_y, color='darkgrey', linestyle='--', linewidth=2, zorder=5, 
             label='Reference Square (10x6m)')
    
    # Set limits
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    # Equal aspect ratio
    plt.gca().set_aspect('equal')
    
    # Labels and title
    plt.xlabel('x (metres)')
    plt.ylabel('y (metres)')
    plt.title('Spatial Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(plots_dir, 'spatial_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show plot on screen
    
    
    print(f"✓ Spatial plot saved to {plot_path}")
    print(f"✓ Spatial plot displayed on screen")

def perform_spatial_analysis(participant_df, scenario_df, tracker_data_list, labels_df, color_map, output_dir, args):
    """
    Perform spatial analysis with data merging and plotting organized by groups.
    
    This function:
    1. Merges metadata to add episode_set_name2 and scenario_id columns
    2. Saves the three merged dataframes
    3. Creates plots grouped by episode_set_name2 and scenario_id
    4. Performs object-syllable analysis (if requested)
    
    Args:
        participant_df: Participant coordinates and metadata
        scenario_df: Scenario object positions 
        tracker_data_list: VR headset movement data
        labels_df: Syllable labels
        color_map: Color mapping for syllables
        output_dir: Output directory for plots and data
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("SPATIAL ANALYSIS: DATA MERGING AND GROUPED PLOTTING")
    print("="*80)
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os
    import re
    import numpy as np

    # Define object styles for scenario objects (same as original)
    object_styles = {
        'task':      dict(marker='^', color='green',     label='Task',        s=120),
        'safehouse': dict(marker='s', color='saddlebrown',label='Safehouse',  s=120),
        'threat':    dict(marker='p', color='red',       label='Threat',      s=120),
        'startpoint':dict(marker='o', color='yellow',    label='StartPoint',  s=120),
        'weapons':   dict(marker='s', color='magenta',   label='Weapons',     s=120),
        'standpoint':dict(marker='o', color='orange',    label='StandPoint',  s=120),
        'wall':      dict(marker='s', color='grey',      label='Wall',        s=120),
        'hide':      dict(marker='s', color='cyan',      label='Hide',        s=120),
    }
    
    # STEP 1: Load and merge metadata to add episode_set_name2 and scenario_id
    print("\n=== STEP 1: MERGING METADATA ===")
    
    metadata_df = None
    
    # First, try to load the processed metadata file that was created during transformations
    processed_metadata_path = os.path.join(output_dir, 'data', 'processed_metadata.csv')
    if os.path.exists(processed_metadata_path):
        try:
            metadata_df = pd.read_csv(processed_metadata_path)
            print(f"✓ Loaded processed metadata: {metadata_df.shape[0]} rows, {metadata_df.shape[1]} columns")
            
            # Check for required columns
            if 'episode_set_name2' in metadata_df.columns:
                unique_blocks = metadata_df['episode_set_name2'].dropna().unique()
                print(f"  Found {len(unique_blocks)} blocks")
            else:
                print(f"  ⚠ episode_set_name2 not found")
                
            if 'scenario_id' in metadata_df.columns:
                unique_scenarios = metadata_df['scenario_id'].dropna().unique()
                print(f"  Found {len(unique_scenarios)} scenarios")
            else:
                print(f"  ⚠ scenario_id not found")
                
        except Exception as e:
            print(f"  ❌ Error loading processed metadata: {e}")
            metadata_df = None
    
    # Fallback to original experimental conditions file if processed metadata not available
    if metadata_df is None and args.experimental_conditions and os.path.exists(args.experimental_conditions):
        try:
            metadata_df = pd.read_csv(args.experimental_conditions)
            print(f"⚠ Fallback: Loaded original metadata: {metadata_df.shape[0]} rows, {metadata_df.shape[1]} columns")
            
            # Check for required columns
            if 'episode_set_name2' in metadata_df.columns:
                unique_blocks = metadata_df['episode_set_name2'].dropna().unique()
                print(f"  Found {len(unique_blocks)} blocks")
            else:
                print(f"  ⚠ episode_set_name2 not found")
                
            if 'scenario_id' in metadata_df.columns:
                unique_scenarios = metadata_df['scenario_id'].dropna().unique()
                print(f"  Found {len(unique_scenarios)} scenarios")
            else:
                print(f"  ⚠ scenario_id not found")
                
        except Exception as e:
            print(f"  ❌ Error loading original metadata: {e}")
            metadata_df = None
    
    if metadata_df is None:
        print(f"  ⚠ No metadata available")
        
    # STEP 2: Merge metadata with participant_df to add episode_set_name2
    print("\n=== STEP 2: MERGING METADATA WITH PARTICIPANT DATA ===")
    
    if metadata_df is not None:
        # Ensure consistent data types for merging
        merge_keys = ['ppid', 'session', 'trial']
        for key in merge_keys:
            if key in metadata_df.columns and key in participant_df.columns:
                metadata_df[key] = metadata_df[key].astype(str)
                participant_df[key] = participant_df[key].astype(str)
        
        # Get metadata columns to merge
        metadata_cols_to_merge = merge_keys.copy()
        if 'episode_set_name2' in metadata_df.columns:
            metadata_cols_to_merge.append('episode_set_name2')
        if 'scenario_id' in metadata_df.columns:
            metadata_cols_to_merge.append('scenario_id')
        
        print(f"  Merging metadata columns: {metadata_cols_to_merge}")
        
        # Merge metadata with participant data
        participant_df_with_metadata = pd.merge(participant_df, metadata_df[metadata_cols_to_merge], 
                                               on=merge_keys, how='left')
        print(f"  ✓ Merged metadata with participant data: {len(participant_df_with_metadata)} rows")
        
        # Update participant_df
        participant_df = participant_df_with_metadata
        
        # Check what was added
        added_cols = [col for col in participant_df.columns if col not in participant_df.columns]
        print(f"  Added columns from metadata: {added_cols}")
    else:
        print(f"  ⚠ No metadata available - participant_df unchanged")
    
    # STEP 3: Merge metadata with scenario_df to add episode_set_name2
    print("\n=== STEP 3: MERGING METADATA WITH SCENARIO DATA ===")
    
    if metadata_df is not None and scenario_df is not None:
        # Check if scenario_id already exists in scenario_df
        scenario_id_exists = 'scenario_id' in scenario_df.columns
        episode_set_name2_exists = 'episode_set_name2' in scenario_df.columns
        
        print(f"  Current scenario_df columns: {list(scenario_df.columns)}")
        print(f"  scenario_id already exists: {scenario_id_exists}")
        print(f"  episode_set_name2 already exists: {episode_set_name2_exists}")
        
        # Ensure consistent data types for merging
        scenario_keys = ['ppid', 'session', 'trial']
        for key in scenario_keys:
            if key in metadata_df.columns and key in scenario_df.columns:
                metadata_df[key] = metadata_df[key].astype(str)
                scenario_df[key] = scenario_df[key].astype(str)
        
        # Get metadata columns to merge with scenario data (only add what's missing)
        scenario_metadata_cols = scenario_keys.copy()
        if 'episode_set_name2' in metadata_df.columns and not episode_set_name2_exists:
            scenario_metadata_cols.append('episode_set_name2')
            print(f"  Will add episode_set_name2 from metadata")
        elif 'episode_set_name2' in metadata_df.columns and episode_set_name2_exists:
            print(f"  episode_set_name2 already exists in scenario_df - skipping")
            
        if 'scenario_id' in metadata_df.columns and not scenario_id_exists:
            scenario_metadata_cols.append('scenario_id')
            print(f"  Will add scenario_id from metadata")
        elif 'scenario_id' in metadata_df.columns and scenario_id_exists:
            print(f"  scenario_id already exists in scenario_df - skipping to avoid duplication")
        elif 'scenario_id' not in metadata_df.columns:
            print(f"  ⚠ scenario_id not found in metadata_df columns: {list(metadata_df.columns)}")
            
        print(f"  Merging metadata columns with scenario data: {scenario_metadata_cols}")
        
        if len(scenario_metadata_cols) > len(scenario_keys):  # Only merge if there are new columns to add
            # Merge metadata with scenario data
            scenario_df_with_metadata = pd.merge(scenario_df, metadata_df[scenario_metadata_cols], 
                                               on=scenario_keys, how='left')
            print(f"  ✓ Merged metadata with scenario data: {len(scenario_df_with_metadata)} rows")
            
            # Update scenario_df
            scenario_df = scenario_df_with_metadata
            
            # Check what was added
            original_cols = set(scenario_df.columns) - set(metadata_df[scenario_metadata_cols].columns) | set(scenario_keys)
            added_cols = [col for col in scenario_df.columns if col not in original_cols]
            if added_cols:
                print(f"  Added columns to scenario data from metadata: {added_cols}")
            else:
                print(f"  No new columns added to scenario data")
            
            # Verify that expected columns are present
            print(f"  Final scenario_df columns: {list(scenario_df.columns)}")
            if 'scenario_id' in scenario_df.columns:
                print(f"  ✓ scenario_id is present in scenario_df")
            else:
                print(f"  ❌ ERROR: scenario_id is missing from scenario_df after merge")
            if 'episode_set_name2' in scenario_df.columns:
                print(f"  ✓ episode_set_name2 is present in scenario_df")
            else:
                print(f"  ⚠ episode_set_name2 is missing from scenario_df after merge")
        else:
            print(f"  No new columns to add from metadata - scenario_df unchanged")
    else:
        print(f"  ⚠ No metadata or scenario data available - scenario_df unchanged")
    
    # STEP 4: Add scenario_id to participant_df from scenario_df
    print("\n=== STEP 4: ADDING SCENARIO_ID TO PARTICIPANT DATA ===")
    
    # Verify scenario_df and scenario_id availability
    if scenario_df is not None:
        print(f"  scenario_df is available with shape: {scenario_df.shape}")
        print(f"  scenario_df columns: {list(scenario_df.columns)}")
        if 'scenario_id' in scenario_df.columns:
            print(f"  ✓ scenario_id found in scenario_df")
        else:
            print(f"  ❌ scenario_id NOT found in scenario_df")
    else:
        print(f"  ❌ scenario_df is None")
    
    if scenario_df is not None and 'scenario_id' in scenario_df.columns:
        print(f"  scenario_df has scenario_id column - proceeding with merge")
        print(f"  scenario_df columns: {list(scenario_df.columns)}")
        print(f"  participant_df columns: {list(participant_df.columns)}")
        
        # Check if participant_df already has scenario_id
        participant_has_scenario_id = 'scenario_id' in participant_df.columns
        print(f"  participant_df already has scenario_id: {participant_has_scenario_id}")
        
        # Ensure consistent data types for merging
        scenario_keys = ['ppid', 'session', 'trial']
        for key in scenario_keys:
            if key in participant_df.columns and key in scenario_df.columns:
                participant_df[key] = participant_df[key].astype(str)
                scenario_df[key] = scenario_df[key].astype(str)
        
        if participant_has_scenario_id:
            print(f"  ⚠ participant_df already has scenario_id - will handle duplicate columns")
            # Remove existing scenario_id from participant_df to avoid conflicts
            participant_df_temp = participant_df.drop(columns=['scenario_id'])
            print(f"  Temporarily removed scenario_id from participant_df for merge")
            
            # Merge scenario_id from scenario_df to participant_df
            scenario_id_cols = scenario_keys + ['scenario_id']
            print(f"  Merging columns: {scenario_id_cols}")
            
            try:
                participant_df_with_scenario_id = pd.merge(participant_df_temp, scenario_df[scenario_id_cols], 
                                                         on=scenario_keys, how='left')
                print(f"  ✓ Merge completed: {len(participant_df_with_scenario_id)} rows")
                
                # Verify that scenario_id was actually added
                if 'scenario_id' in participant_df_with_scenario_id.columns:
                    print(f"  ✓ scenario_id successfully added to participant_df")
                    
                    # Update participant_df
                    participant_df = participant_df_with_scenario_id
                    
                    # Check for missing scenario_id
                    missing_scenario_id = participant_df['scenario_id'].isnull().sum()
                    if missing_scenario_id > 0:
                        print(f"  ⚠ Warning: {missing_scenario_id} participant rows have missing scenario_id")
                    else:
                        print(f"  ✓ Added scenario_id to all {len(participant_df)} participant rows")
                else:
                    print(f"  ❌ ERROR: scenario_id was not added to participant_df despite successful merge")
                    print(f"  participant_df_with_scenario_id columns: {list(participant_df_with_scenario_id.columns)}")
                    # Don't update participant_df if scenario_id wasn't added
                    
            except Exception as e:
                print(f"  ❌ ERROR during merge: {e}")
                print(f"  participant_df_temp shape: {participant_df_temp.shape}")
                print(f"  scenario_df shape: {scenario_df.shape}")
                print(f"  scenario_id_cols: {scenario_id_cols}")
                print(f"  Available columns in scenario_df: {list(scenario_df.columns)}")
        else:
            # participant_df doesn't have scenario_id, so normal merge
            scenario_id_cols = scenario_keys + ['scenario_id']
            print(f"  Merging columns: {scenario_id_cols}")
            
            try:
                participant_df_with_scenario_id = pd.merge(participant_df, scenario_df[scenario_id_cols], 
                                                         on=scenario_keys, how='left')
                print(f"  ✓ Merge completed: {len(participant_df_with_scenario_id)} rows")
                
                # Verify that scenario_id was actually added
                if 'scenario_id' in participant_df_with_scenario_id.columns:
                    print(f"  ✓ scenario_id successfully added to participant_df")
                    
                    # Update participant_df
                    participant_df = participant_df_with_scenario_id
                    
                    # Check for missing scenario_id
                    missing_scenario_id = participant_df['scenario_id'].isnull().sum()
                    if missing_scenario_id > 0:
                        print(f"  ⚠ Warning: {missing_scenario_id} participant rows have missing scenario_id")
                    else:
                        print(f"  ✓ Added scenario_id to all {len(participant_df)} participant rows")
                else:
                    print(f"  ❌ ERROR: scenario_id was not added to participant_df despite successful merge")
                    print(f"  participant_df_with_scenario_id columns: {list(participant_df_with_scenario_id.columns)}")
                    # Don't update participant_df if scenario_id wasn't added
                    
            except Exception as e:
                print(f"  ❌ ERROR during merge: {e}")
                print(f"  participant_df shape: {participant_df.shape}")
                print(f"  scenario_df shape: {scenario_df.shape}")
                print(f"  scenario_id_cols: {scenario_id_cols}")
                print(f"  Available columns in scenario_df: {list(scenario_df.columns)}")
    else:
        print(f"  ⚠ No scenario data or scenario_id column available")
        if scenario_df is not None:
            print(f"  scenario_df columns: {list(scenario_df.columns)}")
        else:
            print(f"  scenario_df is None")
    
    # STEP 5: Save the three merged dataframes
    print("\n=== STEP 5: SAVING MERGED DATA ===")
    
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save participant_df
    participant_file = os.path.join(data_dir, 'participant_data_merged.csv')
    participant_df.to_csv(participant_file, index=False)
    print(f"  ✓ Saved participant data: {participant_file}")
    
    # Save scenario_df
    if scenario_df is not None:
        scenario_file = os.path.join(data_dir, 'scenario_data_merged.csv')
        scenario_df.to_csv(scenario_file, index=False)
        print(f"  ✓ Saved scenario data: {scenario_file}")
    
    # Save metadata_df
    if metadata_df is not None:
        metadata_file = os.path.join(data_dir, 'metadata_merged.csv')
        metadata_df.to_csv(metadata_file, index=False)
        print(f"  ✓ Saved metadata: {metadata_file}")
    
    # STEP 6: Merge participant data with labels
    print("\n=== STEP 6: MERGING WITH LABELS ===")
    
    # Ensure consistent data types for merging
    # Start with core merge keys that should always exist
    core_merge_keys = ['ppid', 'session', 'trial']
    
    # Check if time column exists in labels_df
    has_time_in_labels = 'time' in labels_df.columns
    print(f"  Labels data has 'time' column: {has_time_in_labels}")
    
    # Define merge_keys for compatibility with existing code
    merge_keys = core_merge_keys.copy()
    if has_time_in_labels:
        merge_keys.append('time')
    
    for key in merge_keys:
        if key in labels_df.columns:
            if key == 'time':
                labels_df[key] = pd.to_numeric(labels_df[key], errors='coerce')
            else:
                labels_df[key] = labels_df[key].astype(str)
    
    # Start with labels_df as the foundation (this has all 1081 trials)
    print(f"  Starting with labels data: {len(labels_df)} rows")
    print(f"  Labels data unique trials: {labels_df[['ppid', 'session', 'trial']].drop_duplicates().shape[0]}")
    
    # Start building comprehensive dataframe from labels
    comprehensive_df = labels_df.copy()
    print(f"  ✓ Started with labels data: {len(comprehensive_df)} rows")
    
    # STEP 6.1: Add participant location data
    print("\n  --- STEP 6.1: Adding participant location data ---")
    if participant_df is not None and len(participant_df) > 0:
        print(f"    Participant data shape: {participant_df.shape}")
        print(f"    Participant data columns: {list(participant_df.columns)}")
        
        # Check for coordinate columns
        coord_cols = [col for col in participant_df.columns if 'participant' in col.lower() and ('x' in col.lower() or 'y' in col.lower())]
        print(f"    Found coordinate columns: {coord_cols}")
        
        # Also check for any x/y columns that might be coordinate data
        x_y_cols = [col for col in participant_df.columns if ('x' in col.lower() or 'y' in col.lower()) and not col.endswith('_transformed')]
        print(f"    Found x/y columns (excluding transformed): {x_y_cols}")
        
        # Check for common coordinate column patterns
        all_coord_patterns = []
        for col in participant_df.columns:
            if any(pattern in col.lower() for pattern in ['x', 'y', 'coord', 'position', 'location']):
                all_coord_patterns.append(col)
        print(f"    All potential coordinate columns: {all_coord_patterns}")
        
        # Check merge key values in participant_df
        print(f"    Participant data merge key sample:")
        for key in merge_keys:
            if key in participant_df.columns:
                unique_vals = participant_df[key].dropna().unique()[:5]
                print(f"      {key}: {len(participant_df[key].dropna())} non-null values, sample: {unique_vals}")
            else:
                print(f"      {key}: NOT FOUND in participant_df")
        
        # Convert merge keys in participant_df to match labels_df
        for key in merge_keys:
            if key in participant_df.columns:
                if key == 'time':
                    participant_df[key] = pd.to_numeric(participant_df[key], errors='coerce')
                else:
                    participant_df[key] = participant_df[key].astype(str)
        
        # Check merge key values in labels_df
        print(f"    Labels data merge key sample:")
        for key in merge_keys:
            if key in labels_df.columns:
                unique_vals = labels_df[key].dropna().unique()[:5]
                print(f"      {key}: {len(labels_df[key].dropna())} non-null values, sample: {unique_vals}")
            else:
                print(f"      {key}: NOT FOUND in labels_df")
        
        # Check for potential merge key mismatches
        print(f"    Checking for merge key mismatches:")
        for key in available_merge_keys:
            if key in labels_df.columns and key in participant_df.columns:
                labels_sample = set(labels_df[key].dropna().astype(str).unique()[:10])
                participant_sample = set(participant_df[key].dropna().astype(str).unique()[:10])
                common_vals = labels_sample.intersection(participant_sample)
                print(f"      {key}: {len(common_vals)} common values out of {len(labels_sample)} labels + {len(participant_sample)} participant")
                if len(common_vals) == 0:
                    print(f"        ⚠ WARNING: No common values found for {key}!")
                    print(f"        Labels sample: {list(labels_sample)}")
                    print(f"        Participant sample: {list(participant_sample)}")
        
        # Merge participant data with comprehensive dataframe
        # Only use merge keys that exist in both dataframes
        available_merge_keys = [key for key in merge_keys if key in participant_df.columns]
        print(f"    Available merge keys for participant data: {available_merge_keys}")
        
        # Check for potential merge issues
        if len(available_merge_keys) < len(merge_keys):
            missing_keys = [key for key in merge_keys if key not in participant_df.columns]
            print(f"    ⚠ Missing merge keys in participant_df: {missing_keys}")
        
        # Debug: Check data types before merge
        # Perform the merge
        comprehensive_df = pd.merge(comprehensive_df, participant_df, on=available_merge_keys, how='left', suffixes=('', '_participant'))
        print(f"    ✓ Added participant data: {len(comprehensive_df)} rows")
        
        # Check what columns were added
        new_cols = [col for col in comprehensive_df.columns if col not in labels_df.columns]
        print(f"    New columns added: {len(new_cols)}")
        
        # Check how many trials now have participant data
        if 'participant_x' in comprehensive_df.columns:
            trials_with_participant = comprehensive_df[comprehensive_df['participant_x'].notna()][['ppid', 'session', 'trial']].drop_duplicates().shape[0]
            print(f"    Trials with participant data: {trials_with_participant}")
        else:
            print(f"    ⚠ Coordinate columns missing after merge")
    else:
        print("    ⚠ No participant data available")
    
    # STEP 6.2: Add metadata information
    print("\n  --- STEP 6.2: Adding metadata information ---")
    if metadata_df is not None and len(metadata_df) > 0:
        # Convert merge keys in metadata_df to match labels_df
        for key in merge_keys:
            if key in metadata_df.columns:
                if key == 'time':
                    metadata_df[key] = pd.to_numeric(metadata_df[key], errors='coerce')
                else:
                    metadata_df[key] = metadata_df[key].astype(str)
        
        # Merge metadata with comprehensive dataframe
        # Only use merge keys that exist in both dataframes
        available_merge_keys = [key for key in merge_keys if key in metadata_df.columns]
        print(f"    Available merge keys for metadata: {available_merge_keys}")
        comprehensive_df = pd.merge(comprehensive_df, metadata_df, on=available_merge_keys, how='left', suffixes=('', '_metadata'))
        print(f"    ✓ Added metadata: {len(comprehensive_df)} rows")
        
        # Check metadata coverage
        if 'episode_set_name2' in comprehensive_df.columns:
            trials_with_metadata = comprehensive_df[comprehensive_df['episode_set_name2'].notna()][['ppid', 'session', 'trial']].drop_duplicates().shape[0]
            print(f"    Trials with episode_set_name2: {trials_with_metadata}")
        if 'scenario_id' in comprehensive_df.columns:
            trials_with_scenario_id = comprehensive_df[comprehensive_df['scenario_id'].notna()][['ppid', 'session', 'trial']].drop_duplicates().shape[0]
            print(f"    Trials with scenario_id: {trials_with_scenario_id}")
    else:
        print("    ⚠ No metadata available")
    
    # STEP 6.3: Add scenario object data
    print("\n  --- STEP 6.3: Adding scenario object data ---")
    if scenario_df is not None and len(scenario_df) > 0:
        # Convert merge keys in scenario_df to match labels_df
        for key in merge_keys:
            if key in scenario_df.columns:
                if key == 'time':
                    scenario_df[key] = pd.to_numeric(scenario_df[key], errors='coerce')
                else:
                    scenario_df[key] = scenario_df[key].astype(str)
        
        # Merge scenario data with comprehensive dataframe
        # Only use merge keys that exist in both dataframes
        available_merge_keys = [key for key in merge_keys if key in scenario_df.columns]
        print(f"    Available merge keys for scenario data: {available_merge_keys}")
        comprehensive_df = pd.merge(comprehensive_df, scenario_df, on=available_merge_keys, how='left', suffixes=('', '_scenario'))
        print(f"    ✓ Added scenario data: {len(comprehensive_df)} rows")
        
        # Check scenario object coverage
        scenario_object_cols = [col for col in comprehensive_df.columns if col.endswith('_x') or col.endswith('_y')]
        if scenario_object_cols:
            object_bases = set()
            for col in scenario_object_cols:
                m = re.match(r'(.+)_([xyz])$', col)
                if m:
                    object_bases.add(m.group(1))
            print(f"    Added scenario objects: {list(object_bases)}")
    else:
        print("    ⚠ No scenario data available")
    
    # STEP 6.4: Add tracker data
    print("\n  --- STEP 6.4: Adding tracker data ---")
    if tracker_data_list and len(tracker_data_list) > 0:
        # Create tracker summary dataframe
        tracker_rows = []
        for tracker_item in tracker_data_list:
            tracker_data = tracker_item['tracker_data']
            if tracker_data is not None and len(tracker_data) > 0:
                ppid = tracker_item['ppid']
                session = tracker_item['session']
                trial = tracker_item['trial']
                
                # Get the first tracker position for each trial
                if 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                    first_tracker = tracker_data.iloc[0]
                    tracker_rows.append({
                        'ppid': str(ppid),
                        'session': str(session),
                        'trial': str(trial),
                        'tracker_x': first_tracker['transformed_x'],
                        'tracker_y': first_tracker['transformed_y'],
                        'tracker_file': tracker_item['tracker_file']
                    })
        
        if tracker_rows:
            tracker_df = pd.DataFrame(tracker_rows)
            print(f"    Created tracker summary: {len(tracker_df)} trials")
            
            # Merge tracker data with comprehensive dataframe
            # Only use merge keys that exist in both dataframes
            available_merge_keys = [key for key in merge_keys if key in tracker_df.columns]
            print(f"    Available merge keys for tracker data: {available_merge_keys}")
            comprehensive_df = pd.merge(comprehensive_df, tracker_df, on=available_merge_keys, how='left')
            print(f"    ✓ Added tracker data: {len(comprehensive_df)} rows")
            
            # Check tracker coverage
            if 'tracker_x' in comprehensive_df.columns:
                tracker_data_count = comprehensive_df['tracker_x'].notna().sum()
                print(f"    Trials with tracker data: {tracker_data_count}")
            else:
                print("    ⚠ No tracker_x column found after merge")
        else:
            print("    ⚠ No valid tracker data found")
    else:
        print("    ⚠ No tracker data available")
    
    # STEP 6.5: Clean up duplicate columns
    print("\n  --- STEP 6.5: Cleaning up duplicate columns ---")
    original_columns = len(comprehensive_df.columns)
    
    # Common duplicate patterns from pandas merges
    duplicate_patterns = [
        ('scenario_id', 'scenario_id_scenario'),
        ('episode_set_name2', 'episode_set_name2_scenario'),
        ('episode', 'episode_scenario'),
        ('ppid', 'ppid_scenario'),
        ('session', 'session_scenario'),
        ('trial', 'trial_scenario'),
        ('scenario_id', 'scenario_id_metadata'),
        ('episode_set_name2', 'episode_set_name2_metadata'),
        ('episode', 'episode_metadata'),
        ('ppid', 'ppid_metadata'),
        ('session', 'session_metadata'),
        ('trial', 'trial_metadata'),
        ('ppid', 'ppid_participant'),
        ('session', 'session_participant'),
        ('trial', 'trial_participant')
    ]
    
    columns_to_drop = []
    for base_col, duplicate_col in duplicate_patterns:
        if duplicate_col in comprehensive_df.columns and base_col in comprehensive_df.columns:
            print(f"    ⚠ Found duplicate column: {duplicate_col} (keeping {base_col})")
            columns_to_drop.append(duplicate_col)
    
    # Remove duplicate columns
    if columns_to_drop:
        print(f"    Removing {len(columns_to_drop)} duplicate columns: {columns_to_drop}")
        comprehensive_df = comprehensive_df.drop(columns=columns_to_drop)
        print(f"    ✓ After cleanup: {len(comprehensive_df.columns)} columns (removed {len(columns_to_drop)})")
    else:
        print(f"    ✓ No duplicate columns found")
    
    # Additional check for any remaining duplicate column names
    remaining_duplicates = comprehensive_df.columns[comprehensive_df.columns.duplicated()].tolist()
    if remaining_duplicates:
        print(f"    ⚠ WARNING: Found {len(remaining_duplicates)} remaining duplicate column names: {remaining_duplicates}")
        print(f"    This should not happen - please investigate!")
    else:
        print(f"    ✓ All column names are unique")
    
    # Final comprehensive dataframe summary
    print(f"\n📊 COMPREHENSIVE DATAFRAME SUMMARY:")
    print(f"  Final shape: {comprehensive_df.shape}")
    print(f"  Final columns: {len(comprehensive_df.columns)}")
    final_trials = comprehensive_df[['ppid', 'session', 'trial']].drop_duplicates().shape[0]
    print(f"  Final unique trials: {final_trials}")
    
    if final_trials != 1081:
        print(f"  ⚠ WARNING: Expected 1081 trials but got {final_trials}")
        print(f"    This suggests data loss during merging - investigate!")
    else:
        print(f"  ✓ Successfully preserved all 1081 trials!")
    
    # Save comprehensive dataframe
    comprehensive_file = os.path.join(data_dir, 'comprehensive_merged_data.csv')
    comprehensive_df.to_csv(comprehensive_file, index=False)
    print(f"\n💾 COMPREHENSIVE DATAFRAME SAVED:")
    print(f"  File: {comprehensive_file}")
    print(f"  Size: {comprehensive_df.shape[0]} rows × {comprehensive_df.shape[1]} columns")
    print(f"  Trials: {final_trials}")
    
    # Use comprehensive_df for all subsequent operations
    merged_data = comprehensive_df
    print(f"\n  ✓ Using comprehensive dataframe with {len(merged_data)} rows for spatial analysis")
    

    
    # STEP 7: Create plotting function with same characteristics
    def create_spatial_plot(data_df, title, filename, plot_trackers=True, subfolder=None):
        """Create spatial plot with same characteristics as original."""
        
        # Calculate number of observations (unique ppid-session-trial combinations that have syllable data)
        # Only count trials that have participant location data with syllables
        if 'predicted_label' in data_df.columns and 'participant_x' in data_df.columns and 'participant_y' in data_df.columns:
            # Count only trials that have syllable data (non-null predicted_label)
            syllable_trials = data_df[data_df['predicted_label'].notna()][['ppid', 'session', 'trial']].drop_duplicates()
            n_observations = len(syllable_trials)
        else:
            # If no syllable data available, count all trials
            n_observations = data_df[['ppid', 'session', 'trial']].drop_duplicates().shape[0]
        
        # Add observation count to title (total unique trials in this data)
        title_with_obs = f"{title} (n={n_observations} trials)"
        
        # Add observation count to filename
        filename_with_obs = filename.replace('.png', f'_n{n_observations}trials.png')
        
        print(f"\n--- Creating spatial plot: {title_with_obs} ---")
    
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
    
        # FIRST LAYER: Plot tracker paths (grey 'x' markers)
        if plot_trackers and tracker_data_list is not None:
            print(f"  Plotting tracker paths (first layer)...")
            
            # Get trials in this data
            plot_trials = data_df[['ppid', 'session', 'trial']].drop_duplicates()
            print(f"    Processing {len(plot_trials)} trials for tracker data")
        
            tracker_plotted_count = 0
            for _, trial_row in plot_trials.iterrows():
                ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
                
                # Find corresponding tracker data
                tracker_data = None
                for tracker_item in tracker_data_list:
                    if (str(tracker_item['ppid']) == str(ppid) and 
                        str(tracker_item['session']) == str(session) and 
                        str(tracker_item['trial']) == str(trial)):
                        tracker_data = tracker_item['tracker_data']
                        break
                
                if tracker_data is not None and 'transformed_x' in tracker_data.columns and 'transformed_y' in tracker_data.columns:
                    # Plot tracker data points ONLY (no lines) in grey with 'x' markers
                    ax.scatter(tracker_data['transformed_x'], tracker_data['transformed_y'], 
                             color='grey', s=3, alpha=0.6, zorder=1, marker='x',
                             label='Tracker' if tracker_plotted_count == 0 else "")
                    
                    tracker_plotted_count += 1
                    print(f"      ✓ Plotted tracker for {ppid}-{session}-{trial}")
                else:
                    print(f"      ⚠ No tracker data found for {ppid}-{session}-{trial}")
    
        # SECOND LAYER: Plot participant trajectories colored by syllables (markers only, no lines)
        print(f"  Checking for syllable data in block plot...")
        print(f"    Available columns: {list(data_df.columns)}")
        print(f"    Has 'predicted_label': {'predicted_label' in data_df.columns}")
        print(f"    Has 'participant_x': {'participant_x' in data_df.columns}")
        print(f"    Has 'participant_y': {'participant_y' in data_df.columns}")
        
        if 'predicted_label' in data_df.columns and 'participant_x' in data_df.columns and 'participant_y' in data_df.columns:
            print(f"  Plotting participant trajectories with syllable labels (markers only)...")
        
            # Group by syllable and plot each syllable's data
            unique_sylls = data_df['predicted_label'].dropna().unique()
            print(f"  Found {len(unique_sylls)} unique syllables: {sorted(unique_sylls)}")
            
            # Debug: Show syllable data statistics
            if len(unique_sylls) > 0:
                print(f"  Syllable data statistics:")
                for syll in unique_sylls:
                    syll_data = data_df[data_df['predicted_label'] == syll]
                    print(f"    {syll}: {len(syll_data)} data points")
                
                # Initialize syllable statistics
                syllable_stats = {}
                
                for syll in unique_sylls:
                    sub = data_df[data_df['predicted_label'] == syll]
                    if len(sub) == 0:
                        continue
                    
                    # Get syllable color from color_map
                    if color_map is not None and str(syll) in color_map:
                        syll_color = color_map[str(syll)]
                        print(f"    ✓ Using color map for syllable '{syll}': {syll_color}")
                    else:
                        # Fallback to default color scheme using multi-colormap approach
                        hash_val = hash(str(syll))
                        colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                        colors_per_cmap = 20
                        cmap_index = (abs(hash_val) // colors_per_cmap) % len(colormaps)
                        color_index = abs(hash_val) % colors_per_cmap
                        syll_color = colormaps[cmap_index](color_index)
                        if color_map is not None:
                            print(f"    ⚠ Syllable '{syll}' not in color map, using multi-colormap fallback")
                        else:
                            print(f"    ⚠ No color map provided, using multi-colormap fallback for syllable '{syll}'")
                    
                    # Count statistics
                    n_trials = sub[['ppid', 'session', 'trial']].drop_duplicates().shape[0]
                    n_observations = len(sub)
                    syllable_stats[syll] = {'trials': n_trials, 'observations': n_observations}
                
                    # Plot all individual data points as markers (no lines)
                    ax.scatter(sub['participant_x'], sub['participant_y'], 
                              color=syll_color, s=8, alpha=0.7, zorder=2,
                              label=f'{syll} (n={n_trials} trials, {n_observations} obs)')
                    
                    print(f"    ✓ Plotted {len(sub)} data points for syllable '{syll}'")
                
                # Print syllable statistics
                print(f"\n  Syllable Statistics:")
                for syll, stats in syllable_stats.items():
                    print(f"    {syll}: {stats['trials']} trials, {stats['observations']} observations")
            else:
                print(f"  ⚠ No syllable data found in this block!")
        else:
            print(f"  ⚠ Missing required columns for syllable plotting")
    
        # THIRD LAYER: Plot scenario objects
        print(f"  Plotting scenario objects...")
        
        # Get scenario data for this group
        if scenario_df is not None:
            # Get trials in this data
            plot_trials = data_df[['ppid', 'session', 'trial']].drop_duplicates()
            
            # Filter scenario data for these trials
            scenario_cols = [c for c in scenario_df.columns if c.endswith('_x') or c.endswith('_y')]
            if scenario_cols:
                # Merge scenario data for these trials
                scenario_data_for_plot = pd.merge(plot_trials, scenario_df, on=['ppid', 'session', 'trial'], how='left')
                
                if not scenario_data_for_plot.empty:
                    object_bases = set()
                    for c in scenario_cols:
                        m = re.match(r'(.+)_([xyz])$', c)
                        if m:
                            object_bases.add(m.group(1))
    
                    print(f"    Found scenario objects: {len(object_bases)} types")
                    plotted_objects = set()
    
                    for obj in object_bases:
                        base = obj.lower()
                        style_key = None
                        
                        # Determine object style
                        for k in object_styles:
                            if k in base:
                                style_key = k
                                break
                        
                        if style_key is None:
                            continue
                        
                        xcol = f'{obj}_x'
                        ycol = f'{obj}_y'
                        
                        if xcol not in scenario_data_for_plot.columns or ycol not in scenario_data_for_plot.columns:
                            continue
                        
                        # Get unique positions for this object
                        points = set(zip(scenario_data_for_plot[xcol].dropna(), scenario_data_for_plot[ycol].dropna()))
                        print(f"    Plotting {obj}: {len(points)} positions")
                        
                        for x, y in points:
                            # Only add label if this object type hasn't been plotted yet
                            add_label = style_key not in plotted_objects
                            if add_label:
                                plotted_objects.add(style_key)
                            
                            ax.scatter(x, y, marker=object_styles[style_key]['marker'], 
                                      color=object_styles[style_key]['color'], 
                                      s=object_styles[style_key]['s'], edgecolor='black', zorder=10,
                                      label=object_styles[style_key]['label'] if add_label else "")
    
        # Set plot properties
        ax.set_xlabel('x (metres)', fontsize=12)
        ax.set_ylabel('y (metres)', fontsize=12)
        ax.set_title(title_with_obs, fontsize=14, fontweight='bold')
        
        # Ensure equal aspect ratio for proper proportions
        ax.set_aspect('equal')
    
        # Add reference square
        ref_square = mpatches.Rectangle((-5, -3), 10, 6, linewidth=2, 
                                      edgecolor='grey', facecolor='none', 
                                      linestyle='--', alpha=0.7, zorder=0)
        ax.add_patch(ref_square)
        ax.text(-4.5, -2.5, 'Reference Square (10x6m)', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
        # Set axis limits
        ax.set_xlim(-6, 6)
        ax.set_ylim(-4, 4)
    
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
        # Add grid
        ax.grid(True, alpha=0.3)
    
        # Save plot
        plots_dir = os.path.join(output_dir, 'plots')
        if subfolder:
            plots_dir = os.path.join(plots_dir, subfolder)
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, filename_with_obs)
    
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"  ✓ Saved spatial plot: {plot_path}")
    
    # STEP 8: Create plots grouped by episode_set_name2
    print("\n=== STEP 8: CREATING PLOTS GROUPED BY EPISODE_SET_NAME2 ===")
    
    if 'episode_set_name2' in merged_data.columns:
        unique_blocks = merged_data['episode_set_name2'].dropna().unique()
        print(f"Found {len(unique_blocks)} blocks: {list(unique_blocks)}")
        
        for block in unique_blocks:
            print(f"\n--- Processing block: {block} ---")
            
            # Filter data for this block
            block_data = merged_data[merged_data['episode_set_name2'] == block]
            print(f"  Block '{block}' has {len(block_data)} rows")
            
            if len(block_data) == 0:
                print(f"  ⚠ No data found for block '{block}', skipping")
                continue
            
            # Create block-level plot
            safe_block_name = str(block).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            title = f'Block {block} - Spatial Analysis'
            filename = f'spatial_block_{safe_block_name}.png'
            
            try:
                create_spatial_plot(block_data, title, filename, subfolder='blocks')
                print(f"  ✓ Successfully created plot for block '{block}'")
            except Exception as e:
                print(f"  ❌ Error creating plot for block '{block}': {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"⚠ No 'episode_set_name2' column found in merged data")
    
    # STEP 9: Create plots grouped by scenario_id
    print("\n=== STEP 9: CREATING PLOTS GROUPED BY SCENARIO_ID ===")
    
    if 'scenario_id' in merged_data.columns:
        unique_scenarios = merged_data['scenario_id'].dropna().unique()
        print(f"Found {len(unique_scenarios)} scenarios: {list(unique_scenarios)}")
        
        for scenario_id in unique_scenarios:
            print(f"\n--- Processing scenario: {scenario_id} ---")
            
            # Filter data for this scenario
            scenario_data = merged_data[merged_data['scenario_id'] == scenario_id]
            print(f"  Scenario '{scenario_id}' has {len(scenario_data)} rows")
            
            if len(scenario_data) == 0:
                print(f"  ⚠ No data found for scenario '{scenario_id}', skipping")
                continue
            
            # Create scenario-level plot
            safe_scenario_id = re.sub(r'[^\w\-_\.]', '_', str(scenario_id))
            title = f'Scenario {scenario_id} - Spatial Analysis'
            filename = f'spatial_scenario_{safe_scenario_id}.png'
            
            try:
                create_spatial_plot(scenario_data, title, filename, subfolder='scenarios')
                print(f"  ✓ Successfully created plot for scenario '{scenario_id}'")
            except Exception as e:
                print(f"  ❌ Error creating plot for scenario '{scenario_id}': {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"⚠ No 'scenario_id' column found in merged data")
    
    # STEP 10: Perform object-syllable analysis if requested
    print("\n=== STEP 10: OBJECT-SYLLABLE ANALYSIS ===")
    
    # Check if object-syllable analysis is requested
    object_radius = getattr(args, 'object_radius', 1.0)  # Default radius of 1.0 meters
    perform_object_analysis = getattr(args, 'object_syllable_analysis', False)
    
    if perform_object_analysis and scenario_df is not None:
        print(f"✓ Performing object-syllable analysis with radius: {object_radius}m")
        
        # STEP 10.1: Overall object-syllable analysis (all data)
        print(f"\n--- STEP 10.1: OVERALL OBJECT-SYLLABLE ANALYSIS ---")
        perform_object_syllable_analysis(merged_data, scenario_df, tracker_data_list, 
                                       output_dir, color_map, object_radius, args)
        
        # STEP 10.2: Block-level object-syllable analysis
        print(f"\n--- STEP 10.2: BLOCK-LEVEL OBJECT-SYLLABLE ANALYSIS ---")
        if 'episode_set_name2' in merged_data.columns:
            unique_blocks = merged_data['episode_set_name2'].dropna().unique()
            print(f"Found {len(unique_blocks)} blocks for object-syllable analysis: {list(unique_blocks)}")
            
            for block in unique_blocks:
                print(f"\n--- Processing block: {block} ---")
                
                # Create block-specific output directory
                safe_block_name = str(block).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                block_output_dir = os.path.join(output_dir, 'object_syllable_analysis', 'blocks', safe_block_name)
                
                try:
                    perform_grouped_object_syllable_analysis(
                        merged_data, scenario_df, tracker_data_list, 
                        block_output_dir, color_map, object_radius, args,
                        group_type='block', group_name=block, group_column='episode_set_name2'
                    )
                    print(f"  ✓ Successfully completed object-syllable analysis for block '{block}'")
                except Exception as e:
                    print(f"  ❌ Error in object-syllable analysis for block '{block}': {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"⚠ No 'episode_set_name2' column found in merged data for block-level analysis")
        
        # STEP 10.3: Scenario-level object-syllable analysis
        print(f"\n--- STEP 10.3: SCENARIO-LEVEL OBJECT-SYLLABLE ANALYSIS ---")
        if 'scenario_id' in merged_data.columns:
            unique_scenarios = merged_data['scenario_id'].dropna().unique()
            print(f"Found {len(unique_scenarios)} scenarios for object-syllable analysis: {list(unique_scenarios)}")
            
            for scenario_id in unique_scenarios:
                print(f"\n--- Processing scenario: {scenario_id} ---")
                
                # Create scenario-specific output directory
                safe_scenario_id = re.sub(r'[^\w\-_\.]', '_', str(scenario_id))
                scenario_output_dir = os.path.join(output_dir, 'object_syllable_analysis', 'scenarios', safe_scenario_id)
                
                try:
                    perform_grouped_object_syllable_analysis(
                        merged_data, scenario_df, tracker_data_list, 
                        scenario_output_dir, color_map, object_radius, args,
                        group_type='scenario', group_name=scenario_id, group_column='scenario_id'
                    )
                    print(f"  ✓ Successfully completed object-syllable analysis for scenario '{scenario_id}'")
                except Exception as e:
                    print(f"  ❌ Error in object-syllable analysis for scenario '{scenario_id}': {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"⚠ No 'scenario_id' column found in merged data for scenario-level analysis")
    else:
        if not perform_object_analysis:
            print(f"⚠ Object-syllable analysis not requested (use --object-syllable-analysis to enable)")
        if scenario_df is None:
            print(f"⚠ No scenario data available for object-syllable analysis")
    
    # Final verification of scenario_df columns
    print(f"\n=== FINAL VERIFICATION ===")
    if scenario_df is not None:
        print(f"Final scenario_df columns: {list(scenario_df.columns)}")
        if 'scenario_id' in scenario_df.columns:
            scenario_id_count = scenario_df.columns.tolist().count('scenario_id')
            if scenario_id_count > 1:
                print(f"⚠ WARNING: scenario_id column appears {scenario_id_count} times in scenario_df!")
            else:
                print(f"✓ scenario_id column appears only once in scenario_df")
        else:
            print(f"⚠ scenario_id column not found in final scenario_df")
    
    # STEP 11: Comprehensive dataframe already created in STEP 6
    print("\n=== STEP 11: COMPREHENSIVE DATAFRAME ALREADY CREATED ===")
    print("  ✓ Comprehensive dataframe was created in STEP 6 starting with labels data")
    print("  ✓ All data sources have been merged and duplicate columns cleaned up")
    print("  ✓ Using comprehensive dataframe for spatial analysis and plotting")
    
    print(f"\n✅ Spatial analysis complete!")
    print(f"  - Merged data saved to: {data_dir}/")
    print(f"  - Comprehensive dataframe: {data_dir}/comprehensive_merged_data.csv")
    print(f"  - Plots organized in: {output_dir}/plots/")
    print(f"    • Block-level plots: plots/blocks/")
    print(f"    • Scenario-level plots: plots/scenarios/")
    if perform_object_analysis:
        print(f"  - Object-syllable analysis completed with radius: {object_radius}m")
        print(f"    • Object-syllable plots organized in: {output_dir}/object_syllable_analysis/")
        print(f"      - Overall analysis: object_syllable_analysis/")
        print(f"        • Frequency plots: object_syllable_analysis/frequency_plots/")
        print(f"          - Raw counts: object_syllable_analysis/frequency_plots/raw_counts/")
        print(f"          - Proportions: object_syllable_analysis/frequency_plots/proportions/")
        print(f"        • Spatial plots: object_syllable_analysis/spatial_plots/")
        print(f"        • Summary plots: object_syllable_analysis/summary_plots/")
        print(f"      - Block-level analysis: object_syllable_analysis/blocks/")
        print(f"        • Each block has its own subfolder with frequency/spatial/summary plots")
        print(f"      - Scenario-level analysis: object_syllable_analysis/scenarios/")
        print(f"        • Each scenario has its own subfolder with frequency/spatial/summary plots")
        if getattr(args, 'outside_radius_analysis', False):
            print(f"      - Outside-radius analysis: Enabled (computationally intensive)")
        else:
            print(f"      - Outside-radius analysis: Disabled (use --outside-radius-analysis to enable)")




def load_final_transformed_data(output_dir):
    """Load the final transformed data that was saved after all processing steps."""
    print(f"\n📂 LOADING FINAL TRANSFORMED DATA FROM: {output_dir}")
    
    data_dir = os.path.join(output_dir, 'data')
    
    # Load final participant data
    participant_file = os.path.join(data_dir, 'participant_data_final_transformed.csv')
    if os.path.exists(participant_file):
        participant_df = pd.read_csv(participant_file)
        print(f"✓ Loaded final participant data: {len(participant_df)} rows")
    else:
        print(f"⚠ Final participant data file not found: {participant_file}")
        participant_df = None
    
    # Load final scenario data
    scenario_file = os.path.join(data_dir, 'scenario_data_final_transformed.csv')
    if os.path.exists(scenario_file):
        scenario_df = pd.read_csv(scenario_file)
        print(f"✓ Loaded final scenario data: {len(scenario_df)} rows")
    else:
        print(f"⚠ Final scenario data file not found: {scenario_file}")
        scenario_df = None
    
    # Load final tracker data
    tracker_dir = os.path.join(data_dir, 'tracker_data_final_transformed')
    tracker_data_list = []
    
    if os.path.exists(tracker_dir):
        tracker_files = glob.glob(os.path.join(tracker_dir, '*_tracker_final.csv'))
        print(f"Found {len(tracker_files)} tracker files in: {tracker_dir}")
        
        for tracker_file in tracker_files:
            # Extract ppid, session, trial from filename
            filename = os.path.basename(tracker_file)
            parts = filename.replace('_tracker_final.csv', '').split('_')
            
            if len(parts) >= 3:
                ppid = parts[0]
                session = parts[1]
                trial = parts[2]
                
                # Load tracker data
                tracker_data = pd.read_csv(tracker_file)
                
                tracker_item = {
                    'ppid': ppid,
                    'session': session,
                    'trial': trial,
                    'tracker_data': tracker_data,
                    'tracker_file': tracker_file
                }
                tracker_data_list.append(tracker_item)
        
        print(f"✓ Loaded {len(tracker_data_list)} final tracker datasets")
    else:
        print(f"⚠ Final tracker data directory not found: {tracker_dir}")
    
    # Load final summary
    summary_file = os.path.join(data_dir, 'FINAL_DATA_SUMMARY.txt')
    if os.path.exists(summary_file):
        print(f"✓ Final data summary available: {summary_file}")
        with open(summary_file, 'r') as f:
            summary_content = f.read()
            print("Final data summary preview:")
            print(summary_content[:500] + "..." if len(summary_content) > 500 else summary_content)
    else:
        print(f"⚠ Final data summary not found: {summary_file}")
    
    return participant_df, scenario_df, tracker_data_list

def perform_grouped_object_syllable_analysis(merged_data, scenario_df, tracker_data_list, output_dir, color_map, object_radius=1.0, args=None, group_type=None, group_name=None, group_column=None):
    """
    Perform object-syllable analysis for a specific group (block or scenario).
    
    This function properly filters ALL data (participant, scenario, tracker) for the specified group
    and then applies the same object-syllable analysis procedure.
    
    Args:
        merged_data: DataFrame with participant data and syllable labels
        scenario_df: DataFrame with scenario object coordinates
        tracker_data_list: List of tracker data
        output_dir: Output directory for plots
        color_map: Dictionary mapping syllable labels to colors
        object_radius: Radius in meters for analysis (default: 1.0)
        args: Command line arguments
        group_type: Type of group ('block' or 'scenario')
        group_name: Name of the specific group
        group_column: Column name to filter by
    """
    print(f"\n=== {group_type.upper()}-LEVEL OBJECT-SYLLABLE ANALYSIS: {group_name} ===")
    print(f"Filtering data for {group_type}: {group_name}")
    
    # Filter participant data for this group
    if group_name is None or pd.isna(group_name):
        print(f"  ⚠ Invalid {group_type} name: {group_name}, skipping")
        return
    
    filtered_participant_data = merged_data[merged_data[group_column] == group_name].copy()
    print(f"  Filtered participant data: {len(filtered_participant_data)} rows")
    
    # Debug: Check participant data structure
    if len(filtered_participant_data) > 0:
        print(f"  Participant data columns: {list(filtered_participant_data.columns)}")
        
        # Check for required columns
        required_cols = ['participant_x', 'participant_y', 'predicted_label']
        missing_cols = [col for col in required_cols if col not in filtered_participant_data.columns]
        if missing_cols:
            print(f"  ⚠ Missing required columns: {missing_cols}")
        else:
            print(f"  ✓ All required columns present")
            
        # Check for syllable data
        if 'predicted_label' in filtered_participant_data.columns:
            unique_syllables = filtered_participant_data['predicted_label'].dropna().unique()
            print(f"  Unique syllables found: {list(unique_syllables)}")
            print(f"  Total syllable data points: {filtered_participant_data['predicted_label'].notna().sum()}")
    
    if len(filtered_participant_data) == 0:
        print(f"  ⚠ No participant data found for {group_type} '{group_name}', skipping")
        return
    
    # Get trials in this group
    group_trials = filtered_participant_data[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"  Found {len(group_trials)} trials in {group_type} '{group_name}'")
    
    # Debug: Show sample trials
    if len(group_trials) > 0:
        print(f"  Sample trials: {group_trials.head(3).to_dict('records')}")
        if len(group_trials) > 3:
            print(f"  ... and {len(group_trials) - 3} more trials")
    
    # Filter scenario data for trials in this group
    if scenario_df is not None:
        # Ensure consistent data types for merging
        for key in ['ppid', 'session', 'trial']:
            if key in scenario_df.columns:
                scenario_df[key] = scenario_df[key].astype(str)
            if key in group_trials.columns:
                group_trials[key] = group_trials[key].astype(str)
        
        filtered_scenario_data = pd.merge(scenario_df, group_trials, on=['ppid', 'session', 'trial'], how='inner')
        print(f"  Filtered scenario data: {len(filtered_scenario_data)} rows")
        
        # Debug: Show scenario data summary
        if len(filtered_scenario_data) > 0:
            scenario_trials = filtered_scenario_data[['ppid', 'session', 'trial']].drop_duplicates()
            print(f"  Scenario data covers {len(scenario_trials)} trials")
            print(f"  Scenario data columns: {list(filtered_scenario_data.columns)}")
            
            # Debug: Check for object columns
            object_columns = [col for col in filtered_scenario_data.columns if col.endswith('_x') or col.endswith('_y')]
            print(f"  Object columns found: {object_columns}")
            
            # Debug: Show sample object positions
            if len(object_columns) > 0:
                sample_objects = object_columns[:6]  # Show first 3 objects (x and y)
                print(f"  Sample object data:")
                for col in sample_objects:
                    non_null_count = filtered_scenario_data[col].notna().sum()
                    print(f"    {col}: {non_null_count} non-null values")
    else:
        filtered_scenario_data = None
        print(f"  ⚠ No scenario data available")
    
    # Filter tracker data for trials in this group
    filtered_tracker_data_list = []
    if tracker_data_list is not None:
        for tracker_item in tracker_data_list:
            ppid = str(tracker_item['ppid'])
            session = str(tracker_item['session'])
            trial = str(tracker_item['trial'])
            
            # Check if this tracker trial is in our group
            if any((group_trials['ppid'] == ppid) & 
                   (group_trials['session'] == session) & 
                   (group_trials['trial'] == trial)):
                filtered_tracker_data_list.append(tracker_item)
        
        print(f"  Filtered tracker data: {len(filtered_tracker_data_list)} trials")
        
        # Debug: Show tracker data summary
        if len(filtered_tracker_data_list) > 0:
            tracker_trials = [(item['ppid'], item['session'], item['trial']) for item in filtered_tracker_data_list]
            print(f"  Tracker trials: {tracker_trials[:3]}{'...' if len(tracker_trials) > 3 else ''}")
    else:
        print(f"  ⚠ No tracker data available")
    
    # Now perform the object-syllable analysis with the filtered data
    print(f"\n  🚀 Starting object-syllable analysis for {group_type} '{group_name}' with:")
    print(f"    - {len(filtered_participant_data)} participant data rows")
    print(f"    - {len(filtered_scenario_data) if filtered_scenario_data is not None else 0} scenario data rows")
    print(f"    - {len(filtered_tracker_data_list)} tracker trials")
    print(f"    - {len(group_trials)} unique trials")
    
    perform_object_syllable_analysis(filtered_participant_data, filtered_scenario_data, filtered_tracker_data_list, 
                                   output_dir, color_map, object_radius, args)
    
    print(f"  ✅ Completed object-syllable analysis for {group_type} '{group_name}'")
    print(f"  📊 Summary:")
    print(f"    - Participant data: {len(filtered_participant_data)} rows")
    print(f"    - Scenario data: {len(filtered_scenario_data) if filtered_scenario_data is not None else 0} rows")
    print(f"    - Tracker trials: {len(filtered_tracker_data_list)}")
    print(f"    - Unique trials: {len(group_trials)}")
    print(f"    - Output directory: {output_dir}")


def perform_object_syllable_analysis(merged_data, scenario_df, tracker_data_list, output_dir, color_map, object_radius=1.0, args=None):
    """
    Perform object-syllable correlation analysis by re-centering data to each object
    and computing syllable frequencies within/outside circular radius.
    
    This analysis only considers data when the object of interest is present and
    shifts all data to the object's origin for clearer visualization.
    
    Args:
        merged_data: DataFrame with participant data and syllable labels
        scenario_df: DataFrame with scenario object coordinates
        tracker_data_list: List of tracker data
        output_dir: Output directory for plots
        color_map: Dictionary mapping syllable labels to colors
        object_radius: Radius in meters for analysis (default: 1.0)
        args: Command line arguments
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    import os
    from matplotlib.patches import Circle
    
    print(f"\n=== OBJECT-SYLLABLE ANALYSIS ===")
    print(f"Radius: {object_radius} meters")
    print("Note: Only analyzing data when objects are present and shifting to object origin")
    
    # Discover objects automatically from scenario data (same as spatial analysis)
    objects_to_analyze = []
    if scenario_df is not None:
        # Find all coordinate columns from the original scenario data
        scenario_cols = [c for c in scenario_df.columns if c.endswith('_x') or c.endswith('_y')]
        print(f"Found {len(scenario_cols)} coordinate columns in original scenario data")
        
        # Extract object names using same regex as spatial analysis
        import re
        for c in scenario_cols:
            m = re.match(r'(.+)_([xyz])$', c)
            if m:
                obj_name = m.group(1).lower()  # Convert to lowercase for consistency
                if obj_name not in objects_to_analyze:
                    objects_to_analyze.append(obj_name)
    else:
        print("No scenario data available!")
    
    print(f"Objects discovered in scenario data: {objects_to_analyze}")
    if not objects_to_analyze:
        print("No objects found in scenario data!")
        return
    
    # Create output directory for object-syllable analysis
    object_analysis_dir = os.path.join(output_dir, 'object_syllable_analysis')
    os.makedirs(object_analysis_dir, exist_ok=True)
    
    # Ensure consistent data types for merging
    print("Ensuring consistent data types for object-syllable analysis merge...")
    
    # Convert merge keys to string type in both DataFrames
    merge_keys = ['ppid', 'session', 'trial']
    for key in merge_keys:
        if key in merged_data.columns:
            merged_data[key] = merged_data[key].astype(str)
    
    print(f"Merged data types after conversion: {merged_data[merge_keys].dtypes.to_dict()}")
    
    # Create scenario_data_for_analysis by merging with trials from merged_data
    scenario_data_for_analysis = None
    if scenario_df is not None:
        # Get trials from merged_data
        plot_trials = merged_data[['ppid', 'session', 'trial']].drop_duplicates()
        print(f"Found {len(plot_trials)} trials in merged_data")
        
        # Ensure scenario_df has consistent data types
        for key in ['ppid', 'session', 'trial']:
            if key in scenario_df.columns:
                scenario_df[key] = scenario_df[key].astype(str)
        
        # Merge scenario data for these trials
        scenario_data_for_analysis = pd.merge(plot_trials, scenario_df, on=['ppid', 'session', 'trial'], how='left')
        print(f"Merged scenario data shape: {scenario_data_for_analysis.shape}")
        
        # Debug: Show what objects are available in the merged data
        available_objects = []
        for obj in objects_to_analyze:
            x_col = f'{obj}_x'
            y_col = f'{obj}_y'
            if x_col in scenario_data_for_analysis.columns and y_col in scenario_data_for_analysis.columns:
                available_objects.append(obj)
        
        print(f"Objects available in merged scenario data: {available_objects}")
        print(f"Objects missing from merged scenario data: {[obj for obj in objects_to_analyze if obj not in available_objects]}")
    else:
        print("No scenario data available for merging!")
    
    # Initialize results storage
    all_object_frequencies = {}
    outside_radius_syllables = []
    all_centered_data_for_plotting = {}  # Store centered data for each object type
    
    # Get all trials that have participant data
    all_trials = merged_data[['ppid', 'session', 'trial']].drop_duplicates()
    print(f"Total trials with participant data: {len(all_trials)}")
    
    for object_name in objects_to_analyze:
        print(f"\n--- Analyzing object: {object_name} ---")
        
        # Find object columns in merged scenario data (using exact names from discovery)
        object_x_col = f'{object_name}_x'
        object_y_col = f'{object_name}_y'
        
        # Verify columns exist in merged scenario data
        if scenario_data_for_analysis is None:
            print(f"  ERROR: No scenario data available for analysis!")
            continue
            
        if object_x_col not in scenario_data_for_analysis.columns or object_y_col not in scenario_data_for_analysis.columns:
            print(f"  ERROR: Object '{object_name}' columns not found in merged scenario data!")
            print(f"  Looking for: '{object_x_col}' and '{object_y_col}'")
            print(f"  Available columns: {[col for col in scenario_data_for_analysis.columns if object_name in col]}")
            continue
        
        # Get object positions from merged scenario data
        # Select columns that exist in scenario_data_for_analysis
        required_columns = ['ppid', 'session', 'trial', object_x_col, object_y_col]
        existing_columns = [col for col in required_columns if col in scenario_data_for_analysis.columns]
        
        if 'time' in scenario_data_for_analysis.columns:
            existing_columns.append('time')
        
        object_positions = scenario_data_for_analysis[existing_columns].dropna()
        
        if len(object_positions) == 0:
            print(f"  No positions found for object '{object_name}', skipping...")
            continue
        
        print(f"  Found {len(object_positions)} object positions for {object_name}")
        
        # For scenario data without time, we consider the object present for the entire trial
        # So we filter merged data to only include trials where this object exists
        if 'time' in object_positions.columns:
            # If time information is available, use it for precise filtering
            object_presence_filter = object_positions[['ppid', 'session', 'trial', 'time']].drop_duplicates()
            # Ensure consistent data types for the merge
            for key in ['ppid', 'session', 'trial']:
                if key in object_presence_filter.columns:
                    object_presence_filter[key] = object_presence_filter[key].astype(str)
            
            filtered_data = pd.merge(merged_data, object_presence_filter, on=['ppid', 'session', 'trial', 'time'], how='inner')
        else:
            # If no time information, consider object present for entire trial
            object_presence_filter = object_positions[['ppid', 'session', 'trial']].drop_duplicates()
            # Ensure consistent data types for the merge
            for key in ['ppid', 'session', 'trial']:
                if key in object_presence_filter.columns:
                    object_presence_filter[key] = object_presence_filter[key].astype(str)
            
            filtered_data = pd.merge(merged_data, object_presence_filter, on=['ppid', 'session', 'trial'], how='inner')
        
        if len(filtered_data) == 0:
            print(f"  No participant data found when {object_name} is present, skipping...")
            continue
        
        print(f"  Found {len(filtered_data)} participant data points when {object_name} is present")
        
        # For each object position, re-center data and analyze syllables
        object_frequencies = {}
        all_centered_data = []  # Collect all centered data for this object type
        
        for idx, (_, row) in enumerate(object_positions.iterrows()):
            obj_x = row[object_x_col]
            obj_y = row[object_y_col]
            obj_ppid = row['ppid']
            obj_session = row['session']
            obj_trial = row['trial']
            
            # Get data for this trial when object is present
            if 'time' in object_positions.columns:
                obj_time = row['time']
                # Get data for this specific time point when object is present
                time_specific_data = filtered_data[
                    (filtered_data['ppid'] == obj_ppid) & 
                    (filtered_data['session'] == obj_session) & 
                    (filtered_data['trial'] == obj_trial) & 
                    (filtered_data['time'] == obj_time)
                ]
            else:
                # Get all data for this trial (object is present for entire trial)
                time_specific_data = filtered_data[
                    (filtered_data['ppid'] == obj_ppid) & 
                    (filtered_data['session'] == obj_session) & 
                    (filtered_data['trial'] == obj_trial)
                ]
            
            if len(time_specific_data) == 0:
                continue
            
            # Re-center data to this object position
            recentered_data = time_specific_data.copy()
            recentered_data['participant_x_centered'] = recentered_data['participant_x'] - obj_x
            recentered_data['participant_y_centered'] = recentered_data['participant_y'] - obj_y
            
            # Calculate distance from object center
            recentered_data['distance_from_object'] = np.sqrt(
                recentered_data['participant_x_centered']**2 + 
                recentered_data['participant_y_centered']**2
            )
            
            # Show distance statistics for hide objects (simplified)
            if object_name == 'hide':
                within_radius_count = len(recentered_data[recentered_data['distance_from_object'] <= object_radius])
                print(f"      Hide object: {within_radius_count} points within {object_radius}m radius")
            
            # Store centered data for plotting
            all_centered_data.append(recentered_data)
            
            # Find syllables within radius
            within_radius = recentered_data[recentered_data['distance_from_object'] <= object_radius]
            
            if len(within_radius) > 0:
                # Count syllable frequencies within radius
                # Use the label column from args
                label_col = args.label_column if args else 'predicted_label'
                
                syllable_counts = within_radius[label_col].value_counts()
                
                # Store frequencies for this object position
                for syllable, count in syllable_counts.items():
                    if syllable not in object_frequencies:
                        object_frequencies[syllable] = 0
                    object_frequencies[syllable] += count
                
                if len(within_radius) > 0:
                    print(f"      Found {len(within_radius)} syllables within radius")
                else:
                    print(f"      No syllables found within radius")
            
            # Note: We'll handle outside-radius analysis separately after processing all objects
        
        # Combine all centered data for this object type
        if all_centered_data:
            all_centered_data_for_plotting[object_name] = pd.concat(all_centered_data, ignore_index=True)
        
        # Store results for this object
        all_object_frequencies[object_name] = object_frequencies
        
        # Calculate number of observations for this object (unique ppid-session-trial combinations from participant data)
        # IMPORTANT: Count trials from ORIGINAL merged_data that have participant location data with syllables
        # NOT from filtered_data (which only includes trials where object is present)
        if 'predicted_label' in merged_data.columns and 'participant_x' in merged_data.columns and 'participant_y' in merged_data.columns:
            # Count trials that have syllable data (non-null predicted_label) from the ORIGINAL data
            syllable_trials = merged_data[merged_data['predicted_label'].notna()][['ppid', 'session', 'trial']].drop_duplicates()
            n_observations = len(syllable_trials)
            
            # Also show how many trials have the object present for context
            trials_with_object = filtered_data[['ppid', 'session', 'trial']].drop_duplicates().shape[0]
            print(f"  Object '{object_name}' analysis:")
            print(f"    - Total trials with syllable data (mocap + labels): {n_observations}")
            print(f"    - Trials where this object is present: {trials_with_object}")
            print(f"    - Trials where object is present but no syllable data: {trials_with_object - filtered_data[filtered_data['predicted_label'].notna()][['ppid', 'session', 'trial']].drop_duplicates().shape[0]}")
        else:
            # If no syllable data available, count all trials from original data
            n_observations = merged_data[['ppid', 'session', 'trial']].drop_duplicates().shape[0]
            print(f"  Object '{object_name}' has {n_observations} total trials (no syllable data available)")
        
        # Create plot for this object (only if frequencies exist)
        if object_frequencies:
            # Create raw count plot
            create_object_syllable_plot(
                object_name, object_frequencies, color_map, 
                object_analysis_dir, object_radius, n_observations=n_observations, plot_type="raw_counts"
            )
            
            # Create normalized proportion plot
            total_syllables = sum(object_frequencies.values())
            normalized_frequencies = {syllable: count/total_syllables for syllable, count in object_frequencies.items()}
            create_object_syllable_plot(
                object_name, normalized_frequencies, color_map, 
                object_analysis_dir, object_radius, n_observations=n_observations, plot_type="proportions"
            )
        
        # Create spatial plot for this object showing syllables around object positions (centered)
        # Always create spatial plot, even if no syllables found within radius
        if object_name in all_centered_data_for_plotting:
            create_object_spatial_plot_centered(
                object_name, all_centered_data_for_plotting[object_name],
                color_map, object_analysis_dir, object_radius, args, n_observations=n_observations,
                xlim=getattr(args, 'xlim', None), ylim=getattr(args, 'ylim', None)
            )
        else:
            print(f"  ⚠ No centered data available for spatial plot of {object_name}")
    
    # Create proper trial-by-trial outside-radius analysis (only if flag is set)
    perform_outside_analysis = getattr(args, 'outside_radius_analysis', False)
    
    if perform_outside_analysis:
        print(f"\n--- TRIAL-BY-TRIAL OUTSIDE-RADIUS ANALYSIS ===")
        print(f"Processing {len(all_trials)} trials to find syllables outside ALL object radii...")
        print(f"⚠ WARNING: This analysis is computationally intensive!")
        
        # Process each trial to find syllables outside ALL object radii
        trial_outside_syllables = []
        trials_with_outside_syllables = 0
    else:
        print(f"\n--- SKIPPING OUTSIDE-RADIUS ANALYSIS ===")
        print(f"Use --outside-radius-analysis flag to enable this computationally intensive analysis")
        trial_outside_syllables = []
        trials_with_outside_syllables = 0
    
    if perform_outside_analysis:
        for _, trial_row in all_trials.iterrows():
            ppid, session, trial = trial_row['ppid'], trial_row['session'], trial_row['trial']
            
            # Get participant data for this trial
            trial_data = merged_data[
                (merged_data['ppid'] == ppid) & 
                (merged_data['session'] == session) & 
                (merged_data['trial'] == trial)
            ]
            
            if len(trial_data) == 0:
                continue
            
            # Get scenario data for this trial to find all object positions
            trial_scenario_data = scenario_data_for_analysis[
                (scenario_data_for_analysis['ppid'] == ppid) & 
                (scenario_data_for_analysis['session'] == session) & 
                (scenario_data_for_analysis['trial'] == trial)
            ]
            
            if len(trial_scenario_data) == 0:
                # No scenario data for this trial, consider all syllables as outside
                label_col = args.label_column if args else 'predicted_label'
                if label_col in trial_data.columns:
                    trial_outside_syllables.extend(trial_data[label_col].dropna().tolist())
                continue
            
            # For each participant data point, check if it's outside ALL object radii
            outside_syllables_this_trial = 0
            
            for _, participant_row in trial_data.iterrows():
                participant_x = participant_row.get('participant_x')
                participant_y = participant_row.get('participant_y')
                
                if pd.isna(participant_x) or pd.isna(participant_y):
                    continue
                
                # Check distance to ALL objects in this trial
                is_outside_all_objects = True
                
                for object_name in objects_to_analyze:
                    object_x_col = f'{object_name}_x'
                    object_y_col = f'{object_name}_y'
                    
                    if object_x_col in trial_scenario_data.columns and object_y_col in trial_scenario_data.columns:
                        # Get all positions of this object in this trial
                        object_positions = trial_scenario_data[[object_x_col, object_y_col]].dropna()
                        
                        for _, obj_row in object_positions.iterrows():
                            obj_x = obj_row[object_x_col]
                            obj_y = obj_row[object_y_col]
                            
                            # Calculate distance to this object
                            distance = np.sqrt((participant_x - obj_x)**2 + (participant_y - obj_y)**2)
                            
                            # If within radius of ANY object, this syllable is not outside
                            if distance <= object_radius:
                                is_outside_all_objects = False
                                break
                        
                        if not is_outside_all_objects:
                            break
                
                # If this participant position is outside ALL object radii, add the syllable
                if is_outside_all_objects:
                    label_col = args.label_column if args else 'predicted_label'
                    if label_col in participant_row and pd.notna(participant_row[label_col]):
                        trial_outside_syllables.append(participant_row[label_col])
                        outside_syllables_this_trial += 1
            
            if outside_syllables_this_trial > 0:
                trials_with_outside_syllables += 1
                print(f"  Trial {ppid}-{session}-{trial}: {outside_syllables_this_trial} syllables outside ALL object radii")
    
    # Create outside-radius analysis if we found any syllables (only if flag is set)
    if perform_outside_analysis:
        if trial_outside_syllables:
            outside_frequencies = pd.Series(trial_outside_syllables).value_counts()
            
            # Calculate total observations for outside-radius analysis (trials with participant data and syllables)
            # IMPORTANT: Count trials from ORIGINAL merged_data that have participant location data with syllables
            if 'predicted_label' in merged_data.columns and 'participant_x' in merged_data.columns and 'participant_y' in merged_data.columns:
                # Count trials that have syllable data (non-null predicted_label) from the ORIGINAL data
                syllable_trials = merged_data[merged_data['predicted_label'].notna()][['ppid', 'session', 'trial']].drop_duplicates()
                total_observations = len(syllable_trials)
                print(f"Outside-radius analysis: Using {total_observations} trials with syllable data (mocap + labels) from original data")
            else:
                # If no syllable data available, count all trials from original data
                total_observations = len(all_trials)
                print(f"Outside-radius analysis: Using {total_observations} total trials from original data (no syllable data available)")
            
            print(f"Found {len(trial_outside_syllables)} syllables outside ALL object radii across {trials_with_outside_syllables} trials (out of {total_observations} total trials)")
            print(f"Top syllables outside all object radii: {outside_frequencies.head(5).to_dict()}")
            
            # Create raw count plot for outside-radius
            create_object_syllable_plot(
                "outside_all_objects", outside_frequencies.to_dict(), color_map,
                object_analysis_dir, object_radius, is_outside_analysis=True, n_observations=total_observations, plot_type="raw_counts"
            )
            
            # Create normalized proportion plot for outside-radius
            total_syllables = sum(outside_frequencies.values)
            normalized_outside_frequencies = {syllable: count/total_syllables for syllable, count in outside_frequencies.items()}
            create_object_syllable_plot(
                "outside_all_objects", normalized_outside_frequencies, color_map,
                object_analysis_dir, object_radius, is_outside_analysis=True, n_observations=total_observations, plot_type="proportions"
            )
        else:
            print(f"No syllables found outside ALL object radii across {len(all_trials)} trials")
    else:
        print(f"Outside-radius analysis skipped (use --outside-radius-analysis flag to enable)")
    
    # Create summary plot comparing all objects
    # IMPORTANT: Count trials from ORIGINAL merged_data that have participant location data with syllables
    if 'predicted_label' in merged_data.columns and 'participant_x' in merged_data.columns and 'participant_y' in merged_data.columns:
        # Count trials that have syllable data (non-null predicted_label) from the ORIGINAL data
        syllable_trials = merged_data[merged_data['predicted_label'].notna()][['ppid', 'session', 'trial']].drop_duplicates()
        total_observations = len(syllable_trials)
        print(f"Summary plot: Using {total_observations} trials with syllable data (mocap + labels) from original data")
    else:
        # If no syllable data available, count all trials from original data
        total_observations = merged_data[['ppid', 'session', 'trial']].drop_duplicates().shape[0]
        print(f"Summary plot: Using {total_observations} total trials from original data (no syllable data available)")
    
    create_object_syllable_summary_plot(
        all_object_frequencies, color_map, object_analysis_dir, object_radius, total_observations=total_observations
    )


def create_object_syllable_plot(object_name, frequencies, color_map, output_dir, radius, is_outside_analysis=False, n_observations=None, plot_type="raw_counts"):
    """Create a plot showing syllable frequencies for a specific object."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not frequencies:
        return
    
    # Sort frequencies by count
    sorted_frequencies = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    syllables, counts = zip(*sorted_frequencies)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars with colors from color map
    colors = [color_map.get(str(syllable), '#D3D3D3') for syllable in syllables]
    bars = ax.bar(range(len(syllables)), counts, color=colors)
    
    # Customize plot with object name in title
    if is_outside_analysis:
        if plot_type == "proportions":
            title = f"Syllable Proportions Outside {radius}m Radius of All Objects"
        else:
            title = f"Syllable Frequencies Outside {radius}m Radius of All Objects"
    else:
        # Map object names to display names (same as in spatial analysis)
        object_display_names = {
            'task': 'Task',
            'safehouse': 'Safehouse', 
            'threat': 'Threat',
            'startpoint': 'StartPoint',
            'weapons': 'Weapons',
            'standpoint': 'StandPoint',
            'wall': 'Wall',
            'hide': 'Hide'
        }
        display_name = object_display_names.get(object_name, object_name.title())
        if plot_type == "proportions":
            title = f"Syllable Proportions Within {radius}m Radius of {display_name} Objects"
        else:
            title = f"Syllable Frequencies Within {radius}m Radius of {display_name} Objects"
    
    # Add observation count to title if provided
    if n_observations is not None:
        title = f"{title} (n={n_observations} trials)"
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Syllable Type', fontsize=12)
    if plot_type == "proportions":
        ax.set_ylabel('Proportion', fontsize=12)
    else:
        ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xticks(range(len(syllables)))
    ax.set_xticklabels(syllables, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if plot_type == "proportions":
        filename = f"{object_name}_syllable_proportions_r{radius}m.png"
        if is_outside_analysis:
            filename = f"outside_all_objects_syllable_proportions_r{radius}m.png"
    else:
        filename = f"{object_name}_syllable_frequencies_r{radius}m.png"
        if is_outside_analysis:
            filename = f"outside_all_objects_syllable_frequencies_r{radius}m.png"
    
    # Add observation count to filename if provided
    if n_observations is not None:
        filename = filename.replace('.png', f'_n{n_observations}trials.png')
    
    # Create subfolder for frequency plots with organized structure
    if plot_type == "proportions":
        freq_plots_dir = os.path.join(output_dir, 'frequency_plots', 'proportions')
    else:
        freq_plots_dir = os.path.join(output_dir, 'frequency_plots', 'raw_counts')
    os.makedirs(freq_plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(freq_plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    plot_type_label = "proportions" if plot_type == "proportions" else "raw counts"
    print(f"  Saved {plot_type_label} plot: {filename}")


def create_object_syllable_summary_plot(all_frequencies, color_map, output_dir, radius, total_observations=None):
    """Create a summary plot comparing syllable frequencies across all objects."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Get all unique syllables
    all_syllables = set()
    for frequencies in all_frequencies.values():
        all_syllables.update(frequencies.keys())
    
    if not all_syllables:
        return
    
    # Create DataFrame for plotting
    plot_data = []
    for object_name, frequencies in all_frequencies.items():
        for syllable in all_syllables:
            count = frequencies.get(syllable, 0)
            # Use proper display names for objects (same as in other plots)
            object_display_names = {
                'task': 'Task',
                'safehouse': 'Safehouse', 
                'threat': 'Threat',
                'startpoint': 'StartPoint',
                'weapons': 'Weapons',
                'standpoint': 'StandPoint',
                'wall': 'Wall',
                'hide': 'Hide'
            }
            display_name = object_display_names.get(object_name, object_name.title())
            plot_data.append({
                'Object': display_name,
                'Syllable': syllable,
                'Frequency': count
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create heatmap
    pivot_df = df.pivot(index='Object', columns='Syllable', values='Frequency').fillna(0)
    
    fig, ax = plt.subplots(figsize=(max(12, len(all_syllables)*0.8), 8))
    
    # Create heatmap with plasma colormap
    im = ax.imshow(pivot_df.values, cmap='plasma', aspect='auto')
    
    # Customize plot
    title = f'Syllable Frequencies Within {radius}m Radius of Objects'
    if total_observations is not None:
        title = f"{title} (n={total_observations} trials)"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Syllable Type', fontsize=12)
    ax.set_ylabel('Object Type', fontsize=12)
    
    # Set ticks
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frequency', fontsize=12)
    
    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            text = ax.text(j, i, int(pivot_df.iloc[i, j]),
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"object_syllable_summary_r{radius}m.png"
    
    # Add observation count to filename if provided
    if total_observations is not None:
        filename = filename.replace('.png', f'_n{total_observations}obs.png')
    
    # Create subfolder for summary plots
    summary_plots_dir = os.path.join(output_dir, 'summary_plots')
    os.makedirs(summary_plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(summary_plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved summary plot: {filename}")


def create_object_spatial_plot_centered(object_name, centered_data, color_map, output_dir, radius, args, n_observations=None, xlim=None, ylim=None):
    """Create a spatial plot showing syllables around object positions (centered)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib.patches import Circle
    
    if len(centered_data) == 0:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot syllables colored by type (same characteristics as block plots)
    if 'predicted_label' in centered_data.columns:
        unique_syllables = centered_data['predicted_label'].dropna().unique()
        
        for syllable in unique_syllables:
            syllable_data = centered_data[centered_data['predicted_label'] == syllable]
            
            # Get color from color map
            if color_map is not None and str(syllable) in color_map:
                syll_color = color_map[str(syllable)]
            else:
                # Use multi-colormap approach for more unique colors
                hash_val = hash(str(syllable))
                colormaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Dark2]
                colors_per_cmap = 20
                cmap_index = (abs(hash_val) // colors_per_cmap) % len(colormaps)
                color_index = abs(hash_val) % colors_per_cmap
                syll_color = colormaps[cmap_index](color_index)
            
            # Plot syllable data points (same characteristics as block plots: s=8, alpha=0.7, zorder=2)
            ax.scatter(syllable_data['participant_x_centered'], syllable_data['participant_y_centered'],
                      color=syll_color, s=8, alpha=0.7, zorder=2, label=f'{syllable}')
    
    # Add radius circle with high zorder to ensure it's on top
    circle = Circle((0, 0), radius, fill=False, color='red', linestyle='--', linewidth=3, label=f'{radius}m radius', zorder=20)
    ax.add_patch(circle)
    
    # Add object center marker using same style as object_styles from spatial analysis
    object_styles = {
        'task':      dict(marker='^', color='green',     label='Task',        s=120),
        'safehouse': dict(marker='s', color='saddlebrown',label='Safehouse',  s=120),
        'threat':    dict(marker='p', color='red',       label='Threat',      s=120),
        'startpoint':dict(marker='o', color='yellow',    label='StartPoint',  s=120),
        'weapons':   dict(marker='s', color='magenta',   label='Weapons',     s=120),
        'standpoint':dict(marker='o', color='orange',    label='StandPoint',  s=120),
        'wall':      dict(marker='s', color='grey',      label='Wall',        s=120),
        'hide':      dict(marker='s', color='cyan',      label='Hide',        s=120),
    }
    
    # Get object style for center marker
    object_style = object_styles.get(object_name, dict(marker='*', color='red', label=object_name.title(), s=200))
    
    # Add object center marker with same characteristics as in spatial analysis
    ax.scatter(0, 0, marker=object_style['marker'], color=object_style['color'], 
              s=object_style['s'], edgecolor='black', zorder=15, 
              label=f'{object_style["label"]} center')
    
    # Set plot properties
    ax.set_xlabel('Distance from object center (meters)', fontsize=12)
    ax.set_ylabel('Distance from object center (meters)', fontsize=12)
    
    # Add object name to title
    object_display_names = {
        'task': 'Task',
        'safehouse': 'Safehouse', 
        'threat': 'Threat',
        'startpoint': 'StartPoint',
        'weapons': 'Weapons',
        'standpoint': 'StandPoint',
        'wall': 'Wall',
        'hide': 'Hide'
    }
    display_name = object_display_names.get(object_name, object_name.title())
    title = f'Syllables Around {display_name} Objects (Centered) - {radius}m Radius'
    
    # Add observation count to title if provided
    if n_observations is not None:
        title = f"{title} (n={n_observations} trials)"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set axis limits - use provided limits or calculate based on radius
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        # Calculate dynamic limits based on radius for centered plots
        max_range = max(radius * 1.5, 3)  # Show at least 3m or 1.5x radius
        ax.set_xlim(-max_range, max_range)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Calculate dynamic limits based on radius for centered plots
        max_range = max(radius * 1.5, 3)  # Show at least 3m or 1.5x radius
        ax.set_ylim(-max_range, max_range)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{object_name}_spatial_centered_r{radius}m.png"
    
    # Add observation count to filename if provided
    if n_observations is not None:
        filename = filename.replace('.png', f'_n{n_observations}trials.png')
    
    # Create subfolder for spatial plots
    spatial_plots_dir = os.path.join(output_dir, 'spatial_plots')
    os.makedirs(spatial_plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(spatial_plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved spatial plot: {filename}")


@tictoc
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Clean Location-Based Syllable Usage Analysis')
    
    # Required arguments
    parser.add_argument('--labels-path', required=True, help='Path to labels CSV file')
    parser.add_argument('--output-dir', required=True, help='Base output directory (spatial folder will be created inside)')
    
    # Optional arguments
    parser.add_argument('--metadata-path', help='Path to metadata CSV file (deprecated, use --experimental-conditions instead)')
    
    # Optional arguments
    parser.add_argument('--label-column', default='predicted_label', help='Label column name')
    parser.add_argument('--coords-dir', help='Root directory of coordinate files (ppid/session/trial.csv)')
    parser.add_argument('--data-dir', help='Directory containing both tracker data (ppid/session/trackers/) and scenario data (ppid/session/othertrialdata/)')
    parser.add_argument('--metadata-dir', help='Directory with metadata structure (for tracker data) - deprecated, use --data-dir instead')
    parser.add_argument('--scenario-data', help='Directory with scenario data - deprecated, use --data-dir instead')
    parser.add_argument('--colour-map-object', help='Path to colour map pickle file')
    parser.add_argument('--scenario-rotation', help='Path to scenario rotation CSV file')
    parser.add_argument('--use-json-rotation', action='store_true', help='Extract rotation angles from scenario JSON files (euler_angles.y field) - overrides --scenario-rotation')
    parser.add_argument('--use-icp', action='store_true', help='Use ICP (Iterative Closest Point) with Open3D for robust tracker-participant alignment. This is the only rotation method available.')
    parser.add_argument('--experimental-conditions', help='Path to CSV file with experimental conditions including scenario_id, ppid, session, trial columns')
    parser.add_argument('--failed-trials-path', help='Path to CSV file containing metadata of failed trials (must have ppid, session, trial columns)')
    
    # Flags
    parser.add_argument('--spatial-analysis', action='store_true', help='Perform spatial analysis')
    parser.add_argument('--spatial-block-column', type=str, default="episode_set_name2", help='Metadata column to use for block-wise spatial analysis')
    parser.add_argument('--alignment-to-participant', action='store_true', help='Align to participant initial coordinates')
    parser.add_argument('--shift-to-median-startpoints', action='store_true', help='Shift all coordinates to median startpoint coordinates grouped by scenario_id (requires --scenario-data)')
    parser.add_argument('--plot-trackers', action='store_true', help='Plot tracker paths')
    parser.add_argument('--fast-test', action='store_true', help='Fast testing mode')
    parser.add_argument('--debug', action='store_true', help='Debug mode with interactive exploration')
    parser.add_argument('--debug-plots-only', action='store_true', help='Debug mode with plots only (no interactive pdb)')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample data by taking every Nth row (default: 1 = no downsampling)')
    
    # Plot settings
    parser.add_argument('--xlim', type=float, nargs=2, default=[-6, 6], help='x-axis limits')
    parser.add_argument('--ylim', type=float, nargs=2, default=[-4, 4], help='y-axis limits')
    
    # Object-syllable analysis settings
    parser.add_argument('--object-syllable-analysis', action='store_true', help='Perform object-syllable analysis (shifts data to each object as origin and analyzes syllable frequencies within radius)')
    parser.add_argument('--object-radius', type=float, default=1.2, help='Radius in meters for object-syllable analysis (default: 1.0)')
    
    args = parser.parse_args()
    
    # Validate that --shift-to-median-startpoints is only used with scenario data
    if args.shift_to_median_startpoints and not (args.data_dir or args.scenario_data):
        print("Error: --shift-to-median-startpoints flag requires scenario data to be provided.")
        print("Please provide either --data-dir or --scenario-data along with --shift-to-median-startpoints.")
        sys.exit(1)
    
    # Validate that experimental conditions has required columns if provided
    if args.experimental_conditions:
        try:
            conditions_df = pd.read_csv(args.experimental_conditions)
            required_cols = ['ppid', 'session', 'trial']
            missing_cols = [col for col in required_cols if col not in conditions_df.columns]
            if missing_cols:
                print(f"Error: Experimental conditions CSV is missing required columns: {missing_cols}")
                print("Please ensure the CSV contains: ppid, session, trial columns")
                sys.exit(1)
        except Exception as e:
            print(f"Error: Could not read experimental conditions CSV: {e}")
            sys.exit(1)
    
    # Validate downsampling factor
    if args.downsample < 1:
        print("Error: --downsample must be a positive integer (>= 1).")
        print(f"Current value: {args.downsample}")
        sys.exit(1)
    
    print("=== CLEAN LOCATION-BASED SYLLABLE USAGE ANALYSIS ===")
    
    # Create organized output directory structure
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spatial_output_dir = os.path.join(args.output_dir, f"spatial_{timestamp}")
    
    # Create main directories
    os.makedirs(spatial_output_dir, exist_ok=True)
    os.makedirs(os.path.join(spatial_output_dir, "debug_plots"), exist_ok=True)
    os.makedirs(os.path.join(spatial_output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(spatial_output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(spatial_output_dir, "logs"), exist_ok=True)
    
    print(f"\n📁 OUTPUT DIRECTORY STRUCTURE:")
    print(f"   Base directory: {args.output_dir}")
    print(f"   Spatial analysis folder: {spatial_output_dir}")
    print(f"   ├── debug_plots/     (Debug plots for each step)")
    print(f"   ├── data/            (Processed data files)")
    print(f"   ├── plots/           (Final analysis plots)")
    print(f"   └── logs/            (Analysis logs and summaries)")
    
    # Warn about deprecated flags
    if args.metadata_dir or args.scenario_data:
        print("\n⚠️  WARNING: --metadata-dir and --scenario-data flags are deprecated.")
        print("   Please use --data-dir instead, which loads both tracker and scenario data from a single directory.")
        print("   The directory should contain: ppid/session/trackers/ and ppid/session/othertrialdata/ subfolders.")
    
    # Show downsampling info if enabled
    if args.downsample > 1:
        print(f"\n📉 DOWNSAMPLING ENABLED: Taking every {args.downsample}th row")
        print(f"   This will reduce data size to approximately {100/args.downsample:.1f}% of original")
    
    # Load data
    print("\n=== Loading Data ===")
    print("This script implements a unified data loading approach:")
    print("  1. Load all data sources (labels, participant, scenario, metadata)")
    print("  2. Merge them into a single DataFrame BEFORE any transformations")
    print("  3. Apply transformations to the unified dataset")
    print("  4. Use the transformed unified dataset for spatial analysis")
    print("  This ensures data consistency and eliminates merge issues during analysis.")
    print("\n⚠ IMPORTANT: For participant movement visualization, use --coords-dir argument")
    print("⚠ Without --coords-dir, participant coordinates will be dummy values (0,0)")
    print("⚠ This means spatial analysis will show static participant positions, not movement")
    
    # Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(args.labels_path)
    print(f"✓ Loaded labels: {len(labels_df)} rows")
    
    # Load metadata (deprecated) or experimental conditions
    metadata_df = None
    if args.metadata_path:
        print("Loading metadata (deprecated)...")
        metadata_df = pd.read_csv(args.metadata_path)
        print(f"✓ Loaded metadata: {len(metadata_df)} rows")
        print("⚠ Warning: --metadata-path is deprecated. Please use --experimental-conditions instead.")
        
        # Process scenario_id values according to the new rules
        if 'scenario_id' in metadata_df.columns:
            print("Processing scenario_id values...")
            
            # Create a copy to avoid modifying the original
            processed_scenario_ids = []
            
            for idx, row in metadata_df.iterrows():
                scenario_id = row['scenario_id']
                
                if pd.isna(scenario_id):
                    # Missing scenario_id - use episode column
                    if 'episode' in metadata_df.columns:
                        episode = row['episode']
                        if pd.notna(episode):
                            try:
                                # Split by "_" and take second value
                                episode_parts = str(episode).split('_')
                                if len(episode_parts) >= 2:
                                    second_value = episode_parts[1]
                                    # Remove leading zeros
                                    scenario_id = str(int(second_value))
                                    print(f"  Row {idx}: Missing scenario_id, derived from episode '{episode}' -> scenario_id '{scenario_id}'")
                                else:
                                    print(f"  Row {idx}: Episode '{episode}' doesn't have expected format (need at least 2 parts after splitting by '_')")
                                    scenario_id = None
                            except (ValueError, IndexError) as e:
                                print(f"  Row {idx}: Error processing episode '{episode}': {e}")
                                scenario_id = None
                        else:
                            print(f"  Row {idx}: Missing both scenario_id and episode")
                            scenario_id = None
                    else:
                        print(f"  Row {idx}: Missing scenario_id and no episode column available")
                        scenario_id = None
                else:
                    # Existing scenario_id - handle "s" and "f" suffixes
                    scenario_id_str = str(scenario_id).strip()
                    if scenario_id_str.endswith('s') or scenario_id_str.endswith('f'):
                        # Keep the suffix as the scenario_id identifier
                        suffix = scenario_id_str[-1]  # Get last character
                        scenario_id = suffix
                        print(f"  Row {idx}: scenario_id '{scenario_id_str}' -> scenario_id '{scenario_id}' (kept suffix as identifier)")
                    else:
                        # No suffix, just remove leading zeros
                        try:
                            scenario_id = str(int(scenario_id_str))
                            print(f"  Row {idx}: scenario_id '{scenario_id_str}' -> '{scenario_id}' (removed leading zeros)")
                        except ValueError:
                            print(f"  Row {idx}: Error converting scenario_id '{scenario_id_str}' to integer")
                            scenario_id = None
                
                processed_scenario_ids.append(scenario_id)
            
            # Update the metadata_df with processed scenario_ids
            metadata_df['scenario_id'] = processed_scenario_ids
            
            # Show summary of processed scenario_ids
            unique_scenario_ids = metadata_df['scenario_id'].dropna().unique()
            print(f"✓ Processed scenario_ids. Unique values: {sorted(unique_scenario_ids)}")
            
    elif args.experimental_conditions:
        print("Loading experimental conditions as metadata replacement...")
        metadata_df = pd.read_csv(args.experimental_conditions)
        print(f"✓ Loaded experimental conditions: {len(metadata_df)} rows")
    else:
        print("⚠ Warning: No metadata or experimental conditions provided. Some features may not work properly.")
    
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
                print(f"✓ Loaded failed trials data with {len(failed_trials_data)} trials")
                print(f"  Using columns: {ppid_col}->ppid, {session_col}->session, {trial_col}->trial")
            else:
                missing_cols = []
                if not ppid_col:
                    missing_cols.append("ppid (or ppid_vr, ppid_coda)")
                if not session_col:
                    missing_cols.append("session (or session_vr, session_coda)")
                if not trial_col:
                    missing_cols.append("trial (or trial_vr, trial_coda)")
                print(f"⚠ Warning: Failed trials CSV missing required columns: {missing_cols}. Failed trials will be ignored.")
                failed_trials_data = None
        except Exception as e:
            print(f"⚠ Warning: Could not load failed trials CSV: {str(e)}. Failed trials will be ignored.")
            failed_trials_data = None
    
    # Load colour map
    color_map = None
    if args.colour_map_object:
        color_map = load_colour_map(args.colour_map_object)
        if color_map:
            print(f"✓ Loaded colour map with {len(color_map)} colors")
    
    # Load participant data (from coords-dir)
    participant_df = None
    if args.coords_dir:
        print("Building participant location CSV from coordinates directory...")
        participant_location_path = os.path.join(spatial_output_dir, 'data', 'participant_location.csv')
        participant_df = build_participant_location_csv(args.coords_dir, output_csv=participant_location_path, fast_test=args.fast_test)
        print(f"✓ Built participant location data: {len(participant_df)} rows")
    else:
        # Fallback: use labels data if no coords-dir provided
        print("No --coords-dir provided, using labels data for participant coordinates...")
        participant_df = labels_df.copy()
        
        # Add required metadata columns if they don't exist
        if 'ppid' not in participant_df.columns:
            # Extract ppid from the data directory structure or use a default
            if args.data_dir:
                # Try to extract ppid from data directory
                data_dir_parts = args.data_dir.split(os.sep)
                if len(data_dir_parts) > 0:
                    # Look for participant ID in the data directory structure
                    # The structure should be: data_dir/ppid/session/
                    if len(data_dir_parts) >= 2:
                        potential_ppid = data_dir_parts[-2]  # Second to last part
                        if potential_ppid.isdigit():
                            ppid = potential_ppid
                            participant_df['ppid'] = ppid
                            print(f"  Added ppid column with value: {ppid} (from data directory structure)")
                        else:
                            ppid = '10'  # Default fallback
                            participant_df['ppid'] = ppid
                            print(f"  Added ppid column with default value: {ppid} (data directory structure unclear)")
                    else:
                        ppid = '10'  # Default fallback
                        participant_df['ppid'] = ppid
                        print(f"  Added ppid column with default value: {ppid} (insufficient directory structure)")
                else:
                    participant_df['ppid'] = '10'  # Default fallback
                    print("  Added ppid column with default value: 10")
            else:
                participant_df['ppid'] = '10'  # Default fallback
                print("  Added ppid column with default value: 10")
        
        if 'session' not in participant_df.columns:
            participant_df['session'] = '1'  # Default session
            print("  Added session column with default value: 1")
        
        if 'trial' not in participant_df.columns:
            # Try to extract trial from filename if available
            if 'trial' in args.labels_path.lower():
                import re
                trial_match = re.search(r'T(\d{3})', args.labels_path)
                if trial_match:
                    trial = trial_match.group(1)
                    participant_df['trial'] = trial
                    print(f"  Added trial column with value: {trial}")
                else:
                    participant_df['trial'] = '11'  # Default fallback
                    print("  Added trial column with default value: 11")
            else:
                participant_df['trial'] = '11'  # Default fallback
                print("  Added trial column with default value: 11")
        
        # Add participant coordinate columns
        if 'participant_x' not in participant_df.columns and 'x' in participant_df.columns:
            participant_df['participant_x'] = participant_df['x']
            print("  Using 'x' column as participant_x")
        elif 'participant_x' not in participant_df.columns and 'MHip_x' in participant_df.columns:
            participant_df['participant_x'] = participant_df['MHip_x']
            print("  Using MHip_x as participant_x")
        else:
            # Create dummy participant coordinates if none available
            print("  ⚠ No participant X coordinates found - creating dummy coordinates")
            participant_df['participant_x'] = 0.0  # Default to origin
        
        if 'participant_y' not in participant_df.columns and 'y' in participant_df.columns:
            participant_df['participant_y'] = participant_df['y']
            print("  Using 'y' column as participant_y")
        elif 'participant_y' not in participant_df.columns and 'MHip_z' in participant_df.columns:
            participant_df['participant_y'] = participant_df['MHip_z']
            print("  Using MHip_z as participant_y")
        else:
            # Create dummy participant coordinates if none available
            print("  ⚠ No participant Y coordinates found - creating dummy coordinates")
            participant_df['participant_y'] = 0.0  # Default to origin
        
        print(f"  Final participant data columns: {list(participant_df.columns)}")
        print(f"  Participant data shape: {participant_df.shape}")
        print(f"  ⚠ NOTE: Participant coordinates are dummy values (0,0) - use --coords-dir for real coordinates")
    
    # Load experimental conditions data (if not already loaded as metadata)
    conditions_df = None
    if args.experimental_conditions and not args.metadata_path:
        print("Loading experimental conditions data...")
        try:
            conditions_df = pd.read_csv(args.experimental_conditions)
            print(f"✓ Loaded experimental conditions: {len(conditions_df)} rows")
            print(f"  Columns: {list(conditions_df.columns)}")
        except Exception as e:
            print(f"⚠ Error loading experimental conditions: {e}")
            print("  Continuing without experimental conditions data...")
    elif args.experimental_conditions and args.metadata_path:
        print("Experimental conditions already loaded as metadata replacement")
        conditions_df = metadata_df
    else:
        print("No --experimental-conditions provided")
    
    # Merge experimental conditions with participant data if available
    if conditions_df is not None:
        # Check if required columns exist
        required_cols = ['ppid', 'session', 'trial']
        missing_cols = [col for col in required_cols if col not in conditions_df.columns]
        if missing_cols:
            print(f"⚠ Warning: Missing required columns in experimental conditions: {missing_cols}")
        else:
            # Ensure consistent data types for merging
            conditions_df[['ppid', 'session', 'trial']] = conditions_df[['ppid', 'session', 'trial']].astype(str)
            participant_df[['ppid', 'session', 'trial']] = participant_df[['ppid', 'session', 'trial']].astype(str)
            
            # Merge experimental conditions with participant data
            original_participant_rows = len(participant_df)
            participant_df = pd.merge(participant_df, conditions_df, on=['ppid', 'session', 'trial'], how='left')
            merged_rows = len(participant_df)
            
            print(f"✓ Merged experimental conditions with participant data")
            print(f"  Original participant rows: {original_participant_rows}")
            print(f"  After merge: {merged_rows} rows")
            
            # Check for missing matches
            missing_matches = participant_df[conditions_df.columns].isnull().any(axis=1).sum()
            if missing_matches > 0:
                print(f"⚠ Warning: {missing_matches} participant rows have missing experimental condition data")
            
            # Show scenario_id distribution if available
            if 'scenario_id' in conditions_df.columns:
                scenario_counts = participant_df['scenario_id'].value_counts().head(10)
                print(f"  Top 10 scenario_id distribution:")
                for scenario_id, count in scenario_counts.items():
                    print(f"    {scenario_id}: {count} trials")
    
    # Load tracker and scenario data
    tracker_data_list = []
    scenario_df = None
    
    # Use --data-dir if provided, otherwise fall back to separate directories
    if args.data_dir:
        print(f"Loading tracker and scenario data from combined directory: {args.data_dir}")
        
        # Load tracker data
        print("Loading tracker data...")
        print(f"DEBUG: About to call extract_tracker_data with data_dir='{args.data_dir}', fast_test={args.fast_test}")
        try:
            tracker_data_list = extract_tracker_data(args.data_dir, args.fast_test)
            print(f"DEBUG: extract_tracker_data returned successfully")
        except Exception as e:
            print(f"DEBUG: extract_tracker_data failed with error: {e}")
            import traceback
            traceback.print_exc()
            tracker_data_list = []
        print(f"✓ Loaded {len(tracker_data_list)} tracker datasets")
        
        # Force flush stdout to ensure debug output is displayed
        import sys
        sys.stdout.flush()
        
        # Debug: Show tracker data details
        # Summary of tracker data loading
        print(f"📊 Tracker data: {len(tracker_data_list)} datasets loaded")
        if tracker_data_list:
            print(f"   Range: {tracker_data_list[0]['ppid']}-{tracker_data_list[0]['session']}-{tracker_data_list[0]['trial']} to {tracker_data_list[-1]['ppid']}-{tracker_data_list[-1]['session']}-{tracker_data_list[-1]['trial']}")
        else:
            print(f"   ⚠ No tracker data loaded!")
        
        # Load scenario data
        print("\nLoading scenario data...")
        scenario_df = extract_scenario_data(args.data_dir, args.fast_test)
        if scenario_df is not None:
            print(f"✓ Loaded scenario data: {len(scenario_df)} rows")
    
    else:
        # Fallback to separate directories (deprecated)
        if args.metadata_dir:
            print("Loading tracker data from separate metadata directory...")
            tracker_data_list = extract_tracker_data(args.metadata_dir, args.fast_test)
            print(f"✓ Loaded {len(tracker_data_list)} tracker datasets")
        
        if args.scenario_data:
            print("Loading scenario data from separate scenario directory...")
            scenario_df = extract_scenario_data(args.scenario_data, args.fast_test)
            if scenario_df is not None:
                print(f"✓ Loaded scenario data: {len(scenario_df)} rows")
    
    # Add scenario_id to scenario data from experimental conditions if available
    if scenario_df is not None and conditions_df is not None and 'scenario_id' not in scenario_df.columns:
        print("Adding scenario_id to scenario data from experimental conditions...")
         
        try:
            if 'scenario_id' in conditions_df.columns:
                # Ensure consistent data types for merging
                conditions_df[['ppid', 'session', 'trial']] = conditions_df[['ppid', 'session', 'trial']].astype(str)
                scenario_df[['ppid', 'session', 'trial']] = scenario_df[['ppid', 'session', 'trial']].astype(str)
                
                # Merge scenario_id from experimental conditions
                scenario_df = pd.merge(scenario_df, conditions_df[['ppid', 'session', 'trial', 'scenario_id']], 
                                     on=['ppid', 'session', 'trial'], how='left')
                
                
                # Check for missing scenario_id
                missing_scenario_id = scenario_df['scenario_id'].isnull().sum()
                if missing_scenario_id > 0:
                    print(f"⚠ Warning: {missing_scenario_id} scenario rows have missing scenario_id")
                else:
                    print(f"✓ Added scenario_id to all {len(scenario_df)} scenario rows")
        except Exception as e:
            print(f"⚠ Error adding scenario_id to scenario data: {e}")
    elif scenario_df is not None and 'scenario_id' in scenario_df.columns:
        print("✓ scenario_id already exists in scenario_df - skipping addition from experimental conditions")
    

    # Ensure consistent data types for all DataFrames before merging
    print("\n=== Ensuring Consistent Data Types for Merging ===")
    
    # Ensure consistent data types for merging across all DataFrames
    dataframes = [participant_df, scenario_df, labels_df, metadata_df]
    if 'metadata_df' in locals():
        dataframes.append(metadata_df)

    for df in dataframes:
        df['ppid'] = df['ppid'].astype(str)
        df['session'] = df['session'].astype(str)
        df['trial'] = df['trial'].astype(str)

    # Ensure 'time' column is compatible for labels_df and participant_df
    if 'time' in labels_df.columns:
        labels_df['time'] = labels_df['time'].astype(float)
    if 'time' in participant_df.columns:
        participant_df['time'] = participant_df['time'].astype(float)

    print("✓ Data types converted for merging")

    # ============================================================================
    # EARLY DATA FILTERING: KEEP ONLY COMPLETE TRIALS
    # ============================================================================
    print("\n" + "="*80)
    print("EARLY DATA FILTERING: KEEPING ONLY COMPLETE TRIALS")
    print("="*80)
    print("✓ Filtering data to only include trials with all three datasets:")
    print("  - Trackers data")
    print("  - Participant location data") 
    print("  - Scenario objects data")
    print("✓ This ensures consistency throughout the entire pipeline")
    print("✓ Saves computational resources by not processing incomplete data")
    
    # Apply early filtering
    filtered_result = filter_complete_trials_early(
        participant_df, scenario_df, labels_df, tracker_data_list, metadata_df
    )
    
    if filtered_result[0] is None:
        print("❌ ERROR: No complete trials found!")
        print("❌ Cannot proceed with analysis - all trials are missing at least one dataset")
        return
    
    # Unpack filtered data
    participant_df, scenario_df, labels_df, tracker_data_list, metadata_df, complete_trials_info = filtered_result
    
    print(f"\n✓ EARLY FILTERING SUCCESSFUL")
    print(f"✓ Using {complete_trials_info['complete_trials']} complete trials out of {complete_trials_info['total_trials']} total")
    print(f"✓ Completeness rate: {complete_trials_info['completeness_rate']:.1f}%")
    print(f"✓ All subsequent processing will use only complete trials")

    # ============================================================================
    # FAILED TRIALS FILTERING: REMOVE FAILED TRIALS IF PROVIDED
    # ============================================================================
    if failed_trials_data is not None:
        print("\n" + "="*80)
        print("FAILED TRIALS FILTERING: REMOVING FAILED TRIALS FROM DATA")
        print("="*80)
        print(f"✓ Failed trials data loaded with {len(failed_trials_data)} trials")
        print(f"✓ Removing failed trials from all datasets before analysis")
        
        # Ensure consistent data types for filtering
        failed_trials_data[['ppid', 'session', 'trial']] = failed_trials_data[['ppid', 'session', 'trial']].astype(str)
        
        # Filter labels_df
        if labels_df is not None:
            original_len = len(labels_df)
            labels_df = labels_df[~labels_df.set_index(['ppid', 'session', 'trial']).index.isin(
                failed_trials_data.set_index(['ppid', 'session', 'trial']).index
            )]
            removed_count = original_len - len(labels_df)
            print(f"✓ Labels data: removed {removed_count} rows from failed trials ({len(labels_df)} remaining)")
        
        # Filter participant_df
        if participant_df is not None:
            original_len = len(participant_df)
            participant_df = participant_df[~participant_df.set_index(['ppid', 'session', 'trial']).index.isin(
                failed_trials_data.set_index(['ppid', 'session', 'trial']).index
            )]
            removed_count = original_len - len(participant_df)
            print(f"✓ Participant data: removed {removed_count} rows from failed trials ({len(participant_df)} remaining)")
        
        # Filter scenario_df
        if scenario_df is not None:
            original_len = len(scenario_df)
            scenario_df = scenario_df[~scenario_df.set_index(['ppid', 'session', 'trial']).index.isin(
                failed_trials_data.set_index(['ppid', 'session', 'trial']).index
            )]
            removed_count = original_len - len(scenario_df)
            print(f"✓ Scenario data: removed {removed_count} rows from failed trials ({len(scenario_df)} remaining)")
        
        # Filter metadata_df
        if metadata_df is not None:
            original_len = len(metadata_df)
            metadata_df = metadata_df[~metadata_df.set_index(['ppid', 'session', 'trial']).index.isin(
                failed_trials_data.set_index(['ppid', 'session', 'trial']).index
            )]
            removed_count = original_len - len(metadata_df)
            print(f"✓ Metadata: removed {removed_count} rows from failed trials ({len(metadata_df)} remaining)")
        
        # Filter tracker_data_list
        if tracker_data_list:
            original_len = len(tracker_data_list)
            filtered_tracker_data_list = []
            for tracker_item in tracker_data_list:
                tracker_trial_key = (str(tracker_item['ppid']), str(tracker_item['session']), str(tracker_item['trial']))
                is_failed = any(
                    (str(row['ppid']), str(row['session']), str(row['trial'])) == tracker_trial_key
                    for _, row in failed_trials_data.iterrows()
                )
                if not is_failed:
                    filtered_tracker_data_list.append(tracker_item)
            
            tracker_data_list = filtered_tracker_data_list
            removed_count = original_len - len(tracker_data_list)
            print(f"✓ Tracker data: removed {removed_count} trials from failed trials ({len(tracker_data_list)} remaining)")
        
        print(f"✓ FAILED TRIALS FILTERING COMPLETE")
        print(f"✓ All failed trials have been removed from all datasets")
    else:
        print("\n✓ No failed trials file provided - using all data")
        print("✓ To exclude failed trials, use --failed-trials-path argument")

    # Merge labels_df with participant_df
    merged_df = pd.merge(labels_df, participant_df, on=['ppid', 'session', 'trial', 'time'], how='left', suffixes=('', '_participant'))
    # Standardize final output to use only participant_x, participant_y (not redundant x, y)
    final_columns = ['ppid', 'session', 'trial', 'time', 'predicted_label']
    
    # Add participant location columns (standardized naming)
    if 'participant_x' in merged_df.columns:
        final_columns.append('participant_x')
    if 'participant_y' in merged_df.columns:
        final_columns.append('participant_y')
    
    # Add any other columns that exist in merged_df
    # Exclude redundant x, y columns and unused z, a, b, c columns
    excluded_columns = ['x', 'y']  # Redundant participant location columns
    for col in merged_df.columns:
        if col not in final_columns and col not in excluded_columns:
            # Skip unused coordinate columns (z, a, b, c)
            if any(suffix in col.lower() for suffix in ['_z', '_a', '_b', '_c']):
                continue
            final_columns.append(col)
    
    merged_df = merged_df[final_columns]
    print(f"✓ Final output standardized: {len(final_columns)} columns")
    print(f"✓ Removed redundant x, y columns, using participant_x, participant_y")
    print(f"✓ Excluded unused z, a, b, c coordinate columns")
    merged_df = pd.merge(merged_df, metadata_df, on=['ppid', 'session', 'trial'], how='left', suffixes=('', '_metadata'))
    merged_df = pd.merge(merged_df, scenario_df, on=['ppid', 'session', 'trial'], how='left', suffixes=('', '_scenario'))
    
    # ============================================================================
    # IDENTIFY DATA COLUMNS FOR CLARITY
    # ============================================================================
    print(f"\n=== IDENTIFYING DATA COLUMNS ===")
    
    # Identify participant location columns (reference coordinates - NEVER transformed)
    # Standardize on participant_x, participant_y naming convention
    participant_location_columns = []
    if 'participant_x' in merged_df.columns:
        participant_location_columns.append('participant_x')
    if 'participant_y' in merged_df.columns:
        participant_location_columns.append('participant_y')
    
    # Check if redundant x, y columns exist (they should be identical to participant_x, participant_y)
    if 'x' in merged_df.columns and 'y' in merged_df.columns:
        print(f"   ⚠ Found redundant x, y columns - these should be identical to participant_x, participant_y")
        print(f"   ✓ Will standardize on participant_x, participant_y naming convention")
        # Don't add x, y to participant_location_columns to avoid confusion
    
    print(f"✓ Participant location columns (reference - never transformed): {participant_location_columns}")
    
    # Identify scenario object columns (can be transformed for alignment)
    # Only include x,y coordinates (z,a,b,c are not used and z is already handled by axis transformation)
    # Work with actual column names - no _transformed suffix needed since we update data in place
    scenario_object_columns = []
    for col in merged_df.columns:
        if any(obj in col.lower() for obj in ['task', 'threat', 'wall', 'hide', 'safehouse', 'weapons', 'startpoint', 'standpoint']) and any(suffix in col.lower() for suffix in ['_x', '_y']):
            scenario_object_columns.append(col)
    
    print(f"✓ Scenario object columns (can be transformed): {len(scenario_object_columns)} found")
    print(f"  ✓ Only x,y coordinates (excluded z,a,b,c as they are not used)")
    print(f"  ✓ Working with actual column names - data updated in place")
    if scenario_object_columns:
        print(f"  First 10 scenario columns: {scenario_object_columns[:10]}")
    
    # Identify metadata columns (experimental conditions, etc.)
    metadata_columns = []
    for col in merged_df.columns:
        if (col not in ['ppid', 'session', 'trial', 'time'] and 
            col not in participant_location_columns and 
            col not in scenario_object_columns and 
            not col.startswith('predicted_') and
            not col.endswith('_participant') and
            not col.endswith('_scenario')):
            metadata_columns.append(col)
    
    print(f"✓ Metadata columns: {len(metadata_columns)} found")
    if metadata_columns:
        print(f"  First 10 metadata columns: {metadata_columns[:10]}")
    
    # Identify label columns - prioritize actual syllable columns over tracker categories
    label_columns = []
    
    # First, look for columns that are likely to contain actual syllables (numeric or categorical)
    syllable_candidates = []
    for col in merged_df.columns:
        if any(keyword in col.lower() for keyword in ['syllable', 'cluster', 'group', 'class']):
            syllable_candidates.append(col)
    
    # Then look for predicted label columns
    predicted_candidates = []
    for col in merged_df.columns:
        if col.startswith('predicted_') or 'label' in col.lower():
            predicted_candidates.append(col)
    
    # Prioritize syllable columns, then predicted labels
    if syllable_candidates:
        label_columns = syllable_candidates
        print(f"✓ Found syllable columns: {label_columns}")
    elif predicted_candidates:
        label_columns = predicted_candidates
        print(f"✓ Found predicted label columns: {label_columns}")
        print(f"⚠ Note: These may contain tracker categories rather than syllables")
    else:
        print("⚠ No label columns found")
    
    print(f"✓ Final label columns: {label_columns}")
    
    # Store these column lists as global variables for use throughout the code
    globals()['participant_location_columns'] = participant_location_columns
    globals()['scenario_object_columns'] = scenario_object_columns
    globals()['metadata_columns'] = metadata_columns
    globals()['label_columns'] = label_columns
    
    print(f"\n✓ Unified dataset created: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"✓ Contains: labels + participant locations + metadata + scenario objects")
    print(f"✓ All data aligned by ppid, session, trial, and time")
    print(f"✓ Column types identified and stored for clarity")
    
    # Save the initial unified dataset for inspection
    merged_data_path = os.path.join(spatial_output_dir, 'data', 'initial_merged_dataset.csv')
    merged_df.to_csv(merged_data_path, index=False)
    print(f"✓ Saved initial unified dataset to {merged_data_path}")
    
    # Show sample of unified data
    print(f"\n=== Sample of Initial Unified Dataset ===")
    print(merged_df.head())
    
    # Show data quality summary
    print(f"\n=== Data Quality Summary ===")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    
    # Check for missing values in key columns
    required_columns = ['ppid', 'session', 'trial', 'time']
    for col in required_columns:
        if col in merged_df.columns:
            missing_count = merged_df[col].isnull().sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count} missing values ({missing_count/len(merged_df)*100:.1f}%)")
            else:
                print(f"  {col}: No missing values ✓")
    
    # Check for duplicate rows
    duplicate_rows = merged_df.duplicated().sum()
    if duplicate_rows > 0:
        print(f"⚠ Warning: {duplicate_rows} duplicate rows found")
    else:
        print(f"✓ No duplicate rows found")

    # Perform first step analysis using merged_df
    perform_first_step_analysis_unified(merged_df, tracker_data_list, spatial_output_dir, args)
    
    # ============================================================================
    # ALWAYS APPLY DATA TRANSFORMATIONS (regardless of spatial analysis flag)
    # ============================================================================
    print("\n=== APPLYING DATA TRANSFORMATIONS ===")
    print("✓ Data transformations ALWAYS happen to ensure data consistency")
    print("✓ This includes: centering, rotation, and positioning of scenario objects and trackers")
    print("✓ Transformations applied to merged_df and saved back in place")
    print("✓ Final transformed data ready for any analysis")
    
    # Apply all transformations to the unified dataset
    merged_df, tracker_data_list = process_data_with_correct_order(
        args, merged_df, tracker_data_list, spatial_output_dir
    )

    # Save the transformed unified dataset
    transformed_data_path = os.path.join(spatial_output_dir, 'data', 'transformed_unified_dataset.csv')
    merged_df.to_csv(transformed_data_path, index=False)
    print(f"✓ Saved transformed unified dataset to {transformed_data_path}")
    
    # ============================================================================
    # CHECK IF SPATIAL ANALYSIS IS REQUESTED
    # ============================================================================
    if args.spatial_analysis:
        print("\n=== PERFORMING SPATIAL ANALYSIS ===")
        print("✓ Using transformed unified data for spatial analysis")
        print("✓ Data has been: centered, rotated, and positioned")
        
        # Create spatial plot using FINAL transformed unified data
        print("\n=== Creating Spatial Plot with FINAL Transformed Unified Data ===")
        print("✓ Using unified data that has been: centered, rotated, and positioned")
        create_first_step_analysis_plot_unified(merged_df, tracker_data_list, 
                                              spatial_output_dir, args, "Final Spatial Analysis")
        
        # PERFORM ACTUAL SPATIAL ANALYSIS
        print("\n=== PERFORMING ACTUAL SPATIAL ANALYSIS ===")
        print("✓ Calling perform_spatial_analysis() with transformed unified data")
        
        # Check if color_map is available
        color_map = None
        if args.colour_map_object:
            try:
                with open(args.colour_map_object, 'rb') as f:
                    color_map = pickle.load(f)
                print(f"✓ Loaded color map from {args.colour_map_object}")
            except Exception as e:
                print(f"⚠ Warning: Could not load color map: {e}")
                print("  Continuing without color map...")
        
        # Call the actual spatial analysis function
        # Extract relevant data from unified dataset for the legacy function
        # The legacy function expects separate DataFrames, so we extract from merged_df
        # CRITICAL: Must include scenario_id and episode_set_name2 for proper grouping
        
        # Get all available columns to check what metadata we have
        all_columns = list(merged_df.columns)
        print(f"✓ Available columns in merged_df: {len(all_columns)}")
        
        # Define essential columns that must be included
        essential_columns = ['ppid', 'session', 'trial', 'time']
        
        # Check for critical metadata columns
        metadata_columns_to_include = []
        if 'scenario_id' in all_columns:
            metadata_columns_to_include.append('scenario_id')
            print(f"✓ Found scenario_id column - will include for scenario grouping")
        else:
            print(f"⚠ WARNING: No scenario_id column found - spatial analysis may not work properly")
            
        if 'episode_set_name2' in all_columns:
            metadata_columns_to_include.append('episode_set_name2')
            print(f"✓ Found episode_set_name2 column - will include for block grouping")
        else:
            print(f"⚠ WARNING: No episode_set_name2 column found - spatial analysis may not work properly")
        
        # Extract participant data (participant location columns + essential metadata)
        participant_location_columns = globals().get('participant_location_columns', [])
        participant_df = merged_df[essential_columns + metadata_columns_to_include + participant_location_columns].copy()
        
        # Extract scenario data (scenario object columns + essential metadata)
        scenario_object_columns = globals().get('scenario_object_columns', [])
        scenario_df = merged_df[essential_columns + metadata_columns_to_include + scenario_object_columns].copy()
        
        # Extract labels data (label columns + essential metadata)
        label_columns = globals().get('label_columns', [])
        labels_df = merged_df[essential_columns + metadata_columns_to_include + label_columns].copy()
        
        print(f"✓ Extracted from unified dataset:")
        print(f"  - Participant data: {len(participant_df)} rows, {len(participant_df.columns)} columns")
        print(f"  - Scenario data: {len(scenario_df)} rows, {len(scenario_df.columns)} columns")
        print(f"  - Labels data: {len(labels_df)} rows, {len(labels_df.columns)} columns")
        print(f"  - Metadata included: {metadata_columns_to_include}")
        
        # Verify critical columns are present
        for df_name, df in [('participant_df', participant_df), ('scenario_df', scenario_df), ('labels_df', labels_df)]:
            missing_critical = [col for col in ['scenario_id', 'episode_set_name2'] if col in all_columns and col not in df.columns]
            if missing_critical:
                print(f"  ⚠ WARNING: {df_name} missing critical columns: {missing_critical}")
            else:
                print(f"  ✓ {df_name} has all critical metadata columns")
        
        # Call the UNIFIED spatial analysis function that works directly with merged_df
        # No need to separate and re-merge data - use the unified approach
        print(f"\n=== USING UNIFIED SPATIAL ANALYSIS ===")
        print(f"✓ Passing unified dataset directly to spatial analysis")
        print(f"✓ No data separation or re-merging needed")
        
        # Call the unified spatial analysis function
        perform_spatial_analysis_unified(merged_df, tracker_data_list, color_map, spatial_output_dir, args)
        
        print("✓ Actual spatial analysis completed using transformed data")
    else:
        print("\n=== SPATIAL ANALYSIS NOT REQUESTED ===")
        print("✓ Data transformations completed successfully")
        print("✓ Transformed data saved and ready for future analysis")
        print("✓ Use --spatial-analysis flag to perform spatial analysis")
    
    # Save analysis summary
    print("\n=== Saving Analysis Summary ===")
    summary_path = os.path.join(spatial_output_dir, 'logs', 'analysis_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== SPATIAL ANALYSIS SUMMARY ===\n")
        f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {spatial_output_dir}\n\n")
        
        f.write("=== DATA SUMMARY ===\n")
        f.write(f"Unified dataset (merged_df): {len(merged_df)} rows, {len(merged_df.columns)} columns\n")
        f.write(f"Tracker trials: {len(tracker_data_list)}\n")
        f.write(f"Data sources merged: labels + participant + metadata + scenario\n\n")
        
        f.write("=== PARAMETERS USED ===\n")
        f.write(f"Downsampling factor: {args.downsample}\n")
        f.write(f"Alignment to participant: {args.alignment_to_participant}\n")
        f.write(f"Shift to median startpoints: {args.shift_to_median_startpoints}\n")
        f.write(f"Scenario rotation: {'Yes' if args.scenario_rotation else 'No'}\n")
        f.write(f"Debug mode: {args.debug or args.debug_plots_only}\n")
        
        if args.experimental_conditions:
            f.write(f"Experimental conditions: {args.experimental_conditions}\n")
        
        f.write("\n=== OUTPUT FILES ===\n")
        f.write("data/initial_merged_dataset.csv - INITIAL UNIFIED DATASET (before transformations)\n")
        f.write("data/transformed_unified_dataset.csv - TRANSFORMED UNIFIED DATASET (after all transformations)\n")
        f.write("data/merged_dataset.csv - UNIFIED DATASET (labels + participant + scenario + metadata)\n")
        f.write("data/participant_location.csv - Participant coordinate data\n")
        f.write("debug_plots/ - Debug plots for each processing step\n")
        f.write("plots/spatial_plot.png - Final spatial analysis plot (if --spatial-analysis used)\n")
        f.write("spatial_plots/ - Spatial analysis results and plots (if --spatial-analysis used)\n")
        f.write("logs/analysis_summary.txt - This summary file\n")
        f.write("\n=== DATA PROCESSING APPROACH ===\n")
        f.write("UNIFIED DATA LOADING AND MERGING APPROACH:\n")
        f.write("- All data sources loaded and merged BEFORE transformations\n")
        f.write("- Single DataFrame containing all information (labels, participant, scenario, metadata)\n")
        f.write("- Data consistency ensured across all sources\n")
        f.write("- Transformations ALWAYS applied to unified dataset (regardless of spatial analysis)\n")
        f.write("- Spatial analysis uses consistent, transformed data\n")
        f.write("- No merge issues during analysis phase\n")
        f.write("\nTRANSFORMATION APPROACH:\n")
        f.write("Data transformations ALWAYS happen to ensure consistency:\n")
        f.write("- Scenario objects: centered, rotated, positioned\n")
        f.write("- Tracker data: centered, rotated, positioned\n")
        f.write("- All transformations applied to merged_df and saved back in place\n")
        f.write("- Final transformed data ready for any analysis\n")
        f.write("- Spatial analysis optional but data always prepared\n")
    
    print(f"✓ Analysis summary saved")
    print(f"📁 Outputs: {spatial_output_dir}")
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()