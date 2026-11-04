import os
import pdb
from get_syllable_transition_matrices import plot_matrix, encode_and_get_sequences, restructure_matrix
from pomegranate import MarkovChain
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from pickle import dump
import time


def tictoc(func):
    def inner(*args, **kwargs):
        # Get just the filename without path and extension
        current_file = os.path.splitext(os.path.basename(func.__code__.co_filename))[0].replace("_", " ").title()
        
        # Use time.perf_counter() for high-resolution timing
        T1 = time.perf_counter()
        T1_str = time.strftime("%H:%M:%S", time.localtime())
        print(f"\033[94m|{current_file}| START: {T1_str}\033[0m")
        func_out = func(*args, **kwargs)
        T2 = time.perf_counter()
        T2_str = time.strftime("%H:%M:%S", time.localtime())
        print(f"\033[94m|{current_file}| END: {T2_str}\033[0m")
        
        # Calculate elapsed time in seconds and format it
        elapsed_seconds = T2 - T1
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        T = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        print(f"\n\033[94m|{current_file}| TOTAL RUN TIME: {T}\033[0m\n")
        return func_out

    return inner


def pad_or_truncate_list(lst, n=8):
    return lst[:n] + [None] * max(0, n - len(lst))

def pad_or_truncate_list2(lst, n=4):
    return lst[:n] + [None] * max(0, n - len(lst))

def get_data(metadata_file, ppid_column: str, session_column: str, trial_column: str, save_processed_data=False, label_column: str = "predicted_label") -> tuple[pd.DataFrame, list]:
    target_columns = [
        ppid_column,
        session_column,
        trial_column,
        "time",
        "ppid_vr",
        "session_vr",
        "trial_vr",
        "ppid_coda",
        "session_coda",
        "trial_coda",
        "experiment",
        "start_time",
        "end_time",
        "episode",
        "episode_set_name",
        "timeout",
        "rewards",
        "end_state",
        "reward",
        "bank",
        "distance",
        "threat_action",
        "weapons",
        "safehouse",
        "day_time"
    ]

    dictionary_target_columns = ["scenario", "threat", "visibility", "distance", "threat_action", "weapons", "safehouse", "day_time"]
    # instructed_target_columns = ["scenario", "scenario_id", "dummy", "instructed_action"]

    target_columns = target_columns + dictionary_target_columns 

    target_columns = list(set(target_columns))

    df = pd.read_csv(metadata_file)
    label_column_df = df[label_column]

    columns_aready_present = all(col in df.columns for col in dictionary_target_columns) 

    target_columns = [col for col in target_columns if col in df.columns]

    df = df[target_columns]
    # Get the list of columns to move to the front
    front_columns = [ppid_column, session_column, trial_column, "time"] 
    
    # Get the remaining columns
    remaining_columns = [col for col in df.columns if col not in front_columns]
    
    # Reorder the columns
    df = df[front_columns + remaining_columns]

    # matrix_files_list = [
    #     i
    #     for i in os.listdir(matrix_files)
    #     if i.endswith(".csv") and "transition_matrix_P" in i
    # ]

    if columns_aready_present:
        print(f"\nColumns already present in the dataframe. Processing data...\n")
        # Create a copy of the dataframe for each set
        df["episode_set_name"] = "dictionary"
       
    else:
        df["episode_set_name"] = df["episode_set_name"].replace(
            {
                "Dictionary of Actions": "dictionary",
                'Dictionary of Actions Replication - Main': 'dictionary',
                'Dictionary of Actions Replication - Fixed Trials': 'dictionary',
                'Dictionary of Actions Replication - Stump Trials': 'dictionary',
                "Dictionary of Actions - Instructed Actions": "instructed",
            }
        )

        # Split the 'episode' column values based on "_"
        df["episode_split"] = df["episode"].str.split("_")

        # Create separate dataframes for dictionary and instructed sets
        dictionary_df = df.loc[df["episode_set_name"] == "dictionary"].copy()
        
        dictionary_df[dictionary_target_columns] = pd.DataFrame(
                dictionary_df["episode_split"].apply(pad_or_truncate_list).to_list(), index=dictionary_df.index)

        # # If 'episode_set_name' is "instructed", add the split values to new columns
        # instructed_df[instructed_target_columns] = pd.DataFrame(
        #     instructed_df["episode_split"].apply(pad_or_truncate_list2).to_list(), index=instructed_df.index)

        # Concatenate the dataframes
        df = pd.concat([dictionary_df])

        # Drop the 'episode_split' and 'dummy' columns
        df = df.drop(columns=["episode_split", "dummy"])

    df[label_column] = label_column_df

    if save_processed_data:
        df.to_csv(
            metadata_file.replace(".csv", "_processed.csv"), index=False
        )

    return df

# Function to flatten transition matrix
def flatten_matrix(matrix):
    return matrix.values.flatten()

def align_matrices(matrix1, template_matrix):
    # Convert all syllables to strings
    matrix1.columns = matrix1.columns.astype(str)
    matrix1.index = matrix1.index.astype(str)
    template_matrix.columns = template_matrix.columns.astype(str)
    template_matrix.index = template_matrix.index.astype(str)
    
    # Get all unique syllables from both matrices
    all_syllables = sorted(set(matrix1.columns) | set(template_matrix.columns))
    
    # Create new matrices with the same shape and float dtype
    aligned_matrix1 = pd.DataFrame(0.0, index=all_syllables, columns=all_syllables, dtype=float)
    
    # Fill in the values from original matrices
    for syllable in all_syllables:
        if syllable in matrix1.index and syllable in matrix1.columns:
            # Get the row for this syllable from the original matrix
            row = matrix1.loc[syllable]
            # Only copy values for syllables that exist in the original matrix
            for target_syllable in row.index:
                if target_syllable in all_syllables:
                    aligned_matrix1.loc[syllable, target_syllable] = float(row[target_syllable])
    
    return aligned_matrix1

# Function to create transition matrix for a subset of data
def create_transition_matrix(data_subset, label_col, template_matrix):
    # Get sequences and label encoder
    sequences, le = encode_and_get_sequences(data_subset, label_col)
    
    # Create Markov chain model
    model = MarkovChain.from_samples(sequences)
    matrix = restructure_matrix(model)
    
    # Create a new matrix with the template structure
    new_matrix = template_matrix.copy()
    
    # Fill in the values from the actual transitions
    for syllable in matrix.columns:
        if syllable in new_matrix.columns:
            new_matrix.loc[syllable, matrix.columns] = matrix.loc[syllable]
    
    # Normalize the matrix to get probabilities
    new_matrix = new_matrix.div(new_matrix.sum(axis=1), axis=0).fillna(0)
    
    return new_matrix


def get_transition_matrix(data, label_column, output_dir, save_model: bool = True, save_matrix: bool = True, file_name_prefix: str = "transition_matrix", save_matrix_figure: bool = True):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequences, le = encode_and_get_sequences(data, label_column)

    # Create a Markov chain model
    model = MarkovChain.from_samples(sequences)
    if save_model:
        # Save the model
        model_filename = os.path.join(output_dir, f'{file_name_prefix}_model_with_self_transitions.pkl')
        dump(model, model_filename)
    matrix_with_self_transitions = restructure_matrix(model)

    encoded_labels = matrix_with_self_transitions.columns
    decoded_labels = le.inverse_transform(encoded_labels)
    matrix_with_self_transitions.columns = decoded_labels
    matrix_with_self_transitions.index = decoded_labels
    if save_matrix_figure:
        plot_matrix(matrix_with_self_transitions, labels=matrix_with_self_transitions.columns, output_dir=output_dir, title=f"{file_name_prefix} including self-transitions")
    if save_matrix:
        matrix_with_self_transitions.to_csv(os.path.join(output_dir, f'{file_name_prefix}_with_self_transitions.csv'))

    new_sequence = [sequences[0][0]]
    for i in range(1, len(sequences[0])):
        if sequences[0][i] != sequences[0][i - 1]:
            new_sequence.append(sequences[0][i])

    data2 = pd.DataFrame(new_sequence, columns=[label_column])

    sequences2, le2 = encode_and_get_sequences(data2, label_column)
    # Create a Markov chain model
    model2 = MarkovChain.from_samples(sequences2)
    if save_model:
        # Save the model
        model_filename2 = os.path.join(output_dir, f'{file_name_prefix}_model_without_self_transitions.pkl')
        dump(model2, model_filename2)

    matrix_without_self_transitions = restructure_matrix(model2)
    matrix_without_self_transitions.columns = decoded_labels
    matrix_without_self_transitions.index = decoded_labels
    if save_matrix_figure:
        plot_matrix(matrix_without_self_transitions, labels=matrix_without_self_transitions.columns, output_dir=output_dir, title=f"{file_name_prefix} excluding self-transitions")
    if save_matrix:
        matrix_without_self_transitions.to_csv(os.path.join(output_dir, f'{file_name_prefix}_excluding_self_transitions.csv'))

    return matrix_with_self_transitions, matrix_without_self_transitions


# Function to perform permutation test
def permutation_test(condition1_trials, condition2_trials, template_matrix, label_col, output_dir, group, level1, level2, n_permutations=1000):
    # Calculate reference matrices for each condition
    ref_matrix1 = pd.DataFrame(0.0, index=template_matrix.index, columns=template_matrix.columns, dtype=float)
    ref_matrix2 = pd.DataFrame(0.0, index=template_matrix.index, columns=template_matrix.columns, dtype=float)
    
    print(f"Template matrix shape: {template_matrix.shape}")
    
    # Concatenate all trials for each condition
    condition1_data = pd.concat(condition1_trials)
    condition2_data = pd.concat(condition2_trials)
    
    # Get reference matrices
    ref_matrix1, _ = get_transition_matrix(
        condition1_data, 
        label_col, 
        output_dir, 
        save_model=False, 
        save_matrix=False, 
        file_name_prefix=f"ref_matrix_{group}_{level1}"
    )
    aligned_ref_matrix1 = align_matrices(ref_matrix1, template_matrix)
    
    ref_matrix2, _ = get_transition_matrix(
        condition2_data, 
        label_col, 
        output_dir, 
        save_model=False, 
        save_matrix=False, 
        file_name_prefix=f"ref_matrix_{group}_{level2}"
    )
    aligned_ref_matrix2 = align_matrices(ref_matrix2, template_matrix)
    
    # Get all trials for each condition
    all_trials1 = []
    all_trials2 = []
    
    for trial_data in condition1_trials:
        trial_matrix, _ = get_transition_matrix(
            trial_data, 
            label_col, 
            output_dir, 
            save_model=False, 
            save_matrix=False, 
            file_name_prefix=f"trial_matrix_{group}_{level1}",
            save_matrix_figure=False,
        )
        aligned_trial_matrix1 = align_matrices(trial_matrix, template_matrix)
        all_trials1.append(aligned_trial_matrix1)
    
    for trial_data in condition2_trials:
        trial_matrix, _ = get_transition_matrix(
            trial_data, 
            label_col, 
            output_dir, 
            save_model=False, 
            save_matrix=False, 
            file_name_prefix=f"trial_matrix_{group}_{level2}",
            save_matrix_figure=False,
        )
        aligned_trial_matrix2 = align_matrices(trial_matrix, template_matrix)
        all_trials2.append(aligned_trial_matrix2)
    
    # Calculate observed test statistic
    # For each trial, calculate distance to both reference matrices
    distances1 = []
    for trial in all_trials1:
        trial_vec = flatten_matrix(trial)
        ref_vec = flatten_matrix(aligned_ref_matrix1)
        if len(trial_vec) != len(ref_vec):
            print(f"Warning: Shape mismatch - trial: {trial_vec.shape}, ref: {ref_vec.shape}")
            continue
        # Calculate Euclidean distance between trial and reference vectors
        # This measures how far each trial's transition probabilities are from the reference matrix
        distances1.append(np.linalg.norm(trial_vec - ref_vec))
    
    distances2 = []
    for trial in all_trials2:
        trial_vec = flatten_matrix(trial)
        ref_vec = flatten_matrix(aligned_ref_matrix2)
        if len(trial_vec) != len(ref_vec):
            print(f"Warning: Shape mismatch - trial: {trial_vec.shape}, ref: {ref_vec.shape}")
            continue
        distances2.append(np.linalg.norm(trial_vec - ref_vec))
    
    if not distances1 or not distances2:
        print("Warning: No valid distances calculated")
        return 0.0, 1.0, []
    
    # Calculate mean distance for each condition
    mean_dist1 = np.mean(distances1)
    mean_dist2 = np.mean(distances2)
    observed_diff = mean_dist1 - mean_dist2
    
    # Combine all trials for permutation
    all_trials = all_trials1 + all_trials2
    all_trial_data = condition1_trials + condition2_trials
    n1 = len(all_trials1)
    
    # Perform permutations
    perm_diffs = []
    for _ in range(n_permutations):
        # Create a random permutation of indices
        perm_indices = np.random.permutation(len(all_trials))
        
        # Split into two groups using indices
        perm_indices1 = perm_indices[:n1]
        perm_indices2 = perm_indices[n1:]
        
        # Get the permuted trial data
        perm_data1 = pd.concat([all_trial_data[i] for i in perm_indices1])
        perm_data2 = pd.concat([all_trial_data[i] for i in perm_indices2])

        
        # Calculate new reference matrices using get_transition_matrix
        perm_ref1, _ = get_transition_matrix(
            perm_data1,
            label_col,
            output_dir,
            save_model=False,
            save_matrix=False,
            file_name_prefix=f"perm_ref_matrix_{group}_{level1}",
            save_matrix_figure=False
        )
        aligned_perm_ref1 = align_matrices(perm_ref1, template_matrix)
        
        perm_ref2, _ = get_transition_matrix(
            perm_data2,
            label_col,
            output_dir,
            save_model=False,
            save_matrix=False,
            file_name_prefix=f"perm_ref_matrix_{group}_{level2}",
            save_matrix_figure=False
        )
        aligned_perm_ref2 = align_matrices(perm_ref2, template_matrix)
        
        # Calculate distances
        perm_dist1 = []
        for i in perm_indices1:
            trial = all_trials[i]
            trial_vec = flatten_matrix(trial)
            ref_vec = flatten_matrix(aligned_perm_ref1)
            if len(trial_vec) == len(ref_vec):
                perm_dist1.append(np.linalg.norm(trial_vec - ref_vec))

        
        perm_dist2 = []
        for i in perm_indices2:
            trial = all_trials[i]
            trial_vec = flatten_matrix(trial)
            ref_vec = flatten_matrix(aligned_perm_ref2)
            if len(trial_vec) == len(ref_vec):
                perm_dist2.append(np.linalg.norm(trial_vec - ref_vec))
        
        if not perm_dist1 or not perm_dist2:
            continue
        
        # Calculate mean distances
        perm_mean1 = np.mean(perm_dist1)
        perm_mean2 = np.mean(perm_dist2)
        perm_diff = perm_mean1 - perm_mean2
        
        perm_diffs.append(perm_diff)
    
    if not perm_diffs:
        print("Warning: No valid permutations calculated")
        return 0.0, 1.0, []
    
    # Calculate p-value
    p_value = np.mean(np.array(perm_diffs) >= observed_diff)
    
    plt.close('all')
    
    return observed_diff, p_value, perm_diffs

@tictoc
def main():
    print("\033[34mCode has started running\033[0m\n")
    import argparse

    parser = argparse.ArgumentParser(description='Process transition matrices with metadata')
    parser.add_argument('--metadata', required=True, help='Path to the metadata CSV file')
    parser.add_argument('--output-dir', required=False, default=None, help='Path to the output directory')
    parser.add_argument('--ppid', required=False, default="ppid", help='Name of the ppid column')
    parser.add_argument('--session', required=False, default="session", help='Name of the session column')
    parser.add_argument('--trial', required=False, default="trial", help='Name of the trial column')
    parser.add_argument('--label', required=False, default="predicted_label", help='Name of the label column')

    args = parser.parse_args()
    
    metadata_file = args.metadata
    ppid_column = args.ppid
    session_column = args.session
    trial_column = args.trial
    label_column = args.label
    target_groups = ["end_state", "scenario", "threat", "visibility", "distance", "threat_action", "weapons", "safehouse", "day_time"]

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.metadata), 'random_permutation_test_output')
    else:
        output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    df = get_data(
        metadata_file, ppid_column, session_column, trial_column, label_column = label_column, save_processed_data=True
    )

    # Get the list of participants
    dictionary_df = df.copy()

    # First, get all unique syllables from the entire dataset and convert to strings
    all_syllables = sorted([str(s) for s in dictionary_df[label_column].unique()])
    print(f"\nTotal unique syllables: {len(all_syllables)}")
    print(f"Sample syllables: {list(all_syllables)[:5]}...")
    
    # Create a template matrix with all possible syllables
    template_matrix = pd.DataFrame(0.0, index=all_syllables, columns=all_syllables, dtype=float)
    
    # Perform permutation tests for each target group
    results = {}
    for group in target_groups:
        print(f"\n\033[34mAnalyzing group: {group}\033[0m")
        
        # Get unique levels of the group, excluding NaN
        levels = dictionary_df[group].dropna().unique()
        if len(levels) < 2:
            print(f"Skipping {group} - only one level present")
            continue
            
        # Print data statistics for this group
        total_obs = len(dictionary_df)
        nan_obs = dictionary_df[group].isna().sum()
        non_nan_obs = total_obs - nan_obs
        print(f"\nData statistics for {group}:")
        print(f"Total observations: {total_obs}")
        print(f"NaN observations: {nan_obs} ({nan_obs/total_obs*100:.2f}%)")
        print(f"Non-NaN observations: {non_nan_obs} ({non_nan_obs/total_obs*100:.2f}%)")
        print(f"Unique levels: {len(levels)}")
        
        # Get trials for each level
        level_trials = {}
        for level in levels:
            level_data = dictionary_df[dictionary_df[group] == level]
            print(f"\nProcessing level: {level}")
            print(f"Number of samples for {level}: {len(level_data)}")
            
            if len(level_data) < 10:  # Skip levels with too few samples
                print(f"Warning: Skipping {level} due to insufficient samples ({len(level_data)} < 10)")
                continue
            
            # Group by trial
            trials = level_data.groupby([ppid_column, session_column, trial_column])
            trial_data_list = [group for _, group in trials]
            
            if len(trial_data_list) < 2:  # Need at least 2 trials for comparison
                print(f"Warning: Skipping {level} due to insufficient trials ({len(trial_data_list)} < 2)")
                continue
            
            level_trials[level] = trial_data_list
            print(f"Number of trials: {len(trial_data_list)}")
        
        if len(level_trials) < 2:
            print(f"Skipping {group} - less than 2 valid levels after filtering")
            continue
        
        # Perform pairwise comparisons
        level_pairs = [(a, b) for i, a in enumerate(level_trials.keys()) for b in list(level_trials.keys())[i+1:]]
        for level1, level2 in level_pairs:
            print(f"\nComparing {level1} vs {level2}")
            print(f"Number of trials in {level1}: {len(level_trials[level1])}")
            print(f"Number of trials in {level2}: {len(level_trials[level2])}")
            
            diff, p_value, _ = permutation_test(
                level_trials[level1], 
                level_trials[level2], 
                template_matrix, 
                label_column,
                output_dir,
                group,
                level1,
                level2
            )
            print(f"Observed difference: {diff:.4f}")
            print(f"p-value: {p_value:.4f}")
            results[f"{group}_{level1}_vs_{level2}"] = {
                "difference": diff,
                "p_value": p_value,
                "n_trials_1": len(level_trials[level1]),
                "n_trials_2": len(level_trials[level2])
            }
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, "permutation_test_results.csv"))
    
    print("\n\033[34mThe code has finished running\033[0m\n")


if __name__ == "__main__":
    os.system("cls||clear")
    print("\n\033[35mcode by uSerratos\033[0m\n\n")
    main()
