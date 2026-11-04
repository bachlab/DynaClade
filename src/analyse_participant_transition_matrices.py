import os
import pdb
from get_syllable_transition_matrices import plot_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd


def pad_or_truncate_list(lst, n=8):
    return lst[:n] + [None] * max(0, n - len(lst))

def pad_or_truncate_list2(lst, n=4):
    return lst[:n] + [None] * max(0, n - len(lst))

def get_data(metadata_file, matrix_files,  ppid_column: str, session_column: str, trial_column: str, save_processed_data=False,) -> tuple[pd.DataFrame, list]:
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
        "video_length_stickfigure",
        "video_length_obs",
        "video_difference",
        "notes",
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
    instructed_target_columns = ["scenario", "instructed_action"]

    target_columns = target_columns + dictionary_target_columns + instructed_target_columns

    target_columns = list(set(target_columns))

    df = pd.read_csv(metadata_file)

    columns_aready_present = all(col in df.columns for col in dictionary_target_columns) and all(col in df.columns for col in instructed_target_columns)

    target_columns = [col for col in target_columns if col in df.columns]

    df = df[target_columns]
    # Get the list of columns to move to the front
    front_columns = [ppid_column, session_column, trial_column, "time"] 
    
    # Get the remaining columns
    remaining_columns = [col for col in df.columns if col not in front_columns]
    
    # Reorder the columns
    df = df[front_columns + remaining_columns]

    matrix_files_list = [
        i
        for i in os.listdir(matrix_files)
        if i.endswith(".csv") and "transition_matrix_P" in i
    ]

    if columns_aready_present:
        print(f"\nColumns already present in the dataframe. Processing data...\n")
        # Create a copy of the dataframe for each set
        
        dictionary_df = df[df["instructed_action"] == "dictionary"].copy()
        instructed_df = df[df["instructed_action"] == "instructed"].copy()
        # Concatenate the dataframes
        df = pd.concat([dictionary_df, instructed_df])
    else:
        df["episode_set_name"] = df["episode_set_name"].replace(
            {
                "Dictionary of Actions": "dictionary",
                "Dictionary of Actions - Instructed Actions": "instructed",
            }
        )

        # Split the 'episode' column values based on "_"
        df["episode_split"] = df["episode"].str.split("_")

        # Create separate dataframes for dictionary and instructed sets
        dictionary_df = df.loc[df["episode_set_name"] == "dictionary"].copy()
        instructed_df = df.loc[df["episode_set_name"] == "instructed"].copy()
        
        dictionary_df[dictionary_target_columns] = pd.DataFrame(
                dictionary_df["episode_split"].apply(pad_or_truncate_list).to_list(), index=dictionary_df.index)

        instructed_target_columns = ["scenario", "scenario_id", "dummy", "instructed_action"]

        # If 'episode_set_name' is "instructed", add the split values to new columns
        instructed_df[instructed_target_columns] = pd.DataFrame(
            instructed_df["episode_split"].apply(pad_or_truncate_list2).to_list(), index=instructed_df.index)

        # Concatenate the dataframes
        df = pd.concat([dictionary_df, instructed_df])

        # Drop the 'episode_split' and 'dummy' columns
        df = df.drop(columns=["episode_split", "dummy"])

    if save_processed_data:
        df.to_csv(
            metadata_file.replace(".csv", "_processed.csv"), index=False
        )

    return df, matrix_files_list


def get_group_to_matrix_mapping_dictionary(grouped_data: pd.DataFrame, matrix_files_list: list, ppid_column: str, session_column: str, trial_column: str) -> dict:
    current_target_columns = [
        'end_state', 'scenario',
        'threat', 'visibility', 'distance', 
        'threat_action', 'weapons', 'safehouse', 'day_time'
    ]
    transition_matrix_current_files = {"group": {}, "ppid": {}, "session": {}, "trial": {}}
    for column in current_target_columns:
        transition_matrix_current_files[column] = {}
    for participant, group in tqdm(grouped_data, desc="Getting groups", leave=False):
        for i in tqdm(range(len(group)), desc="Getting files", leave=False):
            ppid = "P{:03d}".format(int(group.iloc[i][ppid_column]))
            session = "S{:03d}".format(int(group.iloc[i][session_column]))
            trial = "T{:03d}".format(int(group.iloc[i][trial_column]))
            transition_matrix_current_file = [
                i
                for i in matrix_files_list
                if f"{ppid}_{session}_{trial}" in i
            ][0]

            pdb.set_trace()
            group_key = (
                group.iloc[i][ppid_column],
                group.iloc[i][session_column],
                group.iloc[i][trial_column],
            )
            transition_matrix_current_files["group"].setdefault(group_key, []).append(transition_matrix_current_file)
            transition_matrix_current_files["ppid"].setdefault(ppid, []).append(transition_matrix_current_file)
            transition_matrix_current_files["session"].setdefault(session, []).append(transition_matrix_current_file)
            transition_matrix_current_files["trial"].setdefault(trial, []).append(transition_matrix_current_file)

            for column in current_target_columns:
                column_value = group.iloc[i][column]
                transition_matrix_current_files[column].setdefault(column_value, []).append(transition_matrix_current_file)

    return transition_matrix_current_files


def compute_mean_transition_matrix(transition_matrix_dict: dict, matrix_files_dir:str, get_plots: bool = False, sub_name:str = None) -> str:
    output_dir0 = os.path.join(matrix_files_dir, "mean_transition_matrices")
    if sub_name is not None:
        output_dir0 = os.path.join(output_dir0, f"{sub_name}")

    os.makedirs(output_dir0, exist_ok=True)

    for key in transition_matrix_dict:
        output_dir1 = os.path.join(output_dir0, f"{key}")
        os.makedirs(output_dir1, exist_ok=True)
        # Iterate over each subkey
        for subkey in tqdm(transition_matrix_dict[key], desc=f"Computing mean transition matrices for {key}", leave=False):
            # Read the CSV files into 3D numpy array
            arrays = [
                pd.read_csv(os.path.join(matrix_files_dir, file), index_col=0).values[:,:,np.newaxis]
                for file in transition_matrix_dict[key][subkey] if not pd.isna(subkey)
            ]

            n = len(arrays)
            print(f"\nProcessing {key}--{subkey} with {n} matrices")

            pdb.set_trace()

            array_concat = np.concatenate(arrays, axis=2)
            if array_concat.shape[2] > 1:
                array_mean = array_concat.mean(axis=2)
                array_std = array_concat.std(axis=2)  # Compute standard deviation

                # Compute correlation for arrays with more than 1 element
                # array_corr = np.corrcoef(array_concat.reshape(array_concat.shape[2], -1))

                row_sums = array_mean.sum(axis=1, keepdims=True)
                # Handle division by zero properly
                array_mean_normalised = np.zeros_like(array_mean)
                valid_rows = (row_sums != 0).flatten()  # Flatten to match array dimensions
                array_mean_normalised[valid_rows, :] = array_mean[valid_rows, :] / row_sums[valid_rows]
            else:
                array_mean_normalised = array_concat[:,:,0]
                array_std = np.zeros_like(array_mean_normalised)  # No variation in a single-element array
                # array_corr = np.ones((1, 1))  # Correlation of a single-element array with itself is 1

            current_file = os.path.join(matrix_files_dir, transition_matrix_dict[key][subkey][0])
            df = pd.read_csv(current_file, index_col=0)
            df_mean = pd.DataFrame(array_mean_normalised, index=df.index, columns=df.columns)
            output_filename = f"{key}--{subkey}--_mean_transition_matrix.csv"
            output_filename_full = os.path.join(output_dir1, f"{key}--{subkey}--n{n}_mean_transition_matrix.csv")
            df_mean.to_csv(output_filename_full, index=True)

            # Save the standard deviation and correlation matrices
            df_std = pd.DataFrame(array_std, index=df.index, columns=df.columns)

            # Assume `array3d` is your 3D numpy array
            array2d = array_concat.reshape(array_concat.shape[0], -1)  # Reshape the array into 2D
            
            # Handle zero standard deviation cases
            stddev = np.std(array2d, axis=1)
            mask = stddev == 0
            if np.any(mask):
                # Replace rows with zero stddev with NaN
                array2d[mask] = np.nan
            
            correlation_matrix = np.corrcoef(array2d)  # Compute the correlation coefficients
            # Create a DataFrame filled with the scalar value from array_corr
            # df_corr = pd.DataFrame(np.full((len(df.index), len(df.columns)), array_corr), index=df.index, columns=df.columns)

            df_std.to_csv(os.path.join(output_dir1, f"{key}--{subkey}--n{n}_std_transition_matrix.csv"), index=True)
            # df_corr.to_csv(os.path.join(output_dir1, f"{key}_{subkey}_n{n}_corr_transition_matrix.csv"), index=True)

            if get_plots:
                plot_matrix(df_mean, labels=df_mean.columns, output_dir=output_dir1, title=output_filename.replace('_', ' ').replace('.csv', '') + f'(n = {n})')

    return output_dir1, output_dir0


def main():
    metadata_file = input("\nEnter the path to the metadata CSV file: ")
    matrix_files = input(
        "\nEnter the path to the directory containing the transition matrix CSV files: "
    )
    ppid_column = input("\nEnter the name of the ppid column: ")
    session_column = input("\nEnter the name of the session column: ")
    trial_column = input("\nEnter the name of the trial column: ")

    df, matrix_files_list = get_data(
        metadata_file, matrix_files, ppid_column, session_column, trial_column, save_processed_data=True
    )


    dictionary_df = df[df["instructed_action"] == "dictionary"]
    instructed_df = df[df["instructed_action"] == "instructed"]

    # Group the data by participant and trial
    grouped_dictionary = dictionary_df.groupby([ppid_column])
    grouped_instructed = instructed_df.groupby([ppid_column])

    # Initialize an empty dictionary to store the transition matrices
    transition_matrices = {}
    
    dictionary_mapping_dictionary = get_group_to_matrix_mapping_dictionary(grouped_dictionary, matrix_files_list, ppid_column, session_column, trial_column)
    instructed_mapping_dictionary = get_group_to_matrix_mapping_dictionary(grouped_instructed, matrix_files_list, ppid_column, session_column, trial_column)
    
    # Handle correlation computation properly
    def safe_corrcoef(x):
        if x.shape[0] < 2:
            return np.nan
        stddev = np.std(x, axis=0)
        if np.any(stddev == 0):
            return np.nan
        return np.corrcoef(x)

    _, output_dir0 = compute_mean_transition_matrix(dictionary_mapping_dictionary, matrix_files, get_plots=True, sub_name="dictionary")
    _, output_dir0 = compute_mean_transition_matrix(instructed_mapping_dictionary, matrix_files, get_plots=True, sub_name="instructed")

    print('The code has finished running')
    print(f'data was saved in {output_dir0}')

if __name__ == "__main__":
    os.system("cls||clear")
    main()
