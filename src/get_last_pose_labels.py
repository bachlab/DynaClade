import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize
import re
from natsort import natsorted
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hdbscan
import umap
from scipy import stats
from matplotlib.colors import ListedColormap
from scipy.interpolate import splprep, splev
from adjustText import adjust_text
from scipy.spatial.distance import pdist
import matplotlib.gridspec as gridspec
import time
import pdb

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


# Function to process CSV files

def process_csv_files(labels_path, metadata_path, group_columns, groups_to_process, output_dir, tail_rows=1, save_all_poses=True):
    # Load the CSV files
    labels_df = pd.read_csv(labels_path)
    metadata_df = pd.read_csv(metadata_path)

    # Convert DataFrame column names to lowercase
    # labels_df.columns = labels_df.columns.str.lower()
    # metadata_df.columns = metadata_df.columns.str.lower()

    # Group by specified columns and get the last row of each group
    last_pose = labels_df.groupby(group_columns).tail(tail_rows)

    # Find common groups in both dataframes
    common_groups = set(labels_df.groupby(group_columns).groups.keys()) & set(metadata_df.groupby(group_columns).groups.keys())

    # Process the 'episode' column in the metadata
    metadata_df = process_episode_column(metadata_df)

    # Merge the last pose labels with the metadata
    merged_df = metadata_df.merge(last_pose, on=group_columns, how='inner')

    if save_all_poses:
        all_poses = labels_df.copy()
        all_poses_df = metadata_df.merge(all_poses, on=group_columns, how='inner')
        save_path = os.path.join(output_dir, 'all_poses.csv')
        all_poses_df.to_csv(save_path, index=False)

    # Convert DataFrame column names to lowercase
    # merged_df.columns = merged_df.columns.str.lower()

    # Check if the 'predicted_label' column exists in the merged DataFrame
    if 'predicted_label' in merged_df.columns:
        # Sort the 'predicted_label' column by frequency from highest to lowest
        label_counts = merged_df['predicted_label'].value_counts().sort_values(ascending=False)
        
        # Plot a bar plot of the 'predicted_label' column to show label frequency
        plt.figure(figsize=(10, 6))
        sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette='plasma', legend=False)
        plt.title('Frequency of Predicted Labels')
        plt.xlabel('Predicted Label')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
    else:
        print("The 'predicted_label' column does not exist in the merged DataFrame.")

    grouped_by_group = dict()
    for group in groups_to_process:
        grouped_by_group[group] = merged_df.groupby(group)
    

    return merged_df, grouped_by_group
    

def plot_label_frequency(label_counts, title, output_path, color_map):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette=color_map, legend=False)
    # plt.title(title, fontsize=20)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks([])  # Remove y-axis ticks
    plt.gca().spines['top'].set_visible(False)  # Remove top box line
    plt.gca().spines['right'].set_visible(False)  # Remove right box line
    plt.gca().spines['left'].set_visible(False)  # Remove left box line
    plt.gca().spines['bottom'].set_linewidth(1.5)  # Increase bottom box line width

    # Add value labels as percentage
    total = label_counts.sum()
    for p in ax.patches:
        percentage = (p.get_height() / total) * 100
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=14, color='black')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Function to perform hierarchical clustering, plot dendrogram, and save cluster labels

def plot_dendrogram_with_labels(data_matrix, labels, title, output_path, dataframe, label_column, num_clusters):
    # Debugging: Check the shape of data_matrix and dataframe before clustering
    print(f"Data matrix shape: {data_matrix.shape}")
    print(f"Dataframe shape: {dataframe.shape}")
    print(f"Data matrix columns: {data_matrix.columns}")
    print(f"Dataframe columns: {dataframe.columns}")
    print(f"Data matrix head: {data_matrix.head()}")
    print(f"Dataframe head: {dataframe.head()}")

    # import pdb; pdb.set_trace()

    plt.figure(figsize=(10, 7))
    linked = linkage(data_matrix, 'complete')
    # Ensure labels match the number of observations
    labels = labels.iloc[:data_matrix.shape[0]].tolist()
    dendrogram(linked,
               labels=labels,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Assign cluster labels
    cluster_labels = fcluster(linked, t=num_clusters, criterion='maxclust')
    dataframe[label_column] = cluster_labels

# Function to compute and plot feature importance

def compute_feature_importance(data_matrix, labels, title, output_path):
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(data_matrix, labels)

    # Get feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(data_matrix.shape[1]), importances[indices], align='center')
    plt.xticks(range(data_matrix.shape[1]), data_matrix.columns[indices], rotation=90)
    plt.xlim([-1, data_matrix.shape[1]])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Save the data with cluster labels
    data_with_labels = data_matrix.copy()
    data_with_labels['cluster'] = labels
    data_with_labels.to_csv(output_path.replace('.png', '_data_with_labels.csv'), index=False)

    # Compute and plot confusion matrix
    y_true = data_matrix.index  # Assuming the true labels are the index of the data_matrix
    y_pred = labels
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_confusion_matrix.png'))
    plt.close()

# Function to normalise data row-wise so each row sums to 1

def normalise_data(data_matrix, skip_columns=[]):
    # If the input is a Series, convert it to a DataFrame
    if isinstance(data_matrix, pd.Series):
        data_matrix = data_matrix.to_frame().T
    
    data_matrix_to_normalise = data_matrix.drop(columns=skip_columns)
    
    # Normalize row-wise
    normalised_data = pd.DataFrame(normalize(data_matrix_to_normalise, norm='l1', axis=1), columns=data_matrix_to_normalise.columns)
    
    normalised_data[skip_columns] = data_matrix[skip_columns]

    # Add a column with the number of observations for each row
    normalised_data['num_observations'] = data_matrix_to_normalise.sum(axis=1)

    return normalised_data

# Function to save normalised data with groups and clusters

def save_normalised_data_with_groups(normalised_data, group_column, cluster_column, output_path):
    normalised_data_with_groups = normalised_data.copy()
    normalised_data_with_groups[group_column] = group_column
    if cluster_column in normalised_data.columns:
        normalised_data_with_groups[cluster_column] = normalised_data[cluster_column]
    normalised_data_with_groups.to_csv(output_path, index=False)

# Function to perform exploratory data analysis

def perform_eda(data_matrix, title, output_dir):
    # Summary statistics
    summary_stats = data_matrix.describe()
    summary_stats.to_csv(os.path.join(output_dir, f'{title}_summary_statistics.csv'))

    # Correlation matrix
    correlation_matrix = data_matrix.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(f'{title} Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}_correlation_matrix.png'))
    plt.close()

    # Check if there are numeric columns before plotting histograms
    if not data_matrix.select_dtypes(include=[np.number]).empty:
        # Distribution plots
        data_matrix.hist(bins=20, figsize=(15, 10), layout=(5, 5))
        plt.suptitle(f'{title} Feature Distributions')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'{title}_feature_distributions.png'))
        plt.close()

    # Pair plots
    sns.pairplot(data_matrix)
    plt.suptitle(f'{title} Pair Plot', y=1.02)
    plt.savefig(os.path.join(output_dir, f'{title}_pair_plot.png'))
    plt.close()

# Function to process the 'episode' column based on 'episode_set_name'
def process_episode_column(metadata_df):
    # Define the new columns
    new_columns = ['scenario', 'episode_id', 'threat', 'visibility', 'distance', 'threat_action', 'weapons', 'safehouse']
    for col in new_columns:
        metadata_df[col] = None

    # Process each row based on 'episode_set_name'
    for index, row in metadata_df.iterrows():
        if '_' in row['episode'] and '-' in row['episode']:
            split_values = [s.strip() for s in re.split('[_-]', row['episode'])]
        elif '_' in row['episode']:
            split_values = [s.strip() for s in row['episode'].split('_')]
        elif '-' in row['episode']:
            split_values = [s.strip() for s in row['episode'].split('-')]
        else:
            split_values = [row['episode']]
        if row['episode_set_name'] == 'Dictionary of Actions Replication - Main':
            metadata_df.loc[index, new_columns] = split_values[:8]
        elif row['episode_set_name'] == 'Dictionary of Actions Replication - Fixed Trials':
            metadata_df.loc[index, ['scenario', 'episode_id', 'threat', 'threat_action']] = [split_values[0], split_values[3], split_values[4], split_values[5]]
        elif row['episode_set_name'] == 'Dictionary of Actions Replication - Stump Trials':
            metadata_df.loc[index, ['scenario', 'episode_id', 'threat']] = split_values[:3]

        metadata_df.loc[index, 'day_time'] = "night" if "night" in metadata_df.loc[index, 'scenario'].lower() else "day"
        metadata_df['threat'] = metadata_df['threat'].str.lower()
        metadata_df['threat_action'] = metadata_df['threat_action'].replace('chase', 'attack')
        metadata_df['episode_id'] = metadata_df['episode_id'].apply(lambda x: 'stump' if x is not None and ('stump' in str(x) or 'Stump' in str(x)) else x)

        # Check if the 'threat' column is 'none'
    return metadata_df


def analyze_feature_importance(data_matrix, labels, title, output_path):
    """Analyze feature importance using Random Forest and correlation analysis"""
    print("\nStarting feature importance analysis...")
    print("Initial DataFrame columns:", data_matrix.columns.tolist())
    print("Data types:", data_matrix.dtypes)
    
    # Create a copy of the data matrix
    df_encoded = data_matrix.copy()
    
    # Convert categorical features to numerical using label encoding
    for col in data_matrix.columns:
        if col in df_encoded.columns:
            print(f"\nProcessing column: {col}")
            print(f"Unique values before encoding: {df_encoded[col].unique()[:5]}...")
            df_encoded[col] = df_encoded[col].astype(str)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            print(f"Unique values after encoding: {df_encoded[col].unique()[:5]}...")
    
    # Prepare features and target
    X = df_encoded
    y = labels
    
    # Train Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': data_matrix.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Save feature importance to CSV
    feature_importance.to_csv(output_path.replace('.png', '.csv'), index=False)
    
    return feature_importance


def analyze_correlations(merged_df, output_dir, feature_columns):
    """Analyze correlations between features and predicted labels"""
    print("\nStarting correlation analysis...")
    print("Initial DataFrame columns:", merged_df.columns.tolist())
    print("Data types:", merged_df.dtypes)
    
    # Create a copy of the DataFrame
    df_encoded = merged_df.copy()
    
    # Convert categorical variables to numerical using label encoding
    
    # Label encode categorical columns
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    
    # First, ensure all categorical columns are strings
    for col in feature_columns:
        if col in df_encoded.columns:
            print(f"\nProcessing column: {col}")
            print(f"Unique values before encoding: {df_encoded[col].unique()[:5]}...")
            df_encoded[col] = df_encoded[col].astype(str)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            print(f"Unique values after encoding: {df_encoded[col].unique()[:5]}...")
    
    # Label encode the predicted_label column
    if 'predicted_label' in df_encoded.columns:
        print("\nProcessing predicted_label column")
        print(f"Unique values before encoding: {df_encoded['predicted_label'].unique()[:5]}...")
        le = LabelEncoder()
        df_encoded['predicted_label'] = le.fit_transform(df_encoded['predicted_label'].astype(str))
        print(f"Unique values after encoding: {df_encoded['predicted_label'].unique()[:5]}...")
    
    # Drop any remaining non-numeric columns
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    print("\nFinal numeric columns:", numeric_df.columns.tolist())
    
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Focus on correlations with predicted_label
    label_correlations = correlation_matrix['predicted_label'].sort_values(ascending=False)
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # Save correlations to CSV
    label_correlations.to_csv(os.path.join(output_dir, 'label_correlations.csv'))
    
    # Create a more readable version of the correlation matrix
    readable_correlations = pd.DataFrame({
        'Feature': label_correlations.index,
        'Correlation': label_correlations.values
    })
    readable_correlations.to_csv(os.path.join(output_dir, 'readable_correlations.csv'), index=False)
    
    return label_correlations


def analyze_feature_distributions(merged_df, output_dir, feature_columns):
    """Analyze distributions of features across different predicted labels"""
    # Find the label column
    label_column = None
    possible_label_columns = ['predicted_label', 'label', 'predicted', 'cluster']
    for col in possible_label_columns:
        if col in merged_df.columns:
            label_column = col
            break
    
    if label_column is None:
        print("Warning: Could not find label column. Available columns:", merged_df.columns.tolist())
        return
    
    
    for feature in feature_columns:
        if feature in merged_df.columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=merged_df, x=feature, hue=label_column)
            plt.title(f'Distribution of {feature} across {label_column}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))
            plt.close()


def perform_statistical_tests(merged_df, output_dir, feature_columns):
    """Perform statistical tests to analyze feature-label relationships"""
    from scipy.stats import chi2_contingency
    
    # Find the label column
    label_column = None
    possible_label_columns = ['predicted_label', 'label', 'predicted', 'cluster']
    for col in possible_label_columns:
        if col in merged_df.columns:
            label_column = col
            break
    
    if label_column is None:
        print("Warning: Could not find label column. Available columns:", merged_df.columns.tolist())
        return None
    
    
    results = []
    
    for feature in feature_columns:
        if feature in merged_df.columns:
            # Create contingency table
            contingency_table = pd.crosstab(merged_df[feature], merged_df[label_column])
            
            # Perform chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            results.append({
                'feature': feature,
                'chi2_statistic': chi2,
                'p_value': p,
                'degrees_of_freedom': dof
            })
    
    # Convert results to DataFrame and save
    stats_results = pd.DataFrame(results)
    stats_results.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
    
    return stats_results

     
def analyze_interactions_logistic_regression(merged_df, output_dir, feature_columns):
    """Analyze feature interactions using Logistic Regression"""
    print("\nStarting interaction analysis with Logistic Regression...")
    
    # Select features for interaction analysis (excluding high cardinality ones initially)
    label_column = 'predicted_label'
    
    if label_column not in merged_df.columns:
        print(f"Warning: Label column '{label_column}' not found. Skipping interaction analysis.")
        return None

    # Keep only relevant columns and drop rows with missing values in these columns
    relevant_cols = feature_columns + [label_column]
    df_analysis = merged_df[relevant_cols].copy().dropna()

    if df_analysis.empty:
        print("Warning: DataFrame is empty after selecting columns and dropping NA. Skipping interaction analysis.")
        return None

    print(f"DataFrame shape for analysis: {df_analysis.shape}")

    # Encode features and label
    df_encoded = df_analysis.copy()
    label_encoders = {}
    
    for col in feature_columns:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            
    # Encode the label column
    df_encoded[label_column] = df_encoded[label_column].astype(str)
    le_label = LabelEncoder()
    df_encoded[label_column] = le_label.fit_transform(df_encoded[label_column])
    label_encoders[label_column] = le_label
    
    X = df_encoded[feature_columns]
    y = df_encoded[label_column]

    if X.empty or len(y) == 0:
        print("Warning: No data available for X or y after encoding. Skipping interaction analysis.")
        return None
        
    # Split data (optional but recommended)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError as e:
        print(f"Warning: Could not stratify split, likely due to insufficient samples for some classes. Performing non-stratified split. Error: {e}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    if X_train.empty or X_test.empty:
         print("Warning: Train or test split resulted in empty data. Skipping interaction analysis.")
         return None

    # Create polynomial features (interactions)
    # interaction_only=True means only interaction terms (a*b), not polynomial terms (a^2)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Get feature names for the polynomial features
    poly_feature_names = poly.get_feature_names_out(input_features=feature_columns)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    # Train Logistic Regression model
    # Increased max_iter for convergence, multi_class='auto' handles binary/multiclass
    log_reg = LogisticRegression(random_state=42, max_iter=2000, multi_class='auto', solver='lbfgs')
    try:
        log_reg.fit(X_train_scaled, y_train)
    except ValueError as e:
        print(f"Error fitting Logistic Regression: {e}. Skipping interaction analysis.")
        print(f"Number of unique classes in y_train: {len(np.unique(y_train))}")
        if len(np.unique(y_train)) < 2:
             print("Need at least 2 classes to train Logistic Regression.")
        return None

    # Evaluate model (optional)
    y_pred = log_reg.predict(X_test_scaled)
    print("\nLogistic Regression Classification Report (on test set):")
    try:
        # ---> FIX 1: Specify labels present in the results and corresponding names <---
        present_labels = np.unique(np.concatenate((y_test, y_pred)))
        present_target_names = le_label.inverse_transform(present_labels)
        report = classification_report(y_test, y_pred, labels=present_labels, target_names=present_target_names, zero_division=0)
        print(report)
        with open(os.path.join(output_dir, 'logistic_regression_report.txt'), 'w') as f:
            f.write(report)
            
        # ---> ADDED: Plot Confusion Matrix <---
        cm = confusion_matrix(y_test, y_pred, labels=present_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=present_target_names, yticklabels=present_target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Logistic Regression Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'logistic_regression_confusion_matrix.png'))
        plt.close()
            
    except ValueError as e:
        print(f"Could not generate classification report or confusion matrix: {e}")


    # Extract coefficients
    coefficients = log_reg.coef_
    
    # Create DataFrame for coefficients
    if coefficients.shape[0] == 1: # Binary classification
        coef_df = pd.DataFrame({
            'feature': poly_feature_names,
            'coefficient': coefficients[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        class_names = ['coefficient'] # Simplified for binary
    else: # Multiclass classification
        class_names = le_label.classes_
        coef_dict = {'feature': poly_feature_names}
        for i, class_name in enumerate(class_names):
            coef_dict[f'coefficient_{class_name}'] = coefficients[i]
        coef_df = pd.DataFrame(coef_dict)
        # Optionally sort by the absolute sum or max of coefficients across classes
        coef_df['abs_coef_sum'] = coef_df.iloc[:, 1:].abs().sum(axis=1)
        coef_df = coef_df.sort_values('abs_coef_sum', ascending=False).drop('abs_coef_sum', axis=1)

    # Separate main effects and interaction effects
    # ---> FIX 2: Convert list to Series/array before using ~ <---
    interaction_terms_mask_list = [' ' in name for name in coef_df['feature']]
    interaction_terms_mask = pd.Series(interaction_terms_mask_list, index=coef_df.index)
    interaction_coefs = coef_df[interaction_terms_mask]
    main_effect_coefs = coef_df[~interaction_terms_mask]

    print("\nTop 10 Interaction Effects (Absolute Magnitude):")
    print(interaction_coefs.head(10))
    
    print("\nTop 10 Main Effects (Absolute Magnitude):")
    print(main_effect_coefs.head(10))

    # Save coefficients
    interaction_coefs.to_csv(os.path.join(output_dir, 'logistic_regression_interaction_coefficients.csv'), index=False)
    main_effect_coefs.to_csv(os.path.join(output_dir, 'logistic_regression_main_coefficients.csv'), index=False)
    coef_df.to_csv(os.path.join(output_dir, 'logistic_regression_all_coefficients.csv'), index=False)
    
    # ---> ADDED: Plot Top Coefficients <---
    n_top_features = 20 # Number of top features/interactions to plot
    
    # Plot Main Effects Coefficients
    if not main_effect_coefs.empty:
        plt.figure(figsize=(10, 8))
        # For multiclass, plot based on the first class coefficient or abs_coef_sum if available
        if 'abs_coef_sum' in coef_df.columns: # If we calculated it earlier for sorting
             # Re-calculate abs_coef_sum for main_effect_coefs specifically for plotting magnitude
             main_effect_coefs = main_effect_coefs.copy()  # Create a copy to avoid SettingWithCopyWarning
             main_effect_coefs.loc[:, 'plot_magnitude'] = main_effect_coefs.iloc[:, 1:].abs().sum(axis=1)
             sort_col = 'plot_magnitude'
        else:
             # For binary or if abs_coef_sum wasn't used, use the absolute value of the first coefficient column
             if len(class_names) > 0:
                 # For multiclass, use the first class's coefficient column
                 sort_col = f'coefficient_{class_names[0]}'
             else:
                 # For binary classification
                 sort_col = 'coefficient'
             main_effect_coefs = main_effect_coefs.copy()  # Create a copy to avoid SettingWithCopyWarning
             main_effect_coefs.loc[:, 'plot_magnitude'] = main_effect_coefs[sort_col].abs()
             sort_col = 'plot_magnitude'

        top_main_effects = main_effect_coefs.nlargest(n_top_features, sort_col)
        sns.barplot(x=sort_col, y='feature', data=top_main_effects, hue='feature', palette='plasma', legend=False)
        plt.title(f'Top {n_top_features} Logistic Regression Main Effect Coefficients (Magnitude)')
        plt.xlabel('Coefficient Magnitude (Sum across classes if multiclass)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'logistic_regression_main_coefficients_plot.png'))
        plt.close()
        # Clean up added column if it exists
        if 'plot_magnitude' in main_effect_coefs.columns:
             main_effect_coefs = main_effect_coefs.drop(columns=['plot_magnitude'])

    # Plot Interaction Effects Coefficients
    if not interaction_coefs.empty:
        plt.figure(figsize=(10, 8))
        # Similarly sort interaction terms
        if 'abs_coef_sum' in coef_df.columns:
            interaction_coefs = interaction_coefs.copy()  # Create a copy to avoid SettingWithCopyWarning
            interaction_coefs.loc[:, 'plot_magnitude'] = interaction_coefs.iloc[:, 1:].abs().sum(axis=1)
            sort_col = 'plot_magnitude'
        else:
            if len(class_names) > 0:
                # For multiclass, use the first class's coefficient column
                sort_col = f'coefficient_{class_names[0]}'
            else:
                # For binary classification
                sort_col = 'coefficient'
            interaction_coefs = interaction_coefs.copy()  # Create a copy to avoid SettingWithCopyWarning
            interaction_coefs.loc[:, 'plot_magnitude'] = interaction_coefs[sort_col].abs()
            sort_col = 'plot_magnitude'
            
        top_interaction_effects = interaction_coefs.nlargest(n_top_features, sort_col)
        sns.barplot(x=sort_col, y='feature', data=top_interaction_effects, hue='feature', palette='plasma', legend=False)
        plt.title(f'Top {n_top_features} Logistic Regression Interaction Effect Coefficients (Magnitude)')
        plt.xlabel('Coefficient Magnitude (Sum across classes if multiclass)')
        plt.ylabel('Interaction Term')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'logistic_regression_interaction_coefficients_plot.png'))
        plt.close()
        # Clean up added column if it exists
        if 'plot_magnitude' in interaction_coefs.columns:
            interaction_coefs = interaction_coefs.drop(columns=['plot_magnitude'])
            
    return coef_df


def plot_normalised_data_subplots(normalised_data, output_dir, color_map):
    """Plot normalized data with subplots for each row, using the key column value as a reference.
    
    Args:
        normalised_data (dict): Dictionary containing DataFrames with normalized data for each key
        output_dir (str): Directory to save the plots
        color_map (dict): Color map for the labels
    """
    normalised_figures_dir = os.path.join(output_dir, 'syllable_counts_all_figures', 'normalised_figures')
    for key, df in normalised_data.items():
        # Create a directory for this key
        key_dir = normalised_figures_dir
        os.makedirs(key_dir, exist_ok=True)

        # Get the number of rows in the DataFrame
        num_rows = len(df)


        # Calculate the number of rows and columns for the subplot grid
        num_cols = int(np.ceil(np.sqrt(num_rows)))
        num_plot_rows = int(np.ceil(num_rows / num_cols))
        
        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=num_plot_rows, ncols=num_cols, 
                                figsize=(5 * num_cols, 5 * num_plot_rows), sharex=True)
        axes = axes.flatten()  # Flatten the axes array for easy iteration
        
        # Get all unique labels across all rows to ensure consistent x-axis
        all_labels = set()
        for _, row in df.iterrows():
            # Remove the 'key' and 'cluster' columns if they are present in the DataFrame
            columns_to_remove = [col for col in [key, 'cluster', 'num_observations'] if col in row.index]
            if columns_to_remove:
                row = row.drop(columns_to_remove)
            all_labels.update(row.index)
        all_labels = sorted(list(all_labels))
        
        # Plot each row in a subplot
        for ax, (_, row) in zip(axes, df.iterrows()):
            # Get the key value for this row
            key_value = row[key]

            # Remove the 'key' and 'cluster' columns if they are present in the DataFrame
            columns_to_remove = [col for col in [key, 'cluster', 'num_observations'] if col in row.index]

            num_observations = row['num_observations']

            if columns_to_remove:
                row = row.drop(columns_to_remove)
            
            # Create a bar plot
            sns.barplot(x=all_labels, y=[row.get(label, 0) for label in all_labels], 
                       hue=all_labels, palette=color_map, ax=ax, legend=False)
            
            # Set title and labels
            ax.set_title(f'{key} = {key_value} | (N={int(num_observations)})')
            ax.set_xlabel('Label')
            ax.set_ylabel('Normalised Frequency')
            ax.set_ylim(0, 1)

            # Set tick locations explicitly before setting labels
            ax.set_xticks(range(len(all_labels)))
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(all_labels, rotation=45, ha='right')
            
            # Add grid
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Hide any unused subplots
        for ax in axes[num_rows:]:
            ax.set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(key_dir, f'normalised_data_{key}_subplots.png'))
        plt.close()


def analyze_combined_effects(merged_df, output_dir, feature_columns):
    """Analyze combined effects of experimental variables on predicted labels"""
    print("\nStarting combined effects analysis...")
    
    # Create analysis directory
    combined_effects_dir = os.path.join(output_dir, 'combined_effects')
    os.makedirs(combined_effects_dir, exist_ok=True)
    
    # Prepare data
    df_encoded = merged_df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in feature_columns:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    
    # Encode the target variable
    le_label = LabelEncoder()
    df_encoded['predicted_label'] = le_label.fit_transform(df_encoded['predicted_label'].astype(str))
    
    # Split data
    X = df_encoded[feature_columns]
    y = df_encoded['predicted_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Logistic Regression with Interaction Terms
    print("\n1. Logistic Regression with Interaction Terms")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Get feature names for interactions
    poly_feature_names = poly.get_feature_names_out(input_features=feature_columns)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    # Fit logistic regression
    log_reg = LogisticRegression(random_state=42, max_iter=2000, multi_class='auto', solver='lbfgs')
    log_reg.fit(X_train_scaled, y_train)
    
    # Get coefficients
    coef_df = pd.DataFrame({
        'feature': poly_feature_names,
        'coefficient': log_reg.coef_[0]  # For binary classification, use [0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    # Save coefficients
    coef_df.to_csv(os.path.join(combined_effects_dir, 'logistic_regression_interaction_coefficients.csv'), index=False)
    
    # 2. Random Forest with Interaction Features
    print("\n2. Random Forest with Interaction Features")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_poly, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': poly_feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(combined_effects_dir, 'random_forest_interaction_importance.csv'), index=False)
    
    # Plot top interactions
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Interactions (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(combined_effects_dir, 'top_interactions.png'))
    plt.close()
    
    # 3. ANOVA Analysis
    print("\n3. ANOVA Analysis")
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # Create formula for ANOVA
    formula = 'predicted_label ~ ' + ' + '.join(feature_columns) + ' + ' + ' + '.join([f'{col1}:{col2}' for i, col1 in enumerate(feature_columns) for col2 in feature_columns[i+1:]])
    
    # Fit ANOVA model
    model = ols(formula, data=df_encoded).fit()
    anova_results = anova_lm(model)
    
    # Save ANOVA results
    anova_results.to_csv(os.path.join(combined_effects_dir, 'anova_results.csv'))
    
    # Plot ANOVA results
    plt.figure(figsize=(12, 8))
    sns.barplot(x=anova_results.index, y='PR(>F)', data=anova_results)
    plt.title('ANOVA Results')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(combined_effects_dir, 'anova_results.png'))
    plt.close()
    
    return coef_df, feature_importance, anova_results


def plot_umap_scatter(umap_result, df, var, output_dir, cluster_labels=None, cluster_method=None):
    """Create detailed 2D and 3D scatter plots of UMAP-transformed data"""
    print(f"\nCreating UMAP scatter plots for {var}...")
    
    # Create a directory for UMAP visualizations
    umap_dir = os.path.join(output_dir, 'umap_visualizations')
    os.makedirs(umap_dir, exist_ok=True)
    
    # Create a DataFrame with the UMAP results
    umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2', 'UMAP3'])
    if var in df.columns:
        umap_df[var] = df[var]
    if cluster_labels is not None:
        umap_df['cluster'] = cluster_labels
    
    # Add method information to filenames
    method_suffix = f"_umap_{cluster_method}" if cluster_method else "_umap"
    
    # 1. 2D Scatter Plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Colored by variable
    if var in df.columns:
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                            c=df[var].astype('category').cat.codes,
                            cmap='plasma', alpha=0.6)
        plt.colorbar(scatter, label=var)
        plt.title(f'UMAP 2D Projection (Colored by {var}){f" - {cluster_method.capitalize()} Clustering" if cluster_method else ""}')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
    
    # Plot 2: Colored by cluster if available
    if cluster_labels is not None:
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                            c=cluster_labels, cmap='plasma', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'UMAP 2D Projection (Colored by Cluster){f" - {cluster_method.capitalize()} Clustering" if cluster_method else ""}')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
    
    # Plot 3: Colored by UMAP3 (depth)
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                        c=umap_result[:, 2], cmap='plasma', alpha=0.6)
    plt.colorbar(scatter, label='UMAP3')
    plt.title(f'UMAP 2D Projection (Colored by UMAP3){f" - {cluster_method.capitalize()} Clustering" if cluster_method else ""}')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    
    # Plot 4: Add convex hulls around clusters if available
    if cluster_labels is not None:
        plt.subplot(2, 2, 4)
        for cluster in np.unique(cluster_labels):
            if cluster != -1:  # Skip noise points
                cluster_points = umap_result[cluster_labels == cluster]
                if len(cluster_points) >= 3:  # Need at least 3 points for convex hull
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(cluster_points[:, :2])
                    plt.fill(cluster_points[hull.vertices, 0], 
                           cluster_points[hull.vertices, 1], 
                           alpha=0.1)
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              label=f'Cluster {cluster}', alpha=0.6)
        plt.title(f'UMAP 2D Projection with Cluster Boundaries{f" - {cluster_method.capitalize()} Clustering" if cluster_method else ""}')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(umap_dir, f'{var}{method_suffix}_2d_scatter.png'))
    plt.close()
    
    # 2. 3D Scatter Plots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Colored by cluster if available
    if 'cluster' in df.columns:
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2],
                           c=df['cluster'], cmap='plasma', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        ax.set_title('3D Scatter Plot (Colored by Cluster)')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
    
    # Plot 2: Colored by variable
    if var in df.columns:
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2],
                           c=df[var].astype('category').cat.codes, 
                           cmap='plasma', alpha=0.6)
        plt.colorbar(scatter, label=var)
        ax.set_title(f'3D Scatter Plot (Colored by {var})')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
    
    # Plot 3: Colored by UMAP1
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2],
                       c=umap_result[:, 0], cmap='plasma', alpha=0.6)
    plt.colorbar(scatter, label='UMAP1')
    ax.set_title('3D Scatter Plot (Colored by UMAP1)')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    
    # Plot 4: Colored by UMAP2
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2],
                       c=umap_result[:, 1], cmap='plasma', alpha=0.6)
    plt.colorbar(scatter, label='UMAP2')
    ax.set_title('3D Scatter Plot (Colored by UMAP2)')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    
    plt.tight_layout()
    plt.savefig(os.path.join(umap_dir, f'{var}{method_suffix}_3d_scatter.png'))
    plt.close()
    
    # Save the UMAP results
    umap_df.to_csv(os.path.join(umap_dir, f'{var}{method_suffix}_results.csv'), index=False)


def analyze_clustering_patterns(normalised_data, output_dir, variables_of_interest, color_map):
    """Analyze clustering patterns in normalized data for specific variables"""
    print("\nAnalyzing clustering patterns...")
    
    # Create analysis directory
    clustering_dir = os.path.join(output_dir, 'clustering_patterns')
    os.makedirs(clustering_dir, exist_ok=True)
    
    # Define pastel color palette and convert to matplotlib colormap
    pastel_palette = sns.color_palette("pastel")
    pastel_cmap = ListedColormap(pastel_palette)
    boundary_alpha = 0.5

    normalised_hdbscan_df = dict()
    for var in variables_of_interest:
        if var in normalised_data:
            print(f"\nAnalyzing clustering patterns for {var}...")
            
            # Get the normalized data for this variable
            df = normalised_data[var].copy()
            
            # Drop non-feature columns
            columns_to_drop = [col for col in [var, 'cluster', 'num_observations'] if col in df.columns]
            df_features = df.drop(columns=columns_to_drop)
            
            # Check if we have enough data points for clustering
            if len(df_features) < 5:  # Need at least 5 points for meaningful clustering
                print(f"Not enough data points for {var} (N={len(df_features)}). Skipping clustering.")
                continue
            
            # Apply UMAP for dimensionality reduction and structure preservation
            print("\nApplying UMAP for dimensionality reduction...")
            umap_reducer = umap.UMAP(
                n_neighbors=min(15, len(df_features) - 1),  # Adjust based on data size
                min_dist=0.1,
                n_components=3,  # Reduce to 3D for visualization
                metric='euclidean',
                random_state=42
            )
            
            # Fit and transform the data
            umap_result = umap_reducer.fit_transform(df_features)
            
            # Create initial scatter plots of UMAP results (before clustering)
            plot_umap_scatter(umap_result, df, var, clustering_dir)
            
            # 1. Hierarchical Clustering on UMAP-transformed data
            print("\nPerforming hierarchical clustering on UMAP-transformed data...")
            linked = linkage(umap_result, 'complete')
            
            # Try different numbers of clusters and evaluate
            n_clusters_range = range(2, min(6, len(umap_result)))  # Ensure n_clusters < n_samples
            best_n_clusters = None
            best_silhouette_score = -1
            
            for n_clusters in n_clusters_range:
                cluster_labels = fcluster(linked, t=n_clusters, criterion='maxclust')
                unique_labels = np.unique(cluster_labels)
                
                # Only calculate silhouette score if we have at least 2 clusters
                if len(unique_labels) > 1:
                    try:
                        silhouette_avg = silhouette_score(umap_result, cluster_labels)
                        if silhouette_avg > best_silhouette_score:
                            best_silhouette_score = silhouette_avg
                            best_n_clusters = n_clusters
                    except ValueError as e:
                        print(f"Warning: Could not calculate silhouette score for {n_clusters} clusters: {e}")
                        continue
            
            # If no valid clustering found, use 2 clusters as default
            if best_n_clusters is None:
                print("No valid clustering found with silhouette score. Using 2 clusters as default.")
                best_n_clusters = 2
                best_silhouette_score = 0  # Set to 0 as we couldn't calculate it
            
            # Get final cluster labels with best number of clusters
            hierarchical_labels = fcluster(linked, t=best_n_clusters, criterion='maxclust')
            
            # Plot hierarchical clustering dendrogram
            plt.figure(figsize=(15, 7))
            dendrogram(linked,
                      labels=df[var].values,
                      orientation='top',
                      distance_sort='descending',
                      show_leaf_counts=True)
            plt.title(f'Hierarchical Clustering Dendrogram (UMAP-transformed) for {var} (N={len(df)})')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(clustering_dir, f'{var}_umap_hierarchical_dendrogram.png'))
            plt.close()
            
            # 2. HDBSCAN Clustering on UMAP-transformed data
            print("\nPerforming HDBSCAN clustering on UMAP-transformed data...")
            # Set min_cluster_size to 2
            min_cluster_size = 2
            hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                metric='euclidean'
            )
            hdbscan_labels = hdbscan_clusterer.fit_predict(umap_result)
            
            # Calculate silhouette score for HDBSCAN if we have at least 2 clusters
            hdbscan_silhouette = -1  # Default value if we can't calculate
            unique_hdbscan_labels = np.unique(hdbscan_labels)
            if len(unique_hdbscan_labels) > 1:
                try:
                    hdbscan_silhouette = silhouette_score(umap_result, hdbscan_labels)
                except ValueError as e:
                    print(f"Warning: Could not calculate HDBSCAN silhouette score: {e}")
            
            # Create separate directories for each clustering method
            hdbscan_dir = os.path.join(clustering_dir, 'hdbscan_results')
            hierarchical_dir = os.path.join(clustering_dir, 'hierarchical_results')
            os.makedirs(hdbscan_dir, exist_ok=True)
            os.makedirs(hierarchical_dir, exist_ok=True)
            
            # Save HDBSCAN results and visualizations
            hdbscan_df = pd.DataFrame({
                'variable': df[var],
                'cluster': hdbscan_labels,
                'UMAP1': umap_result[:, 0],
                'UMAP2': umap_result[:, 1],
                'UMAP3': umap_result[:, 2]
            })
            hdbscan_df.to_csv(os.path.join(hdbscan_dir, f'{var}_hdbscan_results.csv'), index=False)
            
            # Save the normalised data with the cluster labels
            normalised_hdbscan_df[var] = normalised_data[var].copy()
            normalised_hdbscan_df[var]['cluster'] = hdbscan_labels
            normalised_hdbscan_df[var].to_csv(os.path.join(hdbscan_dir, f'{var}_normalised_hdbscan_results.csv'), index=False)
            
            # Save hierarchical clustering results and visualizations
            hierarchical_df = pd.DataFrame({
                'variable': df[var],
                'cluster': hierarchical_labels,
                'UMAP1': umap_result[:, 0],
                'UMAP2': umap_result[:, 1],
                'UMAP3': umap_result[:, 2]
            })
            hierarchical_df.to_csv(os.path.join(hierarchical_dir, f'{var}_hierarchical_results.csv'), index=False)
            
            # Plot UMAP scatter plots with both clustering methods
            for method, labels, directory in [
                ("HDBSCAN", hdbscan_labels, hdbscan_dir),
                ("Hierarchical", hierarchical_labels, hierarchical_dir)
            ]:
                # Plot 1: 2D scatter plot with cluster boundaries
                plt.figure(figsize=(15, 10))
                
                # First plot the cluster boundaries
                unique_clusters = np.unique(labels)
                unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise
                cluster_colors = {cluster: pastel_palette[i % len(pastel_palette)] for i, cluster in enumerate(unique_clusters)}
                
                for cluster in unique_clusters:
                    cluster_mask = labels == cluster
                    cluster_points = umap_result[cluster_mask]
                    
                    if len(cluster_points) >= 3:  # For 3 or more points, use convex hull
                        try:
                            # Calculate the centroid of the cluster points
                            centroid = np.mean(cluster_points[:, :2], axis=0)
                            
                            # Get points for the convex hull
                            from scipy.spatial import ConvexHull
                            hull = ConvexHull(cluster_points[:, :2])
                            hull_points = cluster_points[hull.vertices]
                            
                            # Add padding by moving points outward from centroid
                            padding = 1.1  # 10% padding
                            hull_points_padded = np.zeros_like(hull_points[:, :2])
                            for i, point in enumerate(hull_points[:, :2]):
                                # Vector from centroid to point
                                vector = point - centroid
                                # Scale the vector to add padding
                                padded_point = centroid + vector * padding
                                hull_points_padded[i] = padded_point
                            
                            if len(hull_points) >= 4:
                                # Use periodic spline for smooth boundary
                                hull_points_closed = np.vstack([hull_points_padded, hull_points_padded[0]])
                                tck, u = splprep([hull_points_closed[:, 0], hull_points_closed[:, 1]], s=0, k=3, per=True)
                                u_new = np.linspace(0, 1, 100)
                                smooth_boundary = np.array(splev(u_new, tck)).T
                                plt.fill(smooth_boundary[:, 0], smooth_boundary[:, 1], 
                                       alpha=0.3, color=cluster_colors[cluster])
                            else:
                                # For clusters with 3 points, use the padded hull directly
                                hull_points_closed = np.vstack([hull_points_padded, hull_points_padded[0]])
                                plt.fill(hull_points_closed[:, 0], hull_points_closed[:, 1], 
                                       alpha=0.3, color=cluster_colors[cluster])
                        except Exception as e:
                            print(f"Warning: Could not create boundary for cluster {cluster}: {e}")
                            # Fallback to simple padded boundary
                            centroid = np.mean(cluster_points[:, :2], axis=0)
                            padded_points = np.zeros_like(cluster_points[:, :2])
                            for i, point in enumerate(cluster_points[:, :2]):
                                vector = point - centroid
                                padded_points[i] = centroid + vector * 1.1
                            plt.fill(padded_points[:, 0], padded_points[:, 1],
                                   alpha=0.3, color=cluster_colors[cluster])
                    
                    elif len(cluster_points) == 2:  # Special handling for 2-point clusters
                        # Calculate the centroid
                        centroid = np.mean(cluster_points[:, :2], axis=0)
                        
                        # Get the vector between the two points
                        vector = cluster_points[1, :2] - cluster_points[0, :2]
                        perpendicular = np.array([-vector[1], vector[0]])  # Perpendicular vector
                        
                        # Normalize perpendicular vector and scale it
                        perp_length = np.linalg.norm(perpendicular)
                        if perp_length > 0:
                            perpendicular = perpendicular / perp_length * np.linalg.norm(vector) * 0.5
                        
                        # Create a diamond-like shape around the two points
                        padding = 1.1  # 10% padding
                        points = np.array([
                            cluster_points[0, :2] - perpendicular * padding,  # Left point
                            cluster_points[1, :2] - perpendicular * padding,  # Right point
                            cluster_points[1, :2] + perpendicular * padding,  # Top point
                            cluster_points[0, :2] + perpendicular * padding,  # Bottom point
                        ])
                        
                        # Close the shape
                        points = np.vstack([points, points[0]])
                        
                        # Create a smooth boundary
                        tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=3, per=True)
                        u_new = np.linspace(0, 1, 100)
                        smooth_boundary = np.array(splev(u_new, tck)).T
                        
                        plt.fill(smooth_boundary[:, 0], smooth_boundary[:, 1],
                               alpha=0.3, color=cluster_colors[cluster])
                
                # Then plot the points with normal colors
                # First plot noise points in pastel grey
                noise_mask = labels == -1
                if np.any(noise_mask):
                    plt.scatter(umap_result[noise_mask, 0], umap_result[noise_mask, 1],
                              c='#D3D3D3', alpha=0.8, s=200, label='Noise')
                
                # Then plot clustered points with their respective colors
                for cluster in unique_clusters:
                    cluster_mask = labels == cluster
                    plt.scatter(umap_result[cluster_mask, 0], umap_result[cluster_mask, 1],
                              c=[cluster_colors[cluster]], alpha=0.8, s=200, label=f'Cluster {cluster}')
                
                # Add labels to points with improved positioning
                from adjustText import adjust_text
                texts = []
                for i, label in enumerate(df[var]):
                    # Add labels for all points, including noise
                    texts.append(plt.text(umap_result[i, 0], umap_result[i, 1],
                                        label, fontsize=25, alpha=0.8))
                
                # Adjust text positions to prevent overlap
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
                
                # Add legend
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Get condition information from the variable name
                condition_info = f"Condition: {var}"
                plt.title(f'UMAP 2D Projection (Colored by Cluster) - {method}\n{condition_info}')
                plt.xlabel('UMAP1', fontsize=22)
                plt.ylabel('UMAP2', fontsize=22)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(directory, f'{var}_{method.lower()}_2d_scatter.png'))
                plt.close()
                
                # Plot 2: 3D scatter plot with cluster boundaries
                fig = plt.figure(figsize=(15, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # First plot the cluster boundaries
                for cluster in np.unique(labels):
                    if cluster != -1:  # Skip noise points
                        cluster_points = umap_result[labels == cluster]
                        if len(cluster_points) >= 5:  # Need at least 5 points for 3D Delaunay triangulation
                            try:
                                from scipy.spatial import Delaunay
                                tri = Delaunay(cluster_points)
                                for simplex in tri.simplices:
                                    # Create a smooth surface using a mesh
                                    ax.plot_trisurf(cluster_points[simplex, 0], 
                                                  cluster_points[simplex, 1], 
                                                  cluster_points[simplex, 2],
                                                  alpha=0.5, 
                                                  color=pastel_palette[cluster % len(pastel_palette)])
                            except Exception as e:
                                print(f"Warning: Could not create 3D boundary for cluster {cluster}: {e}")
                                # Fallback to simple scatter plot for this cluster
                                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                                         c=[pastel_palette[cluster % len(pastel_palette)]],
                                         alpha=0.5, s=100)
                        else:
                            # For small clusters, just plot the points
                            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                                     c=[pastel_palette[cluster % len(pastel_palette)]],
                                     alpha=0.5, s=100)
                
                # Then plot the points with normal colors
                # First plot noise points in pastel grey
                if np.any(noise_mask):
                    ax.scatter(umap_result[noise_mask, 0], umap_result[noise_mask, 1], umap_result[noise_mask, 2],
                             c='#D3D3D3', alpha=0.8, s=200, label='Noise')
                
                # Then plot clustered points with normal colors
                if np.any(cluster_mask):
                    scatter = ax.scatter(umap_result[cluster_mask, 0], umap_result[cluster_mask, 1], umap_result[cluster_mask, 2],
                                       c=labels[cluster_mask], cmap='plasma', alpha=0.8, s=200)
                    plt.colorbar(scatter, label='Cluster')
                
                # Add labels to 3D points with improved positioning
                for i, label in enumerate(df[var]):
                    # Add labels for all points, including noise
                    # Calculate offset based on point position to prevent overlap
                    offset = 0.02 * (i % 3 - 1)  # Vary offset based on index
                    ax.text(umap_result[i, 0] + offset, 
                           umap_result[i, 1] + offset, 
                           umap_result[i, 2] + offset,
                           label, fontsize=12, alpha=0.8)
                
                ax.set_title(f'3D UMAP Projection (Colored by Cluster) - {method}\n{condition_info}')
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_zlabel('UMAP3')
                
                plt.tight_layout()
                plt.savefig(os.path.join(directory, f'{var}_{method.lower()}_3d_scatter.png'))
                plt.close()
            
            # Save HDBSCAN results as the main results
            umap_df = pd.DataFrame({
                'variable': df[var],
                'cluster': hdbscan_labels,
                'UMAP1': umap_result[:, 0],
                'UMAP2': umap_result[:, 1],
                'UMAP3': umap_result[:, 2]
            })
            umap_df.to_csv(os.path.join(clustering_dir, f'{var}_umap_data.csv'), index=False)
            
            # Save clustering results
            clustering_results = pd.DataFrame({
                'variable': df[var],
                'cluster': hdbscan_labels,
                'UMAP1': umap_result[:, 0],
                'UMAP2': umap_result[:, 1],
                'UMAP3': umap_result[:, 2]
            })
            clustering_results.to_csv(os.path.join(clustering_dir, f'{var}_clustering_results.csv'), index=False)
            
            # Print clustering information
            print(f"\nClustering results for {var}:")
            print(f"Main clustering method: HDBSCAN")
            print(f"HDBSCAN silhouette score: {hdbscan_silhouette:.4f}")
            print(f"Hierarchical silhouette score: {best_silhouette_score:.4f}")
            print(f"Number of clusters (HDBSCAN): {len(np.unique(hdbscan_labels[hdbscan_labels != -1]))}")
            print(f"Number of clusters (Hierarchical): {best_n_clusters}")
            print(f"Results saved in: {clustering_dir}")
            print(f"HDBSCAN results saved in: {hdbscan_dir}")
            print(f"Hierarchical results saved in: {hierarchical_dir}")

    return normalised_hdbscan_df



def create_data_visualizations(normalised_data, output_dir, variables_of_interest, color_map):
    """Create comprehensive data visualizations for the normalized data"""
    print("\nCreating comprehensive data visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'data_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    for var in variables_of_interest:
        if var in normalised_data:
            print(f"\nCreating visualizations for {var}...")
            df = normalised_data[var].copy()
            
            # Drop non-feature columns
            columns_to_drop = [col for col in [var, 'cluster', 'num_observations'] if col in df.columns]
            df_features = df.drop(columns=columns_to_drop)
            
            # Check if we have enough data points for visualization
            if len(df_features) < 5:
                print(f"Not enough data points for {var} (N={len(df_features)}). Skipping visualizations.")
                continue
            
            # Check for NaN values and handle them
            if df_features.isna().any().any():
                print(f"Warning: NaN values found in {var}. Dropping rows with NaN values.")
                df_features = df_features.dropna()
                if len(df_features) < 5:
                    print(f"Not enough valid data points after removing NaN values. Skipping visualizations.")
                    continue
            
            # 1. Correlation Analysis
            print("Creating correlation visualizations...")
            plt.figure(figsize=(15, 12))
            correlation_matrix = df_features.corr()
            
            # Check if correlation matrix is valid
            if correlation_matrix.isna().all().all():
                print(f"Warning: All correlations are NaN for {var}. Skipping correlation visualizations.")
                continue
                
            # 1.1 Correlation Heatmap
            plt.subplot(2, 2, 1)
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title(f'Feature Correlation Matrix for {var}')
            
            # 1.2 Correlation Clustermap
            plt.subplot(2, 2, 2)
            sns.clustermap(correlation_matrix, cmap='coolwarm', center=0, figsize=(10, 10))
            plt.title(f'Clustered Correlation Matrix for {var}')
            
            # 1.3 Top Correlations Bar Plot
            plt.subplot(2, 2, 3)
            top_correlations = correlation_matrix.abs().unstack().sort_values(ascending=False)
            top_correlations = top_correlations[top_correlations < 1].head(20)
            
            # Check if we have valid correlations to plot
            if len(top_correlations) > 0:
                sns.barplot(x=top_correlations.values, y=[f"{pair[0]} vs {pair[1]}" for pair in top_correlations.index])
                plt.title('Top 20 Feature Correlations')
                plt.xticks(rotation=45)
            else:
                print(f"No valid correlations found for {var}. Skipping correlation bar plot.")
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{var}_correlation_analysis.png'))
            
            # 2. Distribution Analysis
            print("Creating distribution visualizations...")
            plt.figure(figsize=(15, 10))
            
            # 2.1 Feature Distributions with KDE
            for i, col in enumerate(df_features.columns, 1):
                plt.subplot(3, 3, i)
                sns.histplot(data=df_features, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                if i >= 9:  # Limit to 9 plots per figure
                    break
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{var}_feature_distributions.png'))
            plt.close()
            
            # 2.2 QQ Plots for Normality Check
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(df_features.columns, 1):
                plt.subplot(3, 3, i)
                stats.probplot(df_features[col], dist="norm", plot=plt)
                plt.title(f'QQ Plot for {col}')
                if i >= 9:
                    break
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{var}_qq_plots.png'))
            plt.close()
            
            # 3. Feature Relationships
            print("Creating feature relationship visualizations...")
            
            # 3.1 Pair Plot with Density
            plt.figure(figsize=(15, 10))
            top_features = list(correlation_matrix.abs().mean().sort_values(ascending=False).head(5).index)
            if len(top_features) > 1:
                sns.pairplot(df_features[top_features], 
                           diag_kind='kde',
                           plot_kws={'alpha': 0.6},
                           diag_kws={'fill': True})
                plt.suptitle(f'Pair Plot of Top 5 Features for {var}', y=1.02)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'{var}_pair_plot.png'))
                plt.close()
            
            # 3.2 Feature Importance Plot
            if var in df.columns:
                plt.figure(figsize=(15, 10))
                df_melted = pd.melt(df_features.join(df[var]), id_vars=[var])
                
                # Box Plot
                plt.subplot(2, 2, 1)
                sns.boxplot(x=var, y='value', hue='variable', data=df_melted)
                plt.title(f'Feature Distributions by {var}')
                plt.xticks(rotation=45)
                
                # Violin Plot
                plt.subplot(2, 2, 2)
                sns.violinplot(x=var, y='value', hue='variable', data=df_melted, split=True)
                plt.title(f'Feature Distributions by {var} (Violin Plots)')
                plt.xticks(rotation=45)
                
                # Swarm Plot
                plt.subplot(2, 2, 3)
                sns.stripplot(x=var, y='value', hue='variable', data=df_melted, 
                            jitter=True, alpha=0.6, size=4)
                plt.title(f'Feature Distributions by {var} (Strip Plot)')
                plt.xticks(rotation=45)
                
                # Strip Plot
                plt.subplot(2, 2, 4)
                sns.stripplot(x=var, y='value', hue='variable', data=df_melted, jitter=True)
                plt.title(f'Feature Distributions by {var} (Strip Plot)')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'{var}_feature_distributions_by_{var}.png'))
                plt.close()
            
            # 4. Advanced Analysis
            print("Creating advanced analysis visualizations...")
            
            # 4.1 Feature Importance using Random Forest
            if var in df.columns:
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(df_features, df[var])
                
                plt.figure(figsize=(12, 8))
                feature_importance = pd.DataFrame({
                    'feature': df_features.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
                plt.title(f'Top 20 Most Important Features for {var}')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'{var}_feature_importance.png'))
                plt.close()
            
            # 4.2 PCA Analysis
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_features)
            
            plt.figure(figsize=(12, 8))
            if var in df.columns:
                scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                    c=df[var].astype('category').cat.codes,
                                    cmap='plasma', alpha=0.6)
                plt.colorbar(scatter, label=var)
            else:
                plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
            
            plt.title(f'PCA Projection of Features for {var}')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{var}_pca_projection.png'))
            plt.close()
            
            # 4.3 Feature Clustering
            plt.figure(figsize=(12, 8))
            sns.clustermap(df_features.corr(), cmap='coolwarm', center=0)
            plt.title(f'Feature Clustering for {var}')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{var}_feature_clustering.png'))
            plt.close()
            
            # 5. Time Series Analysis (if applicable)
            if 'time' in df.columns or 'session' in df.columns:
                time_col = 'time' if 'time' in df.columns else 'session'
                plt.figure(figsize=(15, 10))
                for i, col in enumerate(df_features.columns, 1):
                    plt.subplot(3, 3, i)
                    sns.lineplot(x=df[time_col], y=df_features[col])
                    plt.title(f'{col} over {time_col}')
                    if i >= 9:
                        break
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'{var}_time_series.png'))
                plt.close()


def plot_observation_scatter(normalised_data, output_dir, variables_of_interest, color_map):
    """Create detailed scatter plots of observations to visualize cluster shapes and separations"""
    print("\nCreating observation scatter plots...")
    
    # Create scatter plot directory
    scatter_dir = os.path.join(output_dir, 'observation_scatter')
    os.makedirs(scatter_dir, exist_ok=True)
    
    for var in variables_of_interest:
        if var in normalised_data:
            print(f"\nCreating scatter plots for {var}...")
            df = normalised_data[var].copy()
            
            # Drop non-feature columns
            columns_to_drop = [col for col in [var, 'cluster', 'num_observations'] if col in df.columns]
            df_features = df.drop(columns=columns_to_drop)
            
            # Check if we have enough data points for visualization
            if len(df_features) < 5:
                print(f"Not enough data points for {var} (N={len(df_features)}). Skipping scatter plots.")
                continue
            
            # Apply PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df_features)
            
            # Create DataFrame with reduced dimensions
            reduced_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'PC3': pca_result[:, 2]
            })
            
            # Add original variable and cluster if available
            if var in df.columns:
                reduced_df[var] = df[var]
            if 'cluster' in df.columns:
                reduced_df['cluster'] = df['cluster']
            
            # 1. Create 2D scatter plots
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Colored by cluster if available
            if 'cluster' in reduced_df.columns:
                plt.subplot(2, 2, 1)
                scatter = plt.scatter(reduced_df['PC1'], reduced_df['PC2'], 
                                    c=reduced_df['cluster'], cmap='plasma', alpha=0.6)
                plt.colorbar(scatter, label='Cluster')
                plt.title('2D Scatter Plot (Colored by Cluster)')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            
            # Plot 2: Colored by variable
            if var in reduced_df.columns:
                plt.subplot(2, 2, 2)
                scatter = plt.scatter(reduced_df['PC1'], reduced_df['PC2'], 
                                    c=reduced_df[var].astype('category').cat.codes, 
                                    cmap='plasma', alpha=0.6)
                plt.colorbar(scatter, label=var)
                plt.title(f'2D Scatter Plot (Colored by {var})')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            
            # Plot 3: Colored by PC3 (depth)
            plt.subplot(2, 2, 3)
            scatter = plt.scatter(reduced_df['PC1'], reduced_df['PC2'], 
                                c=reduced_df['PC3'], cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, label='PC3')
            plt.title('2D Scatter Plot (Colored by PC3)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            
            # Plot 4: Add convex hulls around clusters if available
            if 'cluster' in reduced_df.columns:
                plt.subplot(2, 2, 4)
                for cluster in np.unique(reduced_df['cluster']):
                    if cluster != -1:  # Skip noise points
                        cluster_points = reduced_df[reduced_df['cluster'] == cluster]
                        if len(cluster_points) >= 3:  # Need at least 3 points for convex hull
                            from scipy.spatial import ConvexHull
                            hull = ConvexHull(cluster_points[['PC1', 'PC2']].values)
                            plt.fill(cluster_points.iloc[hull.vertices]['PC1'], 
                                   cluster_points.iloc[hull.vertices]['PC2'], 
                                   alpha=0.1)
                            plt.scatter(cluster_points['PC1'], cluster_points['PC2'], 
                                      label=f'Cluster {cluster}', alpha=0.6)
                plt.title('2D Scatter Plot with Cluster Boundaries')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(scatter_dir, f'{var}_2d_scatter.png'))
            plt.close()
            
            # 2. Create 3D scatter plots
            fig = plt.figure(figsize=(15, 10))
            
            # Plot 1: Colored by cluster if available
            if 'cluster' in reduced_df.columns:
                ax = fig.add_subplot(2, 2, 1, projection='3d')
                scatter = ax.scatter(reduced_df['PC1'], reduced_df['PC2'], reduced_df['PC3'],
                                   c=reduced_df['cluster'], cmap='plasma', alpha=0.6)
                plt.colorbar(scatter, label='Cluster')
                ax.set_title('3D Scatter Plot (Colored by Cluster)')
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            
            # Plot 2: Colored by variable
            if var in reduced_df.columns:
                ax = fig.add_subplot(2, 2, 2, projection='3d')
                scatter = ax.scatter(reduced_df['PC1'], reduced_df['PC2'], reduced_df['PC3'],
                                   c=reduced_df[var].astype('category').cat.codes, 
                                   cmap='plasma', alpha=0.6)
                plt.colorbar(scatter, label=var)
                ax.set_title(f'3D Scatter Plot (Colored by {var})')
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            
            # Plot 3: Colored by PC1
            ax = fig.add_subplot(2, 2, 3, projection='3d')
            scatter = ax.scatter(reduced_df['PC1'], reduced_df['PC2'], reduced_df['PC3'],
                               c=reduced_df['PC1'], cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, label='PC1')
            ax.set_title('3D Scatter Plot (Colored by PC1)')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            
            # Plot 4: Colored by PC2
            ax = fig.add_subplot(2, 2, 4, projection='3d')
            scatter = ax.scatter(reduced_df['PC1'], reduced_df['PC2'], reduced_df['PC3'],
                               c=reduced_df['PC2'], cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, label='PC2')
            ax.set_title('3D Scatter Plot (Colored by PC2)')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            
            plt.tight_layout()
            plt.savefig(os.path.join(scatter_dir, f'{var}_3d_scatter.png'))
            plt.close()
            
            # Save the reduced data
            reduced_df.to_csv(os.path.join(scatter_dir, f'{var}_reduced_data.csv'), index=False)


def create_syllable_frequency_matrix(normalised_data, output_dir, color_map):
    """Create a matrix visualization of syllable frequencies."""
    print("\nCreating syllable frequency matrix visualization...")
    
    # Create output directory for matrices
    matrix_dir = os.path.join(output_dir, 'syllable_frequency_matrices')
    os.makedirs(matrix_dir, exist_ok=True)
    
    for var in normalised_data.keys():
        print(f"\nProcessing {var}...")
        df = normalised_data[var].copy()
        
        # Remove non-syllable columns
        columns_to_drop = [col for col in [var, 'cluster', 'num_observations'] if col in df.columns]
        syllable_data = df.drop(columns=columns_to_drop)
        
        # Get conditions and syllables
        conditions = df[var].values if var in df.columns else df.index
        syllables = syllable_data.columns
        
        # Get the frequency matrix
        freq_matrix = syllable_data.values
        
        # Check if this variable has cluster information
        has_clusters = 'cluster' in df.columns
        
        if has_clusters:
            print(f"  Found cluster information for {var}...")
            cluster_labels = df['cluster'].values
            
            # Create a sorting dataframe for proper cluster organization
            sort_df = pd.DataFrame({
                'condition': conditions,
                'cluster': cluster_labels,
                'index': range(len(conditions))
            })
            
            # Handle noise cluster (-1) by placing it at the end
            sort_df['cluster'] = sort_df['cluster'].replace(-1, float('inf'))
            sort_df = sort_df.sort_values(['cluster', 'condition'])
            sorted_indices = sort_df['index'].values
            
            # Apply sorting to our data
            freq_matrix = freq_matrix[sorted_indices]
            conditions = conditions[sorted_indices]
            cluster_labels = cluster_labels[sorted_indices]
            
            # Create simplified cluster labels (G#)
            cluster_map = {-1: 'N'}  # Keep 'N' for noise
            unique_clusters = sorted(list(set(filter(lambda x: x != -1, cluster_labels))))
            for i, c in enumerate(unique_clusters, 1):
                cluster_map[c] = f'G{i}'
            
            # Create labels for conditions that include cluster info
            y_labels = [f"{cond}" for cond in conditions]
            cluster_annotations = [f"{cluster_map[c]}" for c in cluster_labels]
            
            # Determine cluster boundaries for visualization
            cluster_boundaries = []
            current_cluster = None
            for i, cluster in enumerate(cluster_labels):
                if cluster != current_cluster:
                    if current_cluster is not None:
                        cluster_boundaries.append(i)
                    current_cluster = cluster
            
            # Add cluster info to title
            cluster_title = " (Grouped)"
        else:
            print(f"  No cluster information for {var}, using alphabetical ordering...")
            # Sort alphabetically if no clusters
            sort_indices = np.argsort(conditions)
            freq_matrix = freq_matrix[sort_indices]
            conditions = conditions[sort_indices]
            
            y_labels = conditions
            cluster_annotations = None
            cluster_boundaries = []
            cluster_title = ""
        
        # Create figure with larger size and proper layout for colorbar
        fig = plt.figure(figsize=(15, 12))
        
        # Create GridSpec layout
        gs = gridspec.GridSpec(1, 20)
        
        # Create main axis for the heatmap
        ax_main = plt.subplot(gs[0, :18])  # Main plot takes first 18 columns
        
        # Create custom colormap for each syllable
        custom_cmap = {}
        for syllable in syllables:
            base_color = color_map.get(syllable, '#FFFFFF')  # Use white if syllable not in color_map
            # Convert the hex color to RGB
            rgb = np.array(plt.matplotlib.colors.to_rgb(base_color))
            # Create a list of colors with varying intensity
            colors = [(1, 1, 1)]  # Start with white
            for i in range(1, 101):
                intensity = i / 100
                color = rgb * intensity + np.array([1, 1, 1]) * (1 - intensity)
                colors.append(tuple(color))
            custom_cmap[syllable] = colors
        
        # Plot each cell with appropriate color intensity
        for i in range(len(conditions)):
            for j in range(len(syllables)):
                value = freq_matrix[i, j]
                syllable = syllables[j]
                
                # Get color based on value intensity
                colors = custom_cmap[syllable]
                color_idx = int(value * 100) if value < 1 else 100
                cell_color = colors[color_idx]
                
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=cell_color,
                                   edgecolor='none')
                ax_main.add_patch(rect)
        
        # Set axis limits
        ax_main.set_xlim(-0.5, len(syllables) - 0.5)
        ax_main.set_ylim(-0.5, len(conditions) - 0.5)
        
        # Add cluster markers if available
        if has_clusters:
            # Create a cluster legend on the right side
            ax_right = ax_main.twinx()
            ax_right.set_ylim(-0.5, len(conditions) - 0.5)
            
            # Set cluster labels with larger font
            ax_right.set_yticks(range(len(conditions)))
            ax_right.set_yticklabels(cluster_annotations, fontsize=20, fontweight='bold')
            ax_right.set_ylabel('Group', fontsize=22, fontweight='bold')
            
            # Add visual cluster boundaries with white lines
            for boundary in cluster_boundaries:
                ax_main.axhline(y=boundary - 0.5, color='white', linewidth=2)
        
        # Customize the plot with larger fonts
        plt.suptitle(f'Syllable Frequency Matrix - {var}{cluster_title}', 
                    y=1.02, fontsize=24, fontweight='bold')
        ax_main.set_xlabel('Syllables', fontsize=22, fontweight='bold')
        ax_main.set_ylabel(var, fontsize=22, fontweight='bold')
        
        # Set ticks and ensure they're visible
        ax_main.set_xticks(range(len(syllables)))
        ax_main.set_xticklabels(syllables, rotation=90, ha='right', fontsize=20)
        ax_main.set_yticks(range(len(conditions)))
        ax_main.set_yticklabels(y_labels, fontsize=20)
        
        # Show tick marks but not the full spines
        ax_main.tick_params(axis='both', which='both', length=4, width=1)
        for spine in ax_main.spines.values():
            spine.set_visible(False)
        
        # Add colorbar with its own axis
        ax_cbar = plt.subplot(gs[0, 19])  # Colorbar takes the last column
        norm = plt.Normalize(vmin=0, vmax=freq_matrix.max())
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax_cbar)
        cbar.set_label('Normalized Frequency', fontsize=22, fontweight='bold')
        cbar.ax.tick_params(labelsize=20)
        
        # Adjust layout
        plt.subplots_adjust(right=0.9, bottom=0.2)  # Adjust bottom to accommodate rotated labels
        
        # Save the plot
        plt.savefig(os.path.join(matrix_dir, f'{var}_frequency_matrix.png'), 
                    bbox_inches='tight', dpi=300, facecolor='white',
                    pad_inches=0.5)
        plt.close()


def create_binary_comparison_plot(normalised_data, output_dir, color_map):
    """Create comparative visualizations for binary conditions."""
    print("\nCreating binary comparison visualizations...")
    
    # Create output directory for binary comparison
    binary_dir = os.path.join(output_dir, 'binary_comparison')
    os.makedirs(binary_dir, exist_ok=True)
    
    # Define the binary conditions to plot
    binary_conditions = ['threat_action', 'safehouse', 'visibility']

    # First pass to find global min and max differences for consistent y-axis limits
    all_differences = []
    for condition in binary_conditions:
        if condition in normalised_data:
            df = normalised_data[condition].copy()
            columns_to_drop = [condition, 'cluster', 'num_observations'] if 'cluster' in df.columns else [condition, 'num_observations']
            syllable_data = df.drop(columns=columns_to_drop)
            conditions = df[condition].values
            unique_values = df[condition].unique()
            if len(unique_values) == 2:
                data_1 = syllable_data[conditions == unique_values[0]].mean()
                data_2 = syllable_data[conditions == unique_values[1]].mean()
                differences = data_1 - data_2
                all_differences.extend(differences)

    # Calculate global min and max with some padding
    if all_differences:
        global_min = min(all_differences)
        global_max = max(all_differences)
        padding = (global_max - global_min) * 0.2  # 20% padding
        y_min = global_min - padding
        y_max = global_max + padding
    
    for condition in binary_conditions:
        if condition in normalised_data:
            df = normalised_data[condition].copy()
            
            # Remove non-syllable columns
            columns_to_drop = [condition, 'cluster', 'num_observations'] if 'cluster' in df.columns else [condition, 'num_observations']
            syllable_data = df.drop(columns=columns_to_drop)
            
            # Get conditions and syllables
            conditions = df[condition].values
            syllables = syllable_data.columns
            
            # Get unique binary values
            unique_values = df[condition].unique()
            if len(unique_values) != 2:
                print(f"Skipping {condition} as it does not have exactly two unique values.")
                continue
            
            # Create separate DataFrames for each binary condition
            data_1 = syllable_data[conditions == unique_values[0]].mean()
            data_2 = syllable_data[conditions == unique_values[1]].mean()
            
            # Get sample sizes
            n_1 = len(syllable_data[conditions == unique_values[0]])
            n_2 = len(syllable_data[conditions == unique_values[1]])
            
            # Create the figure
            plt.figure(figsize=(15, 8))
            
            # Set the width of each bar and positions of the bars
            width = 0.35
            x = np.arange(len(syllables))
            
            # Create bars
            plt.bar(x - width/2, data_1, width, label=f'{unique_values[0]} (N={n_1})',
                    color=[color_map.get(s, '#D3D3D3') for s in syllables], alpha=0.7)
            plt.bar(x + width/2, data_2, width, label=f'{unique_values[1]} (N={n_2})',
                    color=[color_map.get(s, '#D3D3D3') for s in syllables], alpha=0.4)
            
            # Customize the plot
            plt.xlabel('Syllables')
            plt.ylabel('Normalized Frequency')
            plt.title(f'Comparison of Syllable Frequencies: {unique_values[0]} vs. {unique_values[1]}')
            plt.xticks(x, syllables, rotation=45, ha='right')
            plt.legend()
            
            # Add value labels
            def add_value_labels(x, y, offset):
                for i, v in enumerate(y):
                    plt.text(x[i], v, f'{v:.3f}', ha='center', va='bottom', rotation=0,
                            fontsize=8, color='black')
            
            add_value_labels(x - width/2, data_1, -0.05)
            add_value_labels(x + width/2, data_2, 0.05)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(binary_dir, f'{condition}_comparison.png'))
            plt.close()
            
            # Save the data
            comparison_df = pd.DataFrame({
                'syllable': syllables,
                unique_values[0]: data_1,
                unique_values[1]: data_2,
                'difference': data_1 - data_2
            })
            comparison_df.to_csv(os.path.join(binary_dir, f'{condition}_comparison.csv'), index=False)
            
            # Create a difference plot with adjusted layout for rotated labels
            plt.figure(figsize=(15, 8))
            
            # Create main axis with extra space on the left for labels
            gs = gridspec.GridSpec(1, 20)
            ax_main = plt.subplot(gs[0, 2:19])  # Main plot takes columns 2-18
            
            differences = data_1 - data_2
            colors = [color_map.get(s, '#D3D3D3') for s in syllables]
            
            # Create bars with different colors for positive and negative differences
            bars = plt.bar(x, differences, color=colors, alpha=0.7)
            
            # Color the bars based on positive/negative values
            for bar, diff in zip(bars, differences):
                if diff < 0:
                    bar.set_alpha(0.4)
            
            plt.xlabel('Syllables', fontsize=16, fontweight='bold')
            plt.title(f'Difference in Syllable Frequencies: {unique_values[0]} vs. {unique_values[1]}')
            plt.xticks(x, syllables, rotation=90, ha='right', fontsize=14, fontweight='bold')
            
            # Set consistent y-axis limits
            plt.ylim(y_min, y_max)
            
            # Remove y-axis and its label
            ax_main.yaxis.set_visible(False)
            
            # Remove box lines except the bottom
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)
            ax_main.spines['left'].set_visible(False)
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add rotated condition labels on the sides
            label_x = -0.1  # Position for left label
            plt.text(label_x, y_max * 0.95, unique_values[0],
                    transform=ax_main.transAxes,
                    rotation=90,
                    ha='right',
                    va='top',
                    fontsize=14,
                    fontweight='bold')
            
            plt.text(label_x, y_min * 0.95, unique_values[1],
                    transform=ax_main.transAxes,
                    rotation=90,
                    ha='right',
                    va='bottom',
                    fontsize=14,
                    fontweight='bold')
            
            # Add value labels as percentages
            for i, v in enumerate(differences):
                percentage = v * 100  # Convert to percentage
                plt.text(i, v, f'{percentage:.1f}%', 
                        ha='center', 
                        va='bottom' if v > 0 else 'top',
                        fontsize=14,
                        # fontweight='bold',
                        color='black')
            
            plt.tight_layout()
            plt.savefig(os.path.join(binary_dir, f'{condition}_differences.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # Create a single figure with subplots for all conditions
            fig, axes = plt.subplots(len(binary_conditions), 1, figsize=(15, 8 * len(binary_conditions)))
            for idx, condition in enumerate(binary_conditions):
                if condition in normalised_data:
                    df = normalised_data[condition].copy()
                    columns_to_drop = [condition, 'cluster', 'num_observations'] if 'cluster' in df.columns else [condition, 'num_observations']
                    syllable_data = df.drop(columns=columns_to_drop)
                    conditions = df[condition].values
                    syllables = syllable_data.columns
                    unique_values = df[condition].unique()
                    if len(unique_values) != 2:
                        continue
                    data_1 = syllable_data[conditions == unique_values[0]].mean()
                    data_2 = syllable_data[conditions == unique_values[1]].mean()
                    n_1 = len(syllable_data[conditions == unique_values[0]])
                    n_2 = len(syllable_data[conditions == unique_values[1]])
                    ax = axes[idx]
                    width = 0.35
                    x = np.arange(len(syllables))
                    ax.bar(x - width/2, data_1, width, label=f'{unique_values[0]} (N={n_1})',
                           color=[color_map.get(s, '#D3D3D3') for s in syllables], alpha=0.7)
                    ax.bar(x + width/2, data_2, width, label=f'{unique_values[1]} (N={n_2})',
                           color=[color_map.get(s, '#D3D3D3') for s in syllables], alpha=0.4)
                    ax.set_xlabel('Syllables')
                    ax.set_ylabel('Normalized Frequency')
                    ax.set_title(f'Comparison of Syllable Frequencies: {unique_values[0]} vs. {unique_values[1]}')
                    ax.set_xticks(x)
                    ax.set_xticklabels(syllables, rotation=90, ha='right')
                    ax.legend()
                    
                    # Add value labels
                    add_value_labels(x - width/2, data_1, -0.05)
                    add_value_labels(x + width/2, data_2, 0.05)
            
            plt.tight_layout()
            plt.savefig(os.path.join(binary_dir, 'combined_binary_comparison.png'))
            plt.close()


def create_conference_style_matrix(normalised_data, output_dir, color_map):
    """Create a conference-friendly matrix visualization."""
    print("\nCreating conference-style matrix visualization...")
    
    # Create output directory
    conference_dir = os.path.join(output_dir, 'conference_figures')
    os.makedirs(conference_dir, exist_ok=True)
    
    for var in normalised_data.keys():
        print(f"\nProcessing {var}...")
        df = normalised_data[var].copy()
        
        if 'cluster' in df.columns:
            # Get cluster information
            clusters = df['cluster'].unique()
            clusters = sorted([c for c in clusters if c != -1]) + ([-1] if -1 in clusters else [])
            
            # Create a mapping of cluster numbers to "G#" format
            cluster_mapping = {cluster: f"G{i+1}" if cluster != -1 else "N" 
                             for i, cluster in enumerate(clusters)}
            
            # Remove non-syllable columns
            columns_to_drop = [col for col in [var, 'cluster', 'num_observations'] if col in df.columns]
            syllable_data = df.drop(columns=columns_to_drop)
            
            # Sort the dataframe by cluster
            df['cluster_for_sort'] = df['cluster'].replace(-1, float('inf'))
            df = df.sort_values('cluster_for_sort')
            df = df.drop('cluster_for_sort', axis=1)
            
            # Get conditions in sorted order
            conditions = df[var].values
            
            # Update syllable_data to match the new order
            syllable_data = df[syllable_data.columns]
            
            # Calculate cluster boundaries and positions
            boundaries = []
            current_pos = 0
            cluster_positions = []
            cluster_sizes = []
            
            for cluster in clusters:
                cluster_size = len(df[df['cluster'] == cluster])
                cluster_sizes.append(cluster_size)
                if current_pos > 0:  # Add boundary before each cluster except the first
                    boundaries.append(current_pos - 0.5)
                # Calculate position for cluster label (center of each group)
                cluster_positions.append(current_pos + (cluster_size / 2))
                current_pos += cluster_size
            
            # Create the matrix visualization with adjusted figure size for square cells
            n_rows, n_cols = syllable_data.shape
            fig_width = 15
            fig_height = fig_width * (n_rows / n_cols)
            plt.figure(figsize=(fig_width, fig_height))
            
            # Use GridSpec to create a layout with more space for labels
            gs = gridspec.GridSpec(1, 20)  # 20 columns to allow fine control
            ax_main = plt.subplot(gs[0, :15])  # Main plot takes first 15 columns
            
            # Create heatmap with square cells
            matrix_data = syllable_data.values
            im = ax_main.imshow(matrix_data, aspect='equal', cmap='plasma')
            
            # Add white lines for cluster boundaries
            for boundary in boundaries:
                ax_main.axhline(y=boundary, color='white', linewidth=2)
            
            # Customize x-axis (syllables)
            plt.xticks(range(len(syllable_data.columns)), syllable_data.columns, 
                      rotation=90, ha='right', fontsize=20, fontweight='bold')
            plt.xlabel('Syllables', fontsize=20, fontweight='bold')
            
            # Add condition labels on the left
            ax_main.yaxis.set_label_position("left")
            ax_main.yaxis.tick_left()
            
            # Set condition labels
            condition_positions = np.arange(len(conditions))
            ax_main.set_yticks(condition_positions)
            ax_main.set_yticklabels(conditions, fontsize=20)
            
            # Add cluster labels on the right
            ax_right = ax_main.twinx()
            ax_right.yaxis.set_label_position("right")
            ax_right.yaxis.tick_right()
            
            # Set cluster labels with increased spacing from the plot
            ax_right.set_ylim(-0.5, len(conditions)-0.5)  # Match the main axis limits
            ax_right.set_yticks(cluster_positions)
            ax_right.set_yticklabels([cluster_mapping[c] for c in clusters],
                                    fontsize=20, fontweight='bold')
            
            # Add colorbar with adjusted position
            ax_cbar = plt.subplot(gs[0, 17:19])  # Colorbar takes columns 17-18, leaving more space
            plt.colorbar(im, cax=ax_cbar)
            ax_cbar.tick_params(labelsize=20)   
            
            # Remove unnecessary spines
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)
            ax_main.spines['left'].set_visible(False)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            
            plt.suptitle(f'Grouped Syllable Patterns - {var}', fontsize=20, fontweight='bold', y=1.02)
            
            # Adjust layout
            plt.subplots_adjust(right=0.85, wspace=0.4)  # Increased wspace for more padding
            
            # Save the figure
            plt.savefig(os.path.join(conference_dir, f'{var}_matrix_detailed.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # Create simplified version without condition labels
            plt.figure(figsize=(fig_width, fig_height))
            gs = gridspec.GridSpec(1, 20)
            ax_main = plt.subplot(gs[0, :15])
            
            im = ax_main.imshow(matrix_data, aspect='equal', cmap='plasma')
            
            # Add white lines for cluster boundaries
            for boundary in boundaries:
                ax_main.axhline(y=boundary, color='white', linewidth=2)
            
            plt.xticks(range(len(syllable_data.columns)), syllable_data.columns, 
                      rotation=90, ha='right', fontsize=20, fontweight='bold')
            plt.xlabel('Syllables', fontsize=20, fontweight='bold')
            
            # Add cluster labels on the right
            ax_main.yaxis.set_label_position("right")
            ax_main.yaxis.tick_right()
            ax_main.set_yticks(cluster_positions)
            ax_main.set_yticklabels([cluster_mapping[c] for c in clusters],
                                   fontsize=20, fontweight='bold')
            
            # Add colorbar with adjusted position
            ax_cbar = plt.subplot(gs[0, 17:19])
            plt.colorbar(im, cax=ax_cbar)
            ax_cbar.tick_params(labelsize=20)
            
            # Remove unnecessary spines
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)
            ax_main.spines['left'].set_visible(False)
            
            plt.suptitle(f'Grouped Syllable Patterns - {var}', fontsize=20, fontweight='bold', y=1.02)
            
            # Adjust layout
            plt.subplots_adjust(right=0.85, wspace=0.4)  # Increased wspace for more padding
            
            # Save the simplified figure
            plt.savefig(os.path.join(conference_dir, f'{var}_matrix_simplified.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()


def plot_hdbscan_clusters(umap_result, labels, output_dir, method='HDBSCAN'):
    """Plot HDBSCAN clustering results with larger font sizes."""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                         c=labels, cmap='plasma', alpha=0.6, s=100)  # Increased point size
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster', fontsize=16, fontweight='bold')  # Increased from default
    cbar.ax.tick_params(labelsize=14)  # Increased from default
    
    # Set labels and title
    plt.xlabel('UMAP1', fontsize=16, fontweight='bold')
    plt.ylabel('UMAP2', fontsize=16, fontweight='bold')
    plt.title(f'{method} Clustering Results', fontsize=20, fontweight='bold')
    
    # Increase tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method.lower()}_clusters.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()


def perform_group_analysis(grouped_data, output_dir, color_map, all_labels):
    groups_analysis = dict()
    for key in grouped_data.keys():
        group_dir = os.path.join(output_dir, str(key))
        os.makedirs(group_dir, exist_ok=True)
        frequency_matrix = pd.DataFrame(columns=[key] + list(all_labels))
        
        for group_name, group in grouped_data[key]:
            if 'predicted_label' in group.columns:
                # Initialize label counts with all labels set to 0
                label_counts = pd.Series(0, index=all_labels)
            
                # Add real observations to the initialized label counts
                real_counts = group['predicted_label'].value_counts()
                label_counts.update(real_counts)
                
                num_observations = label_counts.sum()
                plot_label_frequency(label_counts.sort_values(ascending=False), f'Frequency by {group_name} | {key} | (N={num_observations})', os.path.join(group_dir, f'predicted_labels_{key}_{group_name}.png'), color_map)   
                
                # Append the frequency data to the frequency matrix
                row_data = {key: group_name}
                for label in all_labels:
                    row_data[label] = label_counts.get(label, 0)
                
                # Create a new DataFrame with all required columns
                new_row = pd.DataFrame([row_data], columns=[key] + list(all_labels))
                
                # Concatenate the new row to the frequency matrix
                if frequency_matrix.empty:
                    frequency_matrix = new_row
                else:
                    frequency_matrix = pd.concat([frequency_matrix, new_row], ignore_index=True)

        # Save the frequency matrix to a CSV file
        frequency_matrix.to_csv(os.path.join(group_dir, f'{key}_frequency_matrix.csv'), index=False)
        groups_analysis[key] = frequency_matrix
        
        # Plot counts per key and per case
        unique_groups = list(grouped_data[key].groups.keys())
        num_groups = len(unique_groups)
        num_cols = 3
        num_rows = (num_groups + num_cols - 1) // num_cols

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows), sharex=True)
        axes = axes.flatten()

        # Get all unique labels across all groups
        all_unique_labels = set()
        for _, group in grouped_data[key]:
            if 'predicted_label' in group.columns:
                all_unique_labels.update(group['predicted_label'].unique())
        all_unique_labels = sorted(list(all_unique_labels))

        for ax, (group_name, group) in zip(axes, grouped_data[key]):
            if 'predicted_label' in group.columns:
                label_counts = group['predicted_label'].value_counts()
                num_observations = label_counts.sum()
                plot_df = pd.DataFrame({
                    'label': label_counts.index,
                    'count': label_counts.values,
                })
                sns.barplot(data=plot_df, x='label', y='count', hue=label_counts.index, palette=color_map, ax=ax, legend=False)
                ax.set_title(f'Counts for {group_name} | {key}')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('Count')
                ax.set_xticks(range(len(all_unique_labels)))
                ax.set_xticklabels(all_unique_labels, rotation=45, ha='right')
        
        # Hide any unused subplots
        for ax in axes[num_groups:]:
            ax.set_visible(False)

        plt.tight_layout()
        output_figure_dir = os.path.join(output_dir, 'syllable_counts_all_figures', 'row_counts')
        os.makedirs(output_figure_dir, exist_ok=True)
        plt.savefig(os.path.join(output_figure_dir, f'counts_per_{key}_and_case.png'))
        plt.close()

    return groups_analysis



def process_and_analyze_data(args, merged_df, grouped_by_group, color_map, additional_columns, columns_to_process):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, 'last_pose_output')
    os.makedirs(output_dir, exist_ok=True)

    # Create syllable_counts_all_figures directory
    syllable_counts_dir = os.path.join(output_dir, 'syllable_counts_all_figures')
    os.makedirs(syllable_counts_dir, exist_ok=True)

    if 'predicted_label' in merged_df.columns:
        # Get all unique labels from the labels data
        all_labels = merged_df['predicted_label'].unique()

        # Initialize a DataFrame to store the overall frequency matrix
        frequency_matrix = pd.DataFrame(columns=all_labels)
        normalised_frequency_matrix = pd.DataFrame(columns=all_labels)

        # Initialize label counts with all labels set to 0
        label_counts = pd.Series(0, index=all_labels)
        
        # Add real observations to the initialized label counts
        real_counts = merged_df['predicted_label'].value_counts()
        label_counts.update(real_counts)
        
        num_observations = label_counts.sum()
        plot_label_frequency(label_counts.sort_values(ascending=False), f'Frequency by Labels | (N={num_observations})', os.path.join(syllable_counts_dir, 'predicted_labels_overall.png'), color_map)
        
        # Normalise the label counts
        normalised_label_counts = label_counts / num_observations
        plot_label_frequency(normalised_label_counts.sort_values(ascending=False), f'Normalised Frequency by Labels | (N={num_observations})', os.path.join(syllable_counts_dir, 'normalised_predicted_labels_overall.png'), color_map)

        # Append the frequency data to the frequency matrix
        frequency_matrix = pd.concat([frequency_matrix, pd.DataFrame([label_counts], columns=all_labels)], ignore_index=True)
        
        # Create a new DataFrame for normalized counts and ensure all columns are present
        normalized_row = pd.DataFrame(columns=all_labels)
        for label in all_labels:
            normalized_row.loc[0, label] = normalised_label_counts.get(label, 0)
        normalised_frequency_matrix = pd.concat([normalised_frequency_matrix, normalized_row], ignore_index=True)

        # Save the overall frequency matrix to a CSV file
        frequency_matrix.to_csv(os.path.join(syllable_counts_dir, 'predicted_labels_overall_frequency_matrix.csv'))
        normalised_frequency_matrix.to_csv(os.path.join(syllable_counts_dir, 'normalised_predicted_labels_overall_frequency_matrix.csv'))

    # Perform group analysis
    groups_analysis = perform_group_analysis(grouped_by_group, output_dir, color_map, all_labels)

    # Normalize the frequency data for episodes
    normalised_data = dict()
    for key in groups_analysis.keys():
        for column in columns_to_process:
            if column in groups_analysis[key].columns:
                print(f"Normalising {column} for {key}")
                normalised_data.update({column: normalise_data(groups_analysis[key], skip_columns=column)})

    # Save the normalized data
    normalised_data_dir = os.path.join(output_dir, 'normalised_data')
    os.makedirs(normalised_data_dir, exist_ok=True)
    for key in normalised_data.keys():
        normalised_data[key].to_csv(os.path.join(normalised_data_dir, f'{key}_normalised_data.csv'), index=False)

    # Plot normalized data with subplots for each row
    plot_normalised_data_subplots(normalised_data, output_dir, color_map)

    # Create analysis directory
    analysis_dir = os.path.join(output_dir, 'feature_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    feature_columns = columns_to_process

    merged_df = merged_df[['ppid', 'session', 'trial', 'predicted_label', 'reward', 'day_time'] + additional_columns]
    
    # Update the feature importance analysis call
    feature_importance = analyze_feature_importance(
        merged_df[feature_columns],  # data_matrix
        merged_df['predicted_label'],  # labels
        'Feature Importance Analysis',  # title
        os.path.join(analysis_dir, 'feature_importance.png')  # output_path
    )
    
    correlations = analyze_correlations(merged_df, analysis_dir, feature_columns)
    analyze_feature_distributions(merged_df, analysis_dir, feature_columns)
    stats_results = perform_statistical_tests(merged_df, analysis_dir, feature_columns)

    merged_df.to_csv(os.path.join(analysis_dir, 'last_pose_labels_with_metadata_and_features.csv'), index=False)
    
    # Call interaction analysis
    interaction_results = analyze_interactions_logistic_regression(merged_df, analysis_dir, feature_columns)
    
    print(f"\nFeature analysis results saved to {analysis_dir}")
    print("\nFeature Importance Analysis:")
    print(feature_importance.head())
    print("\nTop Correlations with Predicted Label:")
    print(correlations.head())
    print("\nStatistical Test Results:")
    print(stats_results)

    # Analyze combined effects
    print("\nAnalyzing combined effects of experimental variables...")
    coef_df, feature_importance, anova_results = analyze_combined_effects(merged_df, analysis_dir, feature_columns)
    
    print("\nCombined Effects Analysis Results:")
    print("\nTop 10 Logistic Regression Coefficients:")
    print(coef_df.head(10))
    print("\nTop 10 Random Forest Feature Importances:")
    print(feature_importance.head(10))
    print("\nANOVA Results:")
    print(anova_results)

    # Analyze clustering patterns
    variables_of_interest = ["ppid", "episode_id", "threat", "scenario"]
    normalised_hdbscan_df = analyze_clustering_patterns(normalised_data, output_dir, variables_of_interest, color_map)

    # Create data visualizations
    create_data_visualizations(normalised_data, output_dir, variables_of_interest, color_map)

    # Create detailed observation scatter plots
    plot_observation_scatter(normalised_data, output_dir, variables_of_interest, color_map)

    # Create normalized syllable frequency matrices
    create_syllable_frequency_matrix(normalised_hdbscan_df, output_dir, color_map)

    # Create binary comparison visualization
    create_binary_comparison_plot(normalised_data, output_dir, color_map)
    
    # Create conference-style visualizations
    create_conference_style_matrix(normalised_hdbscan_df, output_dir, color_map)

    return groups_analysis

def parse_args():
    parser = argparse.ArgumentParser(description='Process CSV files to get the last pose labels.')
    parser.add_argument('--labels-data', required=True, help='Path to the labels CSV file')
    parser.add_argument('--labels-column', required=True, help='Label column')
    parser.add_argument('--metadata-data', required=True, help='Path to the metadata CSV file')
    parser.add_argument('--group-columns', nargs='+', default=['ppid', 'session', 'trial'], help='Columns to group by')
    parser.add_argument('--output-dir', required=True, help='Path to the output directory')
    parser.add_argument('--colormap-path', required=True, help='Path to the color map pickle file')
    parser.add_argument('--mode', default='last_pose', choices=['last_pose', 'sequence_poses'], help='Mode to run the script in. Available modes: last_pose') 
    parser.add_argument('--number-of-frames', type=int, default=1, help='Number of frames before unsuccessfull threat encounter to use for the analysis, default is 1')
    parser.add_argument('--data-collection-rate', type=int, default=10, help='Data collection rate, default is 100')
    parser.add_argument('--coordinates', required=True, help='Directory path to the coordinates CSV files')
    parser.add_argument('--ffmpeg-executable', default='./ffmpeg.exe', help='Path to the ffmpeg executable')
    parser.add_argument('--stick-figure-template', default='qualisys_csv', help='Which stickfigure should be used')
    parser.add_argument('--stickfigure-group-mode', default=None, help='How do you want to group the data')
    parser.add_argument('--label-column', default='predicted_label', help='Which column contains the labels')
    parser.add_argument('--skip-syllable-sequence', action='store_true', help='Skip the syllable sequence plot')
    parser.add_argument('--skip-syllable-counts', action='store_true', help='Skip the syllable counts plot')
    parser.add_argument('--skip-stickfigures', action='store_true', help='Skip the stickfigures plot')
    parser.add_argument('--skip-syllable-transition-matrices', action='store_true', help='Skip the syllable transition matrices plot')
    return parser.parse_args()


def load_color_map(colormap_path, args, save_color_map=True):
    # Load the color map from the pickle file
    with open(colormap_path, 'rb') as f:
        color_map = pickle.load(f)

    # Add pastel grey for 'other' label and ensure consistent special cases
    color_map['other'] = '#D3D3D3'
    color_map[-1] = '#D3D3D3'  # Special case for unknown labels
    color_map["-1"] = '#D3D3D3'  # Special case for unknown labels as string

    if save_color_map:
        # Save the color map
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        color_map_path = os.path.join(args.output_dir, "colour_map.pkl")
        with open(color_map_path, 'wb') as f:
            pickle.dump(color_map, f)
        print(f"\n\033[34m\nColor map saved to: {color_map_path}\033[0m\n")

    return color_map, color_map_path

def run_script(script_name, args):
    """Run a Python script with the given arguments."""
    import subprocess
    command = ["python", script_name] + args
    
    # Set environment variable to indicate running as subprocess
    env = os.environ.copy()
    env['RUNNING_AS_SUBPROCESS'] = 'true'
    
    process = subprocess.Popen(command, env=env)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def create_script_arguments(args):
    """Create argument lists for each script in the pipeline."""

    syllable_sequence_output_dir = os.path.join(args.output_dir, "syllable_sequence")
    os.makedirs(syllable_sequence_output_dir, exist_ok=True)
    syllable_counts_output_dir = os.path.join(args.output_dir, "syllable_counts")
    os.makedirs(syllable_counts_output_dir, exist_ok=True)
    stickfigures_output_dir = os.path.join(args.output_dir, "stickfigures")
    os.makedirs(stickfigures_output_dir, exist_ok=True)
    syllable_transition_matrices_output_dir = os.path.join(args.output_dir, "syllable_transition_matrices")
    os.makedirs(syllable_transition_matrices_output_dir, exist_ok=True)

    script_args = {      
        'syllable_sequence': [
            "--data-file", args.labels_data,
            "--labels-column", args.label_column,
            "--data-collection-rate", str(args.data_collection_rate), 
            "--colour-map-object", args.colormap_path,
            "--output-path", syllable_sequence_output_dir
        ],
        
        'syllable_counts': [
            "--labels-path", args.labels_data,
            "--label-column", args.label_column,
            "--vr-metadata-path", args.metadata_data,
            "--output-dir", syllable_counts_output_dir,
            "--colour-map-object", args.colormap_path
        ],
        
        'stickfigures': [
            "--metadata-csv", args.labels_data,
            "--label-column", args.label_column,
            "--coordinates-path", args.coordinates,
            "--ffmpeg-executable-path", args.ffmpeg_executable,
            "--video-output-path", stickfigures_output_dir,
            "--data-sample-rate", str(args.data_collection_rate),
            "--stickfigure-template", args.stick_figure_template,
            "--save-as-video", "--provide-colour-map",
            "--colour-map-object", args.colormap_path,
            "--plot-sample-rate", str(args.data_collection_rate),
            *(["--group-mode", args.stickfigure_group_mode] if args.stickfigure_group_mode is not None else [])
        ],
        'syllable_transition_matrices': [
            "--data-file", args.labels_data,
            "--label-column", args.label_column,
            "--output-dir", syllable_transition_matrices_output_dir
        ]
    }

    return script_args


@tictoc
def main():
    # Parse command-line arguments
    args = parse_args()

    # Load the color map
    color_map, color_map_path = load_color_map(args.colormap_path, args)
    args.colour_map = color_map_path

    # Additional columns for analysis
    additional_columns = ['scenario', 'episode_id', 'threat', 'visibility', 'distance', 'threat_action', 'weapons', 'safehouse'] 
    columns_to_process = additional_columns + ['trial', 'ppid', 'reward', 'day_time']
    output_dir_main = args.output_dir
    args.output_dir = os.path.join(args.output_dir, f"Last_Pose_{args.number_of_frames}_Frames")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'last_pose':
        # Process the CSV files
        merged_df, grouped_by_group = process_csv_files(args.labels_data, args.metadata_data, args.group_columns, groups_to_process=columns_to_process, output_dir=output_dir_main, tail_rows=1, save_all_poses=True)
        
        # Save the main dataset to the output directory for inspection
        merged_df.to_csv(os.path.join(args.output_dir, "last_pose_labels_with_metadata_and_features.csv"), index=False)
        
        # Process and analyze the data
        process_and_analyze_data(args, merged_df, grouped_by_group, color_map, additional_columns, columns_to_process)

    elif args.mode == 'sequence_poses':
        # Process the CSV files
        merged_df, grouped_by_group = process_csv_files(args.labels_data, args.metadata_data, args.group_columns, groups_to_process=columns_to_process, output_dir=output_dir_main, tail_rows=args.number_of_frames, save_all_poses=True)

        # Save the merged dataframe to a CSV file
        merged_df.to_csv(os.path.join(args.output_dir, "last_pose_labels_with_metadata_and_features.csv"), index=False)
        args.labels_data = os.path.join(args.output_dir, "last_pose_labels_with_metadata_and_features.csv")

        # Get all script arguments
        script_args = create_script_arguments(args)

        # Run the script
        if not args.skip_syllable_sequence:
            try:
                run_script("plot_syllable_sequences.py", script_args['syllable_sequence'])
            except Exception as e:
                print(f"\033[91mError running plot_syllable_sequences.py: {e}\033[0m")
        print("\n\033[94mFinished plotting syllable sequence\033[0m")
        if not args.skip_syllable_counts:
            try:
                run_script("get_syllable_counts.py", script_args['syllable_counts'])
            except Exception as e:
                print(f"\033[91mError running get_syllable_counts.py: {e}\033[0m")
        print("\n\033[94mFinished plotting syllable counts\033[0m")
        print(script_args['stickfigures'])
        if not args.skip_stickfigures:
            try:
                try:
                    run_script("plot_syllable_stickfigures.py", script_args['stickfigures'])
                except Exception as e:
                    print(f"\n\033[91mError running plot_syllable_stickfigures.py:\033[0m")
                    print(f"\033[91mError type: {type(e).__name__}\033[0m")
                    print(f"\033[91mError message: {str(e)}\033[0m")
                    print(f"\033[91mFull traceback:\033[0m")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"\033[91mError running plot_syllable_stickfigures.py: {e}\033[0m")
        print("\n\033[94mFinished plotting stickfigures\033[0m")
        if not args.skip_syllable_transition_matrices:
            try:
                run_script("get_syllable_transition_matrices.py", script_args['syllable_transition_matrices'])
            except Exception as e:
                print(f"\033[91mError running get_syllable_transition_matrices.py: {e}\033[0m")
        print("\n\033[94mFinished plotting syllable transition matrices\033[0m")
    else:
        print(f"\nMode {args.mode} not supported")
        print("current available modes are: last_pose")

    print(f"\n\033[94mCode completed successfully!\033[0m\n\n")

if __name__ == '__main__':
    os.system("cls||clear")
    print("\033[95m\nGet last pose labels by [uSerratos]\n\033[0m")
    main()
    plt.close('all')