from typing import Union
import pandas as pd
import os
import numpy as np
import cupy as cp
from cuml.ensemble import RandomForestClassifier
from cuml.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    make_scorer,
    accuracy_score,
    f1_score
)
from utils import print_welcome_message, tictoc, print_color, bold_color
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from natsort import natsorted
import re
import multiprocessing
import gc


def parse_arguments():
    """Parse command line arguments for random forest classifier."""
    parser = argparse.ArgumentParser(description="Train and use a Random Forest Classifier.")
    
    # Mode selection arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train a new classifier"
    )
    mode_group.add_argument(
        "--predict",
        action="store_true",
        help="Use an existing classifier for prediction"
    )
    mode_group.add_argument(
        "--train-and-predict",
        action="store_true",
        help="Train a new classifier and use it for prediction"
    )
    
    # Training arguments
    training_group = parser.add_argument_group("Training arguments")
    training_group.add_argument(
        "--feature-matrix-train",
        type=str,
        help="Path to the feature matrix CSV file for training (required if --train or --train-and-predict)"
    )
    training_group.add_argument(
        "--labels-column",
        type=str,
        default=None,
        help="Name of the column containing the labels (required if --train or --train-and-predict)"
    )
    training_group.add_argument(
        "--labels-file",
        type=str,
        default=None,
        help="Path to separate CSV file containing labels (optional)"
    )
    
    # Prediction arguments
    prediction_group = parser.add_argument_group("Prediction arguments")
    prediction_group.add_argument(
        "--feature-matrix-predict",
        type=str,
        default=None,
        help="Path to the feature matrix CSV file for prediction (required if --predict or --train-and-predict)"
    )
    prediction_group.add_argument(
        "--classifier-file",
        type=str,
        default=None,
        help="Path to the trained classifier pickle file (required if --predict, ignored if --train)"
    )
    prediction_group.add_argument(
        "--label-encoder-file",
        type=str,
        default=None,
        help="Path to the label encoder pickle file (required if --predict, ignored if --train)"
    )
    prediction_group.add_argument(
        "--probability-threshold",
        type=float,
        default=0.0,
        help="Minimum probability threshold for predictions. Predictions below this threshold will be labeled as '-1' (default: 0.0)"
    )
    prediction_group.add_argument(
        "--only-predict-unlabeled-predict",
        action="store_true",
        help="Only predict unlabeled data"
    )
    prediction_group.add_argument(
        "--ignore-unlabeled-train",
        action="store_true",
        help="Ignore unlabeled data in training"
    )

    args = parser.parse_args()
    
    # Validate dependencies
    if args.train or args.train_and_predict:
        if not args.feature_matrix_train:
            parser.error("--feature-matrix-train is required when using --train or --train-and-predict")
        if not args.labels_column:
            parser.error("--labels-column is required when using --train or --train-and-predict")
            
    if args.predict:
        if not args.classifier_file:
            parser.error("--classifier-file is required when using --predict")
        if not args.label_encoder_file:
            parser.error("--label-encoder-file is required when using --predict")
        if not args.feature_matrix_predict:
            parser.error("--feature-matrix-predict is required when using --predict")
            
    if args.train_and_predict and not args.feature_matrix_predict:
        parser.error("--feature-matrix-predict is required when using --train-and-predict")
    
    return args


def check_gpu_availability():
    """Check if GPU is available and return GPU memory info."""
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        # Get GPU memory info
        total_memory = cp.cuda.Device(0).mem_info[1]  # Total memory
        free_memory = cp.cuda.Device(0).mem_info[0]   # Free memory
        
        print_color(f"\nGPU Memory Info:", color=bold_color.GREEN)
        print_color(f"Total GPU Memory: {total_memory / 1024**2:.2f} MB", color=bold_color.GREEN)
        print_color(f"Free GPU Memory: {free_memory / 1024**2:.2f} MB", color=bold_color.GREEN)
        
        return True
    except Exception as e:
        print_color(f"\nGPU not available: {str(e)}", color=bold_color.RED)
        return False

def clear_gpu_memory():
    """Clear GPU memory."""
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        gc.collect()
    except:
        pass

@tictoc
def train_classifier(feature_matrix_file, labels_column, other_file_path_train=None, ignore_unlabeled=False):
    print_color("\nTRAINING", color=bold_color.GREEN)
    
    # Check GPU availability
    if not check_gpu_availability():
        raise RuntimeError("GPU not available. Please ensure you have a compatible NVIDIA GPU and RAPIDS installed in WSL.")
    
    # Calculate number of jobs to use (leave one CPU free)
    total_cpus = multiprocessing.cpu_count()
    n_jobs = max(1, total_cpus - 1)  # Leave one CPU free
    print_color(f"\nTotal CPU cores: {total_cpus}", color=bold_color.GREEN)
    print_color(f"Using {n_jobs} CPU cores (leaving one free)", color=bold_color.GREEN)
    
    # Create output directory for all training results
    output_dir = os.path.join(os.path.dirname(feature_matrix_file), 'training_results_gpu')
    os.makedirs(output_dir, exist_ok=True)
    print_color(f"\nSaving all training results to: {output_dir}\n", color=bold_color.BLUE)
    
    data = pd.read_csv(feature_matrix_file)
    # Columns to drop
    columns_to_drop = ['ppid', 'session', 'trial', 'time']

    # Drop the columns from the DataFrame
    data = data.drop(columns=columns_to_drop, errors='ignore')

    if other_file_path_train:
        other_data = pd.read_csv(other_file_path_train)
        labels = other_data.loc[:, labels_column].values
        feature_matrix = data.values
    else:
        labels = data.loc[:, labels_column].values  # assuming labels are in the last column
        feature_matrix = data.drop(columns=[labels_column]).values

    # Create a boolean mask for labels not equal to -1 or '-1'
    if ignore_unlabeled:
        mask = ~np.isin(labels.astype(str), ['-1'])
    else:
        mask = np.ones(len(labels), dtype=bool)
    
    # Apply the mask to the features and labels
    X = feature_matrix[mask].astype(np.float32)
    y = labels[mask].astype(str)
    
    # Fit the label encoder and transform the labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print_color("\nUnique classes in label encoder:", color=bold_color.GREEN)
    print_color(f"Classes: {le.classes_}", color=bold_color.GREEN)
    
    # Convert data to GPU arrays
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)
    
    # Initialize base classifier with GPU support
    base_clf = RandomForestClassifier(
        random_state=42,
        n_streams=1,  # For reproducible results
        verbose=1
    )

    # Use 5-fold cross-validation only
    n_folds = 5
    print_color(f"\nPerforming Grid Search with {n_folds}-fold cross-validation...", color=bold_color.GREEN)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [300],
        'max_depth': [30],
        'min_samples_split': [2],
        'min_samples_leaf': [4],
        'max_features': [1.0]  # Using 1.0 to consider all features
    }

    # Define scoring metrics
    scoring = {
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }

    # Initialize base classifier with GPU support
    base_clf = RandomForestClassifier(
        random_state=42,
        n_streams=1,  # For reproducible results
        verbose=1
    )

    # Use 5-fold cross-validation only
    n_folds = 5
    print_color(f"\nPerforming Grid Search with {n_folds}-fold cross-validation...", color=bold_color.GREEN)
    
    # Convert GPU arrays to CPU arrays only for GridSearchCV
    X_cpu = X_gpu.get()
    y_cpu = y_gpu.get()
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=n_folds,
        scoring=scoring,
        refit='f1_weighted',
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Fit GridSearchCV
    grid_search.fit(X_cpu, y_cpu)
    
    # Get the best classifier
    best_clf = grid_search.best_estimator_
    
    print_color("\nBest parameters:", color=bold_color.GREEN)
    for param, value in grid_search.best_params_.items():
        print_color(f"  {param}: {value}", color=bold_color.GREEN)
    print_color(f"Best F1 score: {grid_search.best_score_:.4f}", color=bold_color.GREEN)

    # Split the data into training and test sets for final evaluation using GPU arrays
    X_train, X_test, y_train, y_test = train_test_split(X_gpu, y_gpu, test_size=0.2, random_state=42)
    
    # Train the best classifier on the full training set
    best_clf.fit(X_train, y_train)

    # Save the trained classifier
    classifier_output = os.path.join(output_dir, 'random_forest_classifier.pkl')
    label_encoder_output = os.path.join(output_dir, 'label_encoder.pkl')
    with open(classifier_output, 'wb') as f:
        pickle.dump(best_clf, f)
    with open(label_encoder_output, 'wb') as f:
        pickle.dump(le, f)

    print_color(f"\nRandom Forest Classifier trained and saved to: {classifier_output}\n", color=bold_color.BLUE)
    print_color(f"Label encoder saved to: {label_encoder_output}\n", color=bold_color.BLUE)  

    # Get and save feature importance
    feature_names = (data.drop(columns=[labels_column]).columns 
                    if labels_column in data.columns 
                    else [f'feature_{i}' for i in range(X.shape[1])])
    
    # Get feature importance from the classifier
    try:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_clf.feature_importances_.get()  # Convert from GPU array to CPU
        })
    except AttributeError:
        # If feature_importances_ is not available, use a different method
        print_color("\nWarning: feature_importances_ not available. Using permutation importance instead.", color=bold_color.YELLOW)
        # Convert to CPU arrays only for permutation importance
        X_test_cpu = X_test.get()
        y_test_cpu = y_test.get()
        from sklearn.inspection import permutation_importance
        result = permutation_importance(best_clf, X_test_cpu, y_test_cpu, n_repeats=10, random_state=42)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean
        })
    
    # Sort features by importance
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    
    # Calculate cumulative importance
    feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
    
    # Filter features that account for 80% of the total importance
    top_features = feature_importance[feature_importance['cumulative_importance'] <= 0.8]
    
    # Plot feature importance for the top features
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_features)), top_features['importance'])
    plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
    plt.title('Feature Importance (Top 80%)')
    plt.tight_layout()
    
    # Save feature importance plot and data
    feature_importance_plot = os.path.join(output_dir, 'feature_importance.png')
    feature_importance_csv = os.path.join(output_dir, 'feature_importance.csv')
    plt.savefig(feature_importance_plot)
    plt.close()
    feature_importance.to_csv(feature_importance_csv, index=False)

    # Make predictions and probability estimates
    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)

    # Convert GPU arrays to CPU for sklearn metrics
    y_test_cpu = y_test.get()
    y_pred_cpu = y_pred.get()
    y_pred_proba_cpu = y_pred_proba.get()

    # Calculate and save ROC curve and AUC for each class
    plt.figure(figsize=(10, 8))
    colors = plt.colormaps['Set3'](np.linspace(0, 1, len(le.classes_)))
    
    # Convert classes to a format we can iterate over
    classes = le.classes_.to_pandas() if hasattr(le.classes_, 'to_pandas') else le.classes_
    
    for i, (label, color) in enumerate(zip(classes, colors)):
        if len(classes) == 2 and i == 0:  # Skip first class for binary classification
            continue
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test_cpu == i, y_pred_proba_cpu[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'ROC curve (class {label}) (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    # Save ROC curve plot
    roc_curve_output = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(roc_curve_output)
    plt.close()

    # Calculate and plot precision-recall curve for each class
    plt.figure(figsize=(10, 8))
    
    # Convert classes to a format we can iterate over
    classes = le.classes_.to_pandas() if hasattr(le.classes_, 'to_pandas') else le.classes_
    
    for i, (label, color) in enumerate(zip(classes, colors)):
        if len(classes) == 2 and i == 0:  # Skip first class for binary classification
            continue
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test_cpu == i, y_pred_proba_cpu[:, i])
        avg_precision = average_precision_score(y_test_cpu == i, y_pred_proba_cpu[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                label=f'PR curve (class {label}) (AP = {avg_precision:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.tight_layout()  
    
    # Save precision-recall curve plot
    pr_curve_output = os.path.join(output_dir, 'precision_recall_curves.png')
    plt.savefig(pr_curve_output)
    plt.close()

    # Print and save the classification report
    report = classification_report(y_test_cpu, y_pred_cpu)
    print(report)
    report_output = os.path.join(output_dir, 'Random_forest_classifier_classification_report.txt')
    with open(report_output, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nClass Probability Thresholds:\n")
        for i, label in enumerate(le.classes_):
            f.write(f"Class {label}: {np.mean(y_pred_proba_cpu[:, i]):.3f}\n")

    # Create confusion matrix
    cm = confusion_matrix(y_test_cpu, y_pred_cpu)
    
    # Convert classes to a format we can iterate over
    classes = le.classes_.to_pandas() if hasattr(le.classes_, 'to_pandas') else le.classes_
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='YlOrRd', 
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save confusion matrix plot
    confusion_matrix_output = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_output)
    plt.close()

    # Clear GPU memory
    clear_gpu_memory()

    print_color(f"\nRandom Forest Classifier trained and saved to: {classifier_output}\n", color=bold_color.BLUE)
    print_color(f"Label encoder saved to: {label_encoder_output}\n", color=bold_color.BLUE)     
    print_color(f"\nROC curves saved to: {roc_curve_output}\n", color=bold_color.BLUE)
    print_color(f"Precision-Recall curves saved to: {pr_curve_output}\n", color=bold_color.BLUE)
    print_color(f"Classification report saved to: {report_output}\n", color=bold_color.BLUE)
    print_color(f"Confusion matrix saved to: {confusion_matrix_output}\n", color=bold_color.BLUE)

    return classifier_output, label_encoder_output

def get_files_list(files_path) -> list:
    matrix_files_list = []
    for root, _, files in os.walk(files_path):
        for file in files:
            if file.endswith(".csv") and "feature_matrix_T" in file:
                matrix_files_list.append(os.path.join(root, file))

    if not matrix_files_list:
        print_color(f"\nNo transition matrices found in {files_path} or its subdirectories. Skipping...\n", color=bold_color.RED)
        return None
    
    matrix_files_list = natsorted(matrix_files_list)

    return matrix_files_list


@tictoc
def predict_with_classifier(classifier_file, label_encoder_file, feature_matrix_predict_file, probability_threshold=0.0, only_predict_unlabeled=False, label_column='label'):
    print_color(f"\nPREDICTING | Probability threshold: {probability_threshold}\n", color=bold_color.GREEN)
    import dask.dataframe as dd
    import numpy as np
    import os
    from tqdm import tqdm

    if isinstance(feature_matrix_predict_file, pd.DataFrame):
        feature_matrix_predict = feature_matrix_predict_file
    elif isinstance(feature_matrix_predict_file, dd.DataFrame):
        feature_matrix_predict = feature_matrix_predict_file.compute()  # Convert to pandas DataFrame
    elif os.path.isdir(feature_matrix_predict_file):
        files_list = get_files_list(feature_matrix_predict_file)
        if files_list is None:
            raise ValueError(f"No feature matrix files found in {feature_matrix_predict_file}")
        # Read and concatenate all files
        dfs = []
        for file in tqdm(files_list, desc="Reading feature matrix files"):
            file_split = file.split('\\')[-3:]
            ppid = int(file_split[0])
            session = int(f"{file_split[1][-3:]:03}")
            trial = int(re.search(r"T\d{3}", file_split[2]).group()[-3:])
            df = pd.read_csv(file)
            df['ppid'] = [ppid] * len(df)
            df['session'] = [session] * len(df)
            df['trial'] = [trial] * len(df)
            dfs.append(df)
        feature_matrix_predict = pd.concat(dfs, ignore_index=True, axis=0)
        feature_matrix_predict = feature_matrix_predict.reset_index(drop=True)
        # Move ppid, session, trial columns to front of DataFrame
        cols = feature_matrix_predict.columns.tolist()
        meta_cols = ['ppid', 'session', 'trial']
        other_cols = [col for col in cols if col not in meta_cols]
        feature_matrix_predict = feature_matrix_predict[meta_cols + other_cols]
    else:
        feature_matrix_predict = pd.read_csv(feature_matrix_predict_file)
    
    with open(classifier_file, 'rb') as f:
        clf = pickle.load(f)

    with open(label_encoder_file, 'rb') as f:
        le = pickle.load(f)

    print_color("\nLoaded classifier and label encoder:", color=bold_color.BLUE)
    print_color(f"Number of classes in label encoder: {len(le.classes_)}", color=bold_color.BLUE)
    print_color(f"Classes: {le.classes_}", color=bold_color.BLUE)

    # Store metadata and original labels
    feature_matrix_predict_original = feature_matrix_predict.copy()
    if label_column in feature_matrix_predict.columns:
        info = feature_matrix_predict[['ppid', 'session', 'trial', 'time', label_column]]
        info.columns = ['ppid', 'session', 'trial', 'time', 'original_label']
    else:
        info = feature_matrix_predict[['ppid', 'session', 'trial', 'time']]
        info['original_label'] = '-1'  # Add original_label column with default value

    info_original = info.copy()

    if only_predict_unlabeled:
        # Filter for unlabeled data
        mask = info['original_label'].astype(str) == '-1'
        info = info[mask]
        feature_matrix_predict = feature_matrix_predict[mask]

    # Columns to drop after filtering
    columns_to_drop = ['ppid', 'session', 'trial', 'time', 'Time', label_column]
    feature_matrix_predict = feature_matrix_predict.drop(columns=columns_to_drop, errors='ignore')

    # Process data in batches
    BATCH_SIZE = 10000  # Adjust this based on available memory
    n_samples = len(feature_matrix_predict)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE

    # Initialize arrays to store results
    n_classes = len(clf.classes_)  # Use classifier's classes instead of label encoder
    print_color(f"\nNumber of classes in classifier: {n_classes}", color=bold_color.BLUE)
    y_pred_proba_new_data = np.zeros((n_samples, n_classes))
    y_pred_new_data = np.empty(n_samples, dtype=object)

    print_color(f"\nProcessing {n_samples} samples in {n_batches} batches...", color=bold_color.BLUE)

    # Check if the feature matrix has the same number of features as the classifier expects
    n_features_expected = clf.n_features_in_
    n_features_actual = feature_matrix_predict.shape[1]
    
    if n_features_expected != n_features_actual:
        print_color(f"\nWARNING: Feature count mismatch!", color=bold_color.RED)
        print_color(f"Expected {n_features_expected} features but got {n_features_actual} features.", color=bold_color.RED)
        print_color("This may cause prediction errors. Please ensure your feature matrix matches the training data.", color=bold_color.RED)
        raise ValueError(f"Feature count mismatch: expected {n_features_expected} features but got {n_features_actual}")

    import sys, os
    # Redirect stdout to devnull to suppress output
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')

    for i in tqdm(range(n_batches), desc="Processing batches", unit="batch", leave=False):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, n_samples)
        
        # Get batch data
        batch_data = feature_matrix_predict.iloc[start_idx:end_idx]

        # Get probabilities for batch
        sys.stdout = devnull  # Redirect stdout to suppress output
        batch_proba = clf.predict_proba(batch_data.values)
        y_pred_proba_new_data[start_idx:end_idx] = batch_proba
        
        # Get predictions for batch
        batch_pred = clf.predict(batch_data.values)

        if hasattr(batch_pred, 'get'):
            # batch_pred is GPU array (cuPy), assume le is also GPU (cuML)
            # Pass the GPU array directly to cuML's inverse_transform
            transformed_labels = le.inverse_transform(batch_pred)
        else:
            # batch_pred is CPU array (NumPy), assume le is also CPU (sklearn)
            transformed_labels = le.inverse_transform(batch_pred)

        # Ensure the final result is a CPU array before assignment
        # The output of inverse_transform could be GPU (cuDF) or CPU (NumPy)
        if hasattr(transformed_labels, 'get'): # Check for cuPy array
            labels_cpu = transformed_labels.get()
        elif hasattr(transformed_labels, 'to_numpy'): # Check for cuDF Series/DataFrame
            labels_cpu = transformed_labels.to_numpy()
        else: # Assume it's already a NumPy array
            labels_cpu = transformed_labels
            
        y_pred_new_data[start_idx:end_idx] = labels_cpu
        sys.stdout = old_stdout  # Restore stdout

    # Close devnull
    devnull.close()

    # Get max probability for each prediction
    max_probabilities = np.max(y_pred_proba_new_data, axis=1)
    
    # Calculate the sum of probabilities for each sample
    sum_probabilities = np.sum(y_pred_proba_new_data, axis=1)
    
    # Normalize probabilities by dividing by the sum (this ensures they sum to 1)
    y_pred_proba_new_data_normalized = y_pred_proba_new_data / sum_probabilities[:, np.newaxis]
    
    print_color("\nProbability Distribution:", color=bold_color.BLUE)
    print_color(f"Available classes in classifier: {clf.classes_}", color=bold_color.BLUE)
    print_color(f"Probability matrix shape: {y_pred_proba_new_data.shape}", color=bold_color.BLUE)
    print_color(f"Original probability range: {np.min(max_probabilities):.3f} to {np.max(max_probabilities):.3f}", color=bold_color.BLUE)
    print_color(f"Normalized probability range: {np.min(y_pred_proba_new_data_normalized):.3f} to {np.max(y_pred_proba_new_data_normalized):.3f}", color=bold_color.BLUE)
    
    # Get the maximum normalized probability for each sample
    max_normalized_probabilities = np.max(y_pred_proba_new_data_normalized, axis=1)
    
    # Replace predictions below threshold with '-1' using normalized probabilities
    y_pred_new_data[max_normalized_probabilities <= probability_threshold] = '-1'

    print_color("\nPredicted Labels Distribution:", color=bold_color.BLUE)
    unique_labels, counts = np.unique(y_pred_new_data, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print_color(f"Label {label}: {count}", color=bold_color.BLUE)

    # Create DataFrame with both original and normalized probabilities using classifier's classes
    proba_columns = [f'probability_{label}' for label in clf.classes_]
    proba_columns_normalized = [f'normalized_probability_{label}' for label in clf.classes_]
    
    probabilities_df = pd.DataFrame(y_pred_proba_new_data, columns=proba_columns)
    probabilities_df_normalized = pd.DataFrame(y_pred_proba_new_data_normalized, columns=proba_columns_normalized)

    # Add max probability columns for both original and normalized
    probabilities_df['max_probability'] = max_probabilities
    probabilities_df_normalized['max_normalized_probability'] = max_normalized_probabilities

    if not only_predict_unlabeled:
        # Combine all information
        y_pred_new_data_with_info = pd.concat([
            info.reset_index(drop=True), 
            pd.DataFrame(y_pred_new_data, columns=['predicted_label']).reset_index(drop=True),
            probabilities_df.reset_index(drop=True),
            probabilities_df_normalized.reset_index(drop=True)
        ], axis=1)
    else:
        # Create a DataFrame for the new predictions
        predictions_df = pd.DataFrame(y_pred_new_data, columns=['predicted_label']).reset_index(drop=True)
        
        # Get the original feature matrix with all data
        feature_matrix_predict_original = feature_matrix_predict_original.copy()
        
        # Create a mask for unlabeled data in the original feature matrix
        unlabeled_mask = feature_matrix_predict_original[label_column].astype(str) == '-1'
        
        # Create a DataFrame with all predictions initialized as '-1'
        all_predictions = pd.Series('-1', index=feature_matrix_predict_original.index)
        
        # For unlabeled data, use the predictions
        # For labeled data, use the original labels
        all_predictions[unlabeled_mask] = predictions_df['predicted_label'].values
        all_predictions[~unlabeled_mask] = feature_matrix_predict_original.loc[~unlabeled_mask, label_column].values
        
        # Ensure 'original_label' is included in the DataFrame
        if 'original_label' not in feature_matrix_predict_original.columns:
            feature_matrix_predict_original['original_label'] = feature_matrix_predict_original[label_column].astype(str)

        # Add the predictions to the original feature matrix
        feature_matrix_predict_original['predicted_label'] = all_predictions
        
        # Combine the original feature matrix with probabilities
        y_pred_new_data_with_info = pd.concat([
            feature_matrix_predict_original,
            probabilities_df_normalized.reset_index(drop=True)
        ], axis=1)

    # Save detailed predictions with probabilities
    output_dir = os.path.dirname(feature_matrix_predict_file) if isinstance(feature_matrix_predict_file, str) else '.'
    output_dir = os.path.join(output_dir, 'predictions_results')
    os.makedirs(output_dir, exist_ok=True)
    detailed_predictions = os.path.join(output_dir, 'detailed_predictions.csv')
    data_with_info_and_labels_only = y_pred_new_data_with_info[['ppid', 'session', 'trial', 'time', 'original_label', 'predicted_label']]
    data_with_info_and_labels_only.to_csv(detailed_predictions, index=False)
    
    # Print summary of predictions
    total_predictions = len(y_pred_new_data)
    below_threshold = np.sum(max_normalized_probabilities <= probability_threshold)
    print_color(f"\nPrediction Summary:", color=bold_color.BLUE)
    print_color(f"Total predictions: {total_predictions}", color=bold_color.BLUE)
    print_color(f"Predictions below threshold ({probability_threshold}): {below_threshold} ({below_threshold/total_predictions*100:.1f}%)", color=bold_color.BLUE)
    print_color(f"Predictions above threshold: {total_predictions - below_threshold} ({(total_predictions-below_threshold)/total_predictions*100:.1f}%)", color=bold_color.BLUE)
    print_color(f"\nDetailed predictions with probabilities saved to: {detailed_predictions}", color=bold_color.BLUE)

    clear_gpu_memory()

    return y_pred_new_data_with_info


@tictoc
def main(args):
    mode = "TRAINING" if args.train or args.train_and_predict else "PREDICTING"
    print_color(f"\nRandom Forest Classifier Mode: {mode}\n", color=bold_color.GREEN)
    
    classifier_output = None
    y_new_data = None
    
    # Train the classifier if requested
    if args.train or args.train_and_predict:
        classifier_output, label_encoder_output = train_classifier(
            args.feature_matrix_train,
            args.labels_column,
            other_file_path_train=args.labels_file,
            ignore_unlabeled=args.ignore_unlabeled_train
        )
        if args.train_and_predict:
            args.classifier_file = classifier_output
            args.label_encoder_file = label_encoder_output

    # Predict using the classifier if requested
    if args.predict or args.train_and_predict:
        # y_new_data, y_new_data_with_info = predict_with_classifier(
        predict_with_classifier(
            args.classifier_file,
            args.label_encoder_file,
            args.feature_matrix_predict,
            probability_threshold=args.probability_threshold,
            only_predict_unlabeled=args.only_predict_unlabeled_predict,
            label_column=args.labels_column
        )
        

    return None

if __name__ == "__main__":
    os.system("cls||clear")
    print_welcome_message()
    args = parse_arguments()
    main(args)