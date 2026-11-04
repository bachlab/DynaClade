import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from utils import print_welcome_message, tictoc, ask_question_options
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.patches import ConnectionPatch
import pdb
from joblib import dump


@tictoc
def main():
    print_welcome_message()

    # Load the CSV file
    data_file = input("\nEnter the path to the CSV file with the labels: ")
    label_column = input("\nEnter the name of the column with labels: ")
    filter_consecutive_labels = ask_question_options("Do you want to filter out actions with consecutive observations less than a threshold?", ["y", "n"]) == "y"
    if filter_consecutive_labels:
        action_threshold = int(input("\nEnter the minimum number of consecutive action observations to be considered in the sequence: "))
    output_dir = os.path.dirname(data_file)

    data = pd.read_csv(data_file)

    # Filter out rows with -1 or '-1' in the label column
    data = data[~data[label_column].isin([-1, '-1'])]

    # Encode the labels
    le = LabelEncoder()
    data[label_column] = le.fit_transform(data[label_column])
    # After fitting the LabelEncoder

    # Save the LabelEncoder to a file
    le_filename = os.path.join(output_dir, 'label_encoder_hmm.joblib')
    dump(le, le_filename)

    # Group the data by 'ppid', 'session', and 'trial'
    groups = data.groupby(['ppid', 'session', 'trial'])

    # Initialize a list to store the observations
    observations = []

    # Define a function to filter groups based on their size
    def filter_func(x):
        return x[label_column].value_counts().min() >= action_threshold

    # Filter out labels that appear less than n consecutive times

    # For each group
    for name, group in groups:
        # Remove consecutive rows with the same label within the group
        if filter_consecutive_labels:
            group = group.groupby((group[label_column] != group[label_column].shift()).cumsum()).filter(filter_func)
        else:
            group = group[group[label_column] != group[label_column].shift()]

        # Add the sequence of states from the group to the observations
        observations.append(np.array(group[label_column]).reshape(-1, 1))


    # Determine the number of unique states
    n_states = data[label_column].nunique()

    # Concatenate all observations into a single array
    all_observations = np.concatenate(observations)

    # Create a list of the lengths of the observations
    lengths = [len(obs) for obs in observations]

    # Train a single Markov model on all observations
    model = hmm.MultinomialHMM(n_components=n_states)
    model.fit(all_observations, lengths=lengths)

    # Save the model to a file
    model_filename = os.path.join(output_dir, 'hmm_model.joblib')
    dump(model, model_filename)

    # Set the diagonal of the transition matrix to 0
    np.fill_diagonal(model.transmat_, 0)

    # Normalize the transition matrix so that each row sums to 1
    model.transmat_ /= model.transmat_.sum(axis=1, keepdims=True)

    # Initialize a list to store the output
    output = []

    # For each observation
    for i, observation in enumerate(observations):
        # Get the log probability under the model
        log_prob = model.score(observation)

        # Get the most likely state sequence given observed sequence
        state_sequence = model.predict(observation)

        # Decode the state sequence back to the original string names
        state_sequence = le.inverse_transform(state_sequence)

        # Add the output to the list
        output.append([log_prob, state_sequence])

    # Create a list of state names
    state_names = le.classes_

    # Create a DataFrame from the state transition matrix
    transition_matrix_df = pd.DataFrame(model.transmat_, columns=state_names, index=state_names)

    # Save the DataFrame to a CSV file
    transition_matrix_filename = os.path.join(output_dir, 'transition_matrix.csv')
    transition_matrix_df.to_csv(transition_matrix_filename)

    # Save the model to a file
    model_filename = os.path.join(output_dir, 'hmm_model.joblib')
    dump(model, model_filename)

    # Initialize a DataFrame to store the predictions
    predictions_df = pd.DataFrame(columns=['initial_state'] + [f'S{i}' for i in range(1, 11)])

    # For each unique state
    for state in range(n_states):
        # Initialize the sequence with the current state
        sequence = [state]

        # Predict the next 10 states
        for _ in range(10):
            # Compute the probabilities of the next state given the current state
            next_state_probs = model.transmat_[sequence[-1]]

            # Choose the next state with the highest probability
            next_state = np.argmax(next_state_probs)

            # Add the next state to the sequence
            sequence.append(next_state)

        # Decode the sequence back to the original string names
        sequence = le.inverse_transform(sequence)

        # Add the sequence to the DataFrame
        predictions_df.loc[len(predictions_df)] = sequence

    # Save the DataFrame to a CSV file
    predictions_filename = os.path.join(output_dir, 'hmm_predictions.csv')
    predictions_df.to_csv(predictions_filename, index=False)


    # Create a new figure with a specified size
    plt.figure(figsize=(15, 15))

    # Plot the Markov model
    G = nx.DiGraph()

    # Create a list of edges from the transition matrix
    edges = [(i, j, model.transmat_[i, j]) for i in range(model.n_components) for j in range(model.n_components) if model.transmat_[i, j] > 0]

    # Add the edges to the graph
    G.add_weighted_edges_from(edges)

    # Sort the edges by weight in descending order
    edges.sort(key=lambda x: x[2], reverse=True)

    # Draw the graph with state names as labels
    pos = nx.spring_layout(G, k=0.15)  # use spring layout with higher k value

    # Draw the nodes only
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)

    # Draw the labels
    label_dict = dict(zip(range(len(state_names)), state_names))
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=12)

    # Draw curved edges with probabilities
    for u, v, d in G.edges(data=True):
        ax = plt.gca()
        xy_from = pos[u]
        xy_to = pos[v]
        arrowprops=dict(arrowstyle="-|>", color='grey', lw=1.5)  # set arrow color to grey

        # Check if the transition is one-way or two-way
        if G.has_edge(v, u):
            # Two-way transition, draw two separate curved arrows
            rad = 0.3  # adjust this value to make the arrows curve more or less
            connectionstyle = f'arc3,rad={rad}'
        else:
            # One-way transition, draw a straight line
            rad = 0
            connectionstyle = 'arc3'

        # Calculate the direction of the arrow
        direction = np.array(xy_to) - np.array(xy_from)
        direction = direction / np.linalg.norm(direction)

        # Move the start and end points of the arrow along the direction by the radius of the nodes
        node_radius = 0.05  # increase this value to avoid overlapping of arrows with nodes
        xy_from = xy_from + direction * node_radius
        xy_to = xy_to - direction * node_radius

        ax.annotate('', xy=xy_to, xytext=xy_from, arrowprops=dict(arrowstyle='-|>', color='grey', lw=1.5, connectionstyle=connectionstyle))

        # Calculate a midpoint for the label that takes into account the curvature of the arrow
        mid_x = (xy_from[0] + xy_to[0]) / 2
        mid_y = (xy_from[1] + xy_to[1]) / 2

        # Calculate the perpendicular direction
        perp_direction = np.array([-direction[1], direction[0]])

        # Move the midpoint along the perpendicular direction by a distance proportional to the curvature of the arrow
        mid_x = mid_x + rad * perp_direction[0]
        mid_y = mid_y + rad * perp_direction[1]

        ax.text(mid_x, mid_y, f'{d["weight"]:.2f}', color='black', ha='center', va='center')

    # Remove the figure borders
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save the plot as a PNG file
    output_image_filename = os.path.join(output_dir, 'model.png')
    plt.savefig(output_image_filename)

    # Compute the transition matrix
    transition_matrix = model.transmat_

    # Create a figure with two subplots
    plt.figure(figsize=(15, 15))

    # Create a heatmap for the first half of the transition matrix
    sns.heatmap(transition_matrix, annot=False, fmt='.2f', cmap='plasma', xticklabels=state_names, yticklabels=state_names)
    plt.title('Transition matrix of state transitions (First Half)', fontsize=24)
    plt.xlabel('Next state', fontsize=20)
    plt.ylabel('Current state', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()

    # Save the figure in a vector format
    plt.savefig(os.path.join(output_dir, 'transition_matrix.png'))


if __name__ == "__main__":
    os.system("cls||clear")
    main()