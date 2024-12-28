import os
from os.path import join as pjoin
import scipy.io as sio
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from scipy.linalg import norm
from scipy import signal as sig
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt

def bandpass(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to each channel independently (data is Samples x Channels)
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[1]):  # Iterate over each channel
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered_data


# Frobenius Norm calculation
def frobenius_norm(matrix1, matrix2):
    return norm(matrix1 - matrix2, 'fro')

def compute_graph_distance(matrix1, matrix2):
    return np.sum(np.abs(matrix1 - matrix2))

# Covariance calculation
def compute_covariance(eeg_data):
    return np.cov(eeg_data.T)

# Pearson correlation coefficient matrix calculation
def compute_pearson_matrix(eeg_data):
    num_channels = eeg_data.shape[1]
    pearson_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                pearson_matrix[i, j] = pearsonr(eeg_data[:, i], eeg_data[:, j])[0]
    return pearson_matrix

# PLV calculation
def plvfcn(eegData):
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

# Graph creation
def create_graphs(matrix, threshold=0.0):
    graphs = []
    for i in range(matrix.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(matrix.shape[0]))
        for u in range(matrix.shape[0]):
            for v in range(matrix.shape[0]):
                if u != v and matrix[u, v, i] > threshold:
                    G.add_edge(u, v, weight=matrix[u, v, i])
        graphs.append(G)
    return graphs

def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val == 0:  # Avoid division by zero
        return np.zeros_like(matrix)  # Return a matrix of zeros
    return (matrix - min_val) / (max_val - min_val)

def compute_metrics(graphs):
    fn_values, gd_values = [], []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            adj1 = nx.to_numpy_array(graphs[i], weight='weight')
            adj2 = nx.to_numpy_array(graphs[j], weight='weight')
            fn_values.append(frobenius_norm(adj1, adj2))
            gd_values.append(compute_graph_distance(adj1, adj2))
    return np.mean(fn_values), np.mean(gd_values)

# Prepare Data
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Data'

# Define the subject numbers as the range 1 to 9
subject_numbers = list(range(1, 10))

# Initialize storage for subject metrics
metrics = {
    "PLV_Left_FN": [], "PLV_Left_GD": [],
    "PLV_Right_FN": [], "PLV_Right_GD": [],
    "Cov_Left_FN": [], "Cov_Left_GD": [],
    "Cov_Right_FN": [], "Cov_Right_GD": [],
    "Pearson_Left_FN": [], "Pearson_Left_GD": [],
    "Pearson_Right_FN": [], "Pearson_Right_GD": []
}

# Dictionary to hold the loaded data for each subject
subject_data = {}

# Efficiently load data for all subjects with a progress bar
for subject_number in tqdm(subject_numbers, desc="Processing Subjects"):
    mat_fname = pjoin(data_dir, f'A{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents['subject'][0, 0]
    subject_data[f'S{subject_number}'] = mat_contents['subject']
    S = subject_data[f'S{subject_number}']
    S = S[9,:].reshape(1, -1)
    fs =250
    
    # Bandpass filter for 'L' and 'R' data
    for key in ['L', 'R']:
        for trial_idx in range(S[key].shape[0]):
            S[key][0,trial_idx] = bandpass(S[key][0,trial_idx], 8, 30, fs)

    # Initialize containers for matrices
    plv_matrices, covariance_matrices, pearson_matrices = [], [], []

    # Loop through trials for the 'L' and 'R' data
    for key in ['L', 'R']:
        for trial_idx in range(S[key].shape[1]):
            eeg_data = S[key][0,trial_idx]
            plv_matrix = plvfcn(eeg_data)
            covariance_matrix = compute_covariance(eeg_data)
            pearson_matrix = compute_pearson_matrix(eeg_data)

            # Normalize matrices
            covariance_matrix = normalize_matrix(covariance_matrix)
            pearson_matrix = normalize_matrix(pearson_matrix)

            # Store matrices
            plv_matrices.append(plv_matrix)
            covariance_matrices.append(covariance_matrix)
            pearson_matrices.append(pearson_matrix)

    # Convert to numpy arrays for graph creation
    plv_matrices = np.stack(plv_matrices, axis=2)
    covariance_matrices = np.stack(covariance_matrices, axis=2)
    pearson_matrices = np.stack(pearson_matrices, axis=2)

    # Create graphs for PLV, Covariance, and Pearson
    plv_graphs = create_graphs(plv_matrices)
    covariance_graphs = create_graphs(covariance_matrices)
    pearson_graphs = create_graphs(pearson_matrices)

    # Split into Left and Right classes
    trials = len(plv_graphs) // 2
    plv_left, plv_right = plv_graphs[:trials], plv_graphs[trials:]
    cov_left, cov_right = covariance_graphs[:trials], covariance_graphs[trials:]
    pearson_left, pearson_right = pearson_graphs[:trials], pearson_graphs[trials:]

    # Metrics for PLV, Covariance, and Pearson
    plv_left_fn, plv_left_gd = compute_metrics(plv_left)
    plv_right_fn, plv_right_gd = compute_metrics(plv_right)
    cov_left_fn, cov_left_gd = compute_metrics(cov_left)
    cov_right_fn, cov_right_gd = compute_metrics(cov_right)
    pearson_left_fn, pearson_left_gd = compute_metrics(pearson_left)
    pearson_right_fn, pearson_right_gd = compute_metrics(pearson_right)

    # Store results
    metrics["PLV_Left_FN"].append(plv_left_fn)
    metrics["PLV_Left_GD"].append(plv_left_gd)
    metrics["PLV_Right_FN"].append(plv_right_fn)
    metrics["PLV_Right_GD"].append(plv_right_gd)
    metrics["Cov_Left_FN"].append(cov_left_fn)
    metrics["Cov_Left_GD"].append(cov_left_gd)
    metrics["Cov_Right_FN"].append(cov_right_fn)
    metrics["Cov_Right_GD"].append(cov_right_gd)
    metrics["Pearson_Left_FN"].append(pearson_left_fn)
    metrics["Pearson_Left_GD"].append(pearson_left_gd)
    metrics["Pearson_Right_FN"].append(pearson_right_fn)
    metrics["Pearson_Right_GD"].append(pearson_right_gd)

# Compute averages and standard deviations across subjects
averages = {key: np.mean(values) for key, values in metrics.items()}
std_devs = {key: np.std(values) for key, values in metrics.items()}

# Print averages and standard deviations
print("\nAggregated Results Across Subjects:")
for key in metrics.keys():
    print(f"{key.replace('_', ' ')}: {averages[key]:.4f} ± {std_devs[key]:.4f}")
