"""
Created on Tuesday 30th July 2024 at 12:22pm
Rishan Patel, Bioelectronics and Aspire Create Group, UCL


https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""

import os
import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.fftpack import fft, ifft
import mne
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score

#%%%%%%%%%%%% ALS Dataset - CSP %%%%%%%%%%%%%%%%

# Define the channels
channels = np.array([
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
    'T7', 'C3', 'CZ', 'C4', 'T8', 
    'P7', 'P3', 'PZ', 'P4', 'P8', 
    'O1', 'O2'
])

# Define the directory containing your .mat files
directory = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Create a dictionary to store the concatenated data and labels by subject
data_by_subject = {}

# Define the list of subject IDs you're interested in
subject_ids = [1, 2, 5, 9, 21, 31, 34, 39]

# Function to remove the last entry in the list if it's all zeros
def remove_last_entry_if_all_zeros(data_list):
    if len(data_list) > 0:
        if np.all(data_list[-1] == 0):
            return data_list[:-1]
    return data_list

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .mat file
    if filename.endswith('.mat'):
        # Extract the numeric part from the filename assuming format 'OGFS<number>.mat'
        subject_number_str = filename[len('OGFS'):-len('.mat')]
        
        # Check if the subject number is one you're interested in
        if subject_number_str.isdigit() and int(subject_number_str) in subject_ids:
            # Generate the full path to the file
            file_path = os.path.join(directory, filename)
            
            # Load the .mat file
            mat_data = loadmat(file_path)
            
            # Extract the variable name (usually 'SubjectX')
            subject_variable_name = f'Subject{subject_number_str}'
            
            # Check if the expected variable exists in the .mat file
            if subject_variable_name in mat_data:
                # Access the 1xN array of void192
                void_array = mat_data[subject_variable_name]
                
                # Initialize the subject entry in the dictionary if not already present
                subject_id = f'S{subject_number_str}'
                if subject_id not in data_by_subject:
                    data_by_subject[subject_id] = {
                        'L': [],
                        'R': [],
                    }
                
                # Loop through each element in the void array
                for item in void_array[0]:
                    # Extract the 'L', 'R', and 'Re' fields
                    L_data = item['L']
                    R_data = item['R']
                    
                    # Append data to the respective lists
                    data_by_subject[subject_id]['L'].append(L_data)
                    data_by_subject[subject_id]['R'].append(R_data)
                
                # Clean up the lists by removing the last entry if it is full of zeros
                data_by_subject[subject_id]['L'] = remove_last_entry_if_all_zeros(data_by_subject[subject_id]['L'])
                data_by_subject[subject_id]['R'] = remove_last_entry_if_all_zeros(data_by_subject[subject_id]['R'])

# Clean up unnecessary variables
del directory, file_path, filename, item, mat_data, subject_id, subject_ids, subject_number_str, subject_variable_name, void_array

#%%

fs= 256

# Bandpass filter function
def bandpass_filter_trials(data_split, low_freq, high_freq, sfreq):
    filtered_data_split = {}

    # Design the bandpass filter
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')

    for subject in data_split:
        subject_data = data_split[subject]
        filtered_subject_data = {}

        # Apply filter to each direction: 'L', 'R', and 'Re'
        for direction in ['L', 'R']:
            trials = subject_data[direction]  # List of trials, each trial is an array (Samples, 22 Channels)
            filtered_trials = []

            for trial_data in trials:
                # Remove the last 3 channels, reducing it to (Samples, 19 Channels)
                trial_data = trial_data[:, :19]

                # Apply bandpass filter to each channel
                filtered_trial_data = np.zeros_like(trial_data)
                for ch in range(trial_data.shape[1]):  # Loop over channels (now 19 channels)
                    filtered_trial_data[:, ch] = filtfilt(b, a, trial_data[:, ch])

                filtered_trials.append(filtered_trial_data)

            # Keep the list of filtered trials as a list instead of converting to a NumPy array
            filtered_subject_data[direction] = filtered_trials

        filtered_data_split[subject] = filtered_subject_data

    return filtered_data_split

# Apply bandpass filter to the data_split
filtered_data_split = bandpass_filter_trials(data_by_subject, low_freq=8, high_freq=30, sfreq=fs)

# Optionally delete the original data to free up memory
del data_by_subject

#%%

def zero_pad_trials(trials, max_length):
    """
    Zero pad each trial in the trials list to match the max_length.
    
    Parameters:
    - trials: List of numpy arrays representing trials with shape (time_samples, 19)
    - max_length: Integer representing the maximum number of time samples

    Returns:
    - padded_trials: numpy array of shape (max_length, 19, num_trials)
    """
    num_trials = len(trials)
    padded_trials = np.zeros((max_length, 19, num_trials))
    
    for i, trial in enumerate(trials):
        trial_length = trial.shape[0]
        padded_trials[:trial_length, :, i] = trial  # Zero pad shorter trials
        
    return padded_trials

def process_subject_data(filtered_data_split):
    """
    Processes the filtered_data_split dictionary to create zero-padded arrays
    for each subject (S1 to S39), using the maximum trial length between 'L' and 'R' trials
    for zero-padding.
    
    Parameters:
    - filtered_data_split: dictionary containing subject data for S1-S39, where
      each subject has 'L' and 'R' keys with lists of numpy arrays (trials).
      
    Returns:
    - subject_arrays: Dictionary with subject IDs as keys and a dictionary
      containing zero-padded arrays for 'L' and 'R' trials.
    """
    subject_arrays = {}

    # Loop over all subjects (S1 to S39)
    for subject_id in range(1, 40):
        subject_key = f"S{subject_id}"
        
        # Check if the subject exists in the data
        if subject_key in filtered_data_split:
            subject_data = filtered_data_split[subject_key]
            
            # Get maximum trial length between 'L' and 'R' trials
            max_L_trial_length = max((trial.shape[0] for trial in subject_data.get('L', [])), default=0)
            max_R_trial_length = max((trial.shape[0] for trial in subject_data.get('R', [])), default=0)
            
            max_trial_length = max(max_L_trial_length, max_R_trial_length)
            
            # Process 'L' trials with zero-padding
            if 'L' in subject_data:
                L_trials = subject_data['L']
                padded_L_trials = zero_pad_trials(L_trials, max_trial_length)
            else:
                padded_L_trials = np.array([])  # Empty if no 'L' trials
            
            # Process 'R' trials with zero-padding
            if 'R' in subject_data:
                R_trials = subject_data['R']
                padded_R_trials = zero_pad_trials(R_trials, max_trial_length)
            else:
                padded_R_trials = np.array([])  # Empty if no 'R' trials
            
            # Store the results in the subject_arrays dictionary
            subject_arrays[subject_key] = {
                'L': padded_L_trials,
                'R': padded_R_trials
            }
    
    return subject_arrays


data = process_subject_data(filtered_data_split)



#%%

from sklearn.model_selection import KFold


def combine_L_R_data(data):
    """Combine L and R trials by alternating between them, reshape, and create label vectors."""
    # Iterate over the subjects in the data dictionary
    for subject in data.keys():
        # Ensure the subject contains both 'L' and 'R' keys
        if 'L' in data[subject] and 'R' in data[subject]:
            # Get the data for L and R
            L_data = data[subject]['L']
            R_data = data[subject]['R']

            # Ensure that L and R have the same number of trials
            num_L_trials = L_data.shape[2]
            num_R_trials = R_data.shape[2]
            min_trials = min(num_L_trials, num_R_trials)  # Find the minimum number of trials

            # Interleave the L and R trials to ensure an equal number of both
            combined_data = []
            label_vector = []

            for i in range(min_trials):
                combined_data.append(L_data[:, :, i])  # Add one L trial
                combined_data.append(R_data[:, :, i])  # Add one R trial
                label_vector.append(0)  # Label for L trial
                label_vector.append(1)  # Label for R trial

            # Convert combined_data and label_vector to numpy arrays
            combined_data = np.array(combined_data)  # Shape: (2 * min_trials, channels, samples)
            label_vector = np.array(label_vector)  # Shape: (2 * min_trials,)

            # Reshape combined_data from (trials, channels, samples)
            combined_data = np.transpose(combined_data, (0, 2, 1))  

            # Store the reshaped combined data and labels in the data dictionary
            data[subject] = {
                'data': combined_data,
                'label': label_vector
            }

    return data

# Combine L and R data
data = combine_L_R_data(data)


#%%

def train_and_test_csp(train_data, test_data, n_components=10, plot_csp=False):
    """Train CSP on train_data and test on test_data."""
    csp = CSP(n_components=n_components, log=True, norm_trace=True, component_order='mutual_info')
    
    # Prepare data (assuming data is already in [trials, channels, samples] format)
    X_train = np.array(train_data['data'])  # Shape: [trials, channels, samples]
    y_train = np.array(train_data['labels'])  # Shape: [trials]
    
    X_test = np.array(test_data['data'])  # Shape: [trials, channels, samples]
    y_test = np.array(test_data['labels'])  # Shape: [trials]
    
    # Check the shapes to ensure they're correct
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # Fit CSP on training data (expects [n_trials, n_channels, n_samples])
    csp.fit(X_train, y_train)
    
    # Transform both train and test data
    X_train_transformed = csp.transform(X_train)
    X_test_transformed = csp.transform(X_test)
    
    # Standardize data
    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train_transformed)
    X_test_transformed = scaler.transform(X_test_transformed)
    
    # Train a classifier (SVM) with balanced class weights
    classifier = SVC(class_weight='balanced')
    classifier.fit(X_train_transformed, y_train)
    
    # Test the classifier
    y_pred = classifier.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


# Perform 10-fold cross-validation automatically
def cross_validate(data, n_folds=10):
    """Perform 10-fold cross-validation on the data."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)  # 10-fold CV
    accuracies = {}

    for subject, subject_data in data.items():
        X = subject_data['data'] # trials first for cross-validation
        y = subject_data['label']
        
        accuracies[subject] = []
        
        # KFold cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Wrap the train/test data in a dictionary format
            train_data = {'data': X_train, 'labels': y_train}
            test_data = {'data': X_test, 'labels': y_test}
            
            # Train and test CSP
            accuracy = train_and_test_csp(train_data, test_data)
            accuracies[subject].append(accuracy)
        
        # Calculate mean accuracy for the subject
        mean_accuracy = np.mean(accuracies[subject])
        print(f'Subject: {subject}, Mean Accuracy: {mean_accuracy}')
    
    return accuracies

# Perform cross-validation and get accuracies
accuracies = cross_validate(data, n_folds=10)

#%% Plots



def compute_mean_accuracies(accuracies):
    subject_stats = {}
    
    # Calculate mean, min, and max accuracy for each subject
    for subject, acc_list in accuracies.items():
        subject_mean = np.mean(acc_list)
        subject_min = np.min(acc_list)
        subject_max = np.max(acc_list)
        subject_stats[subject] = {
            'mean': subject_mean,
            'min': subject_min,
            'max': subject_max
        }
        print(f"Subject: {subject}")
        print(f"  Mean accuracy: {subject_mean:.4f}")
        print(f"  Min accuracy: {subject_min:.4f}")
        print(f"  Max accuracy: {subject_max:.4f}")
    
    # Calculate the mean of min and max values across all subjects
    min_values = [stats['min'] for stats in subject_stats.values()]
    max_values = [stats['max'] for stats in subject_stats.values()]
    
    mean_min = np.mean(min_values)
    mean_max = np.mean(max_values)
    
    # Calculate the overall cohort mean by averaging subject means
    cohort_mean = np.mean([stats['mean'] for stats in subject_stats.values()])
    print(f"\nCohort mean accuracy: {cohort_mean:.4f}")
    print(f"Cohort mean of min accuracies: {mean_min:.4f}")
    print(f"Cohort mean of max accuracies: {mean_max:.4f}")
    
    return subject_stats, cohort_mean, mean_min, mean_max

# Example: computing the mean accuracies, min, and max
subject_stats, cohort_mean, mean_min, mean_max = compute_mean_accuracies(accuracies)


