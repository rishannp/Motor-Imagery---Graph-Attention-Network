import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from os.path import join as pjoin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import mne
from mne import io
import os
import scipy.signal as sig
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP


# EEGNet-specific imports
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

#%%
# Directory containing the GDF files
data_dir = r'C:\Users\uceerjp\Desktop\PhD\IV2a\BCICIV_2a_gdf\Data'

# List of subject IDs
subjects = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# Event IDs for the four classes (from the BCI Competition IV 2a description)
event_id = {
    'left_hand': 7,   # LH
    'right_hand': 8,  # RH
}

# Reverse mapping from event IDs to labels
id_to_label = {v: k for k, v in event_id.items()}

# Sampling frequency after resampling
original_sfreq = 250  # Hz
new_sfreq = 128       # Hz

# Time window (2 seconds = 500 samples)
tmin, tmax = 0, 2  # seconds

# Dictionary to store data for each subject
subject_data = {}

# Loop through each subject
for subject in subjects:
    # Load both the training and testing GDF files
    train_file = os.path.join(data_dir, f'A0{subject}E.gdf')
    test_file = os.path.join(data_dir, f'A0{subject}T.gdf')
    
    # Load GDF files using MNE
    raw_train = mne.io.read_raw_gdf(train_file, preload=True)
    raw_test = mne.io.read_raw_gdf(test_file, preload=True)
    
    # Concatenate the raw data
    raw = mne.concatenate_raws([raw_train, raw_test])
    
    # Apply bandpass filtering between 8-30 Hz
    raw.filter(l_freq=8, h_freq=30, fir_design='firwin', skip_by_annotation='edge')
    
    # Perform EOG regression
    eog_channels = mne.pick_types(raw.info, meg=False, eog=True)
    if eog_channels:
        raw = raw.copy().set_eog_proj(True)
        raw.apply_proj()
    
    # Remove the last three channels (assumed EOG channels)
    all_channel_names = raw.info['ch_names']
    eog_channel_names = all_channel_names[-3:]  # Get the names of the last three channels
    raw.drop_channels(eog_channel_names)
    
    # Downsample to 128 Hz
    raw.resample(new_sfreq, npad='auto')
    
    # Get events and annotations
    events, _ = mne.events_from_annotations(raw)
    print(f'Event IDs for subject S{subject}:', event_id)
    print(f'First few events for subject S{subject}:', events[:10])
    
    # Segment the data for each class
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                        baseline=None, preload=True, event_repeated='drop')
    
    # Extract the data as a numpy array
    data = epochs.get_data()
    
    # Create label vector
    labels = np.array([id_to_label.get(event[2], 'unknown') for event in events])
    labels = np.array([event_id[label] for label in labels if label != 'unknown'])
    
    # Store the data and labels in the dictionary with the key 'S1', 'S2', etc.
    subject_data[f'S{subject}'] = {'data': data, 'labels': labels}
    
    print(f'Stored data for subject S{subject} with shape {data.shape} and labels shape {labels.shape}')

# Now, subject_data['S1'], subject_data['S2'], ..., subject_data['S9'] will give you the data and labels

#%%

# Define the subject numbers
subject_numbers = list(range(1, 10))  # This will generate [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Initialize dictionary to store best accuracies
accuracies = {}

# Define k-fold cross-validation
k = 10  # Number of folds
kf = KFold(n_splits=k, shuffle=False)

# Iterate over each subject
for subject_number in subject_numbers:
    S1 = subject_data[f'S{subject_number}']
    data = S1['data']
    labels = S1['labels']

    # No need to reshape data for CSP, but ensure labels are correctly formatted
    labels = labels.flatten()  # Ensure labels are 1D for scikit-learn
    
    # Initialize list to store accuracies for each fold
    fold_accuracies = []
    
    # Perform k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        train_data, val_data = data[train_index], data[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]
    
        # Create a CSP pipeline with LDA classifier
        csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
        clf = make_pipeline(csp, LDA())
        
        # Train the pipeline (CSP + LDA)
        clf.fit(train_data, train_labels)
        
        # Evaluate the classifier on the validation set
        val_score = clf.score(val_data, val_labels)
        
        print(f"Subject: S{subject_number}, Classification accuracy for fold {fold + 1}: {val_score}")
        
        # Store the fold accuracy
        fold_accuracies.append(val_score)
    
    # Store the fold accuracies for this subject
    accuracies[f'Subject_{subject_number}'] = fold_accuracies

# Save the accuracies dictionary in the current directory
def save_best_accuracies(best_accuracies, save_path='CSP_LDA_Accuracies.npy'):
    np.save(save_path, best_accuracies)

# Get the current working directory
current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'CSP_LDA_Accuracies.npy')
save_best_accuracies(accuracies, save_path=save_path)

print(f"Accuracies saved to {save_path}")

# Function to print mean, min, and max accuracies
def print_mean_min_max_accuracies(accuracies):
    """
    Prints the mean, minimum, and maximum accuracies for each subject.

    Parameters:
        accuracies (dict): Dictionary containing fold accuracies for each subject.
                           Keys are subject numbers and values are lists of fold accuracies.
    """
    # Iterate through each subject in the accuracies dictionary
    for subject_number, fold_accuracies in accuracies.items():
        # Calculate mean, min, and max of the fold accuracies
        mean_accuracy = np.mean(fold_accuracies)
        min_accuracy = np.min(fold_accuracies)
        max_accuracy = np.max(fold_accuracies)
        
        # Print the mean, min, and max accuracy for the subject
        print(f'Subject {subject_number}: Mean Accuracy = {mean_accuracy:.4f}, '
              f'Min Accuracy = {min_accuracy:.4f}, Max Accuracy = {max_accuracy:.4f}')

# Example usage:
print_mean_min_max_accuracies(accuracies)
