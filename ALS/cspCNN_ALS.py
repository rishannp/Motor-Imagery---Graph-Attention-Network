# -*- coding: utf-8 -*-
"""
Rishan Patel, UCL, Bioelectronics Group.
CSP-CNN Implementation -  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9325918
"""


import os
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy import signal
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras 
from keras import layers
import keras_cv
import math
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import butter, lfilter
import pywt
from sklearn.model_selection import KFold
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Conv3D, Conv2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from collections import Counter
from mne.decoding import CSP
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
#%% Functions

def split_and_extend_dataset(S1, chunk_size=256, fs=256, duration=5):
    """
    Splits the trials in S1 into chunks of size `chunk_size` using only the middle `duration` seconds of each trial 
    and returns the extended dataset.

    Parameters:
    - S1: A dictionary containing the EEG data for classes 'L' and 'R'. Each entry is an array of trials.
    - chunk_size: The size of each chunk along the time dimension (default is 256).
    - fs: Sampling frequency (default is 256 Hz).
    - duration: The duration in seconds to retain from the middle of each trial (default is 5 seconds).

    Returns:
    - extended_dataset: A dictionary with keys 'L' and 'R', containing the extended datasets for both classes.
                        Each trial will have its own list of chunks.
    """
    # Calculate the number of data points to keep based on the duration and sampling frequency
    points_to_keep = fs * duration
    
    # Initialize empty lists to store the extended dataset for 'L' and 'R'
    extended_dataset_L = []
    extended_dataset_R = []

    # Iterate over both classes 'L' and 'R'
    for class_label in ['L', 'R']:
        # Get the list of arrays for the current class
        class_arrays = S1[class_label]

        # Get the number of trials for the current class (assuming class_arrays is of size [1, Number_of_Trials])
        num_trials = class_arrays.shape[1]

        # Loop through each trial
        for trial_idx in range(num_trials):
            # Access the trial array (shape: (N, num_electrodes)) using the index S1['L' or 'R'][0, trial_idx]
            trial_array = class_arrays[0, trial_idx]

            # Get the shape (N, num_electrodes) of the trial array
            N, num_electrodes = trial_array.shape

            # Calculate the starting and ending indices to extract the middle `points_to_keep` data points
            start_idx = max(0, (N - points_to_keep) // 2)
            end_idx = start_idx + points_to_keep
            trial_array = trial_array[start_idx:end_idx, :]

            # Initialize a list to store the chunks for the current trial
            trial_chunks = []

            # Split trial_array into chunks of size `chunk_size` along the time dimension (axis 0)
            num_chunks = trial_array.shape[0] // chunk_size

            # Loop through the full chunks
            for i in range(num_chunks):
                chunk = trial_array[i*chunk_size:(i+1)*chunk_size, :]
                trial_chunks.append(chunk)

            # Handle leftover data if the number of points is not divisible by chunk_size (optional)
            leftover = trial_array.shape[0] % chunk_size
            if leftover > 0:
                leftover_chunk = trial_array[-leftover:, :]
                trial_chunks.append(leftover_chunk)

            # Append the list of chunks for the current trial to the appropriate dataset list
            if class_label == 'L':
                extended_dataset_L.append(trial_chunks)
            else:
                extended_dataset_R.append(trial_chunks)

    # Combine both class datasets into one dictionary
    extended_dataset = {'L': extended_dataset_L, 'R': extended_dataset_R}

    return extended_dataset

def bandpass_filter(data, lowcut=8, highcut=30, fs=256, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def apply_bandpass_and_select_channels(subject_data):
    """
    Applies an 8-30 Hz bandpass filter to channels 8, 9, and 10 for all
    conditions ('L' and 'R') in the given subject's data. Overwrites the original
    data structure, retaining only the filtered channels.

    Parameters:
    subject_data (dict): Dictionary containing data for a single subject with
                         structure subject_data['Idx'][Trial][Epoch].
    """
    for idx in ['L', 'R']:
        trials = subject_data[idx]  # Access trials for the current index
        num_trials = len(trials)
        
        for trial in range(num_trials):
            epoch_data = trials[trial][0]  # Access the data at epoch 0
            selected_channels = epoch_data[:,:]
            
            # Overwrite the entire data with only the filtered channels
            subject_data[idx][trial][0] = selected_channels

    return subject_data

def apply_filterbank(data, fs=256):
    """
    Applies a filterbank with frequency bands from 4-8 Hz to 36-40 Hz in steps of 4 Hz across all channels
    for each trial in the given data. The output is a new array with dimensions Filterbank x Trials x Channels x Samples.

    Parameters:
    - data (np.ndarray): 3D array of shape (trials, channels, samples), representing EEG data.
    - fs (int): Sampling frequency (default is 256 Hz).

    Returns:
    - filtered_data_output (np.ndarray): 4D array of shape (Filterbank, trials, channels, samples), containing
                                         the filtered data for each frequency band in the filterbank.
    """

    # Define the filterbank frequency ranges with non-overlapping 4 Hz bands (4-8, 8-12, ..., 36-40)
    filterbank_ranges = [(low, low + 4) for low in range(4, 40, 4)]  # (4-8, 8-12, ..., 36-40)
    num_bands = len(filterbank_ranges)
    num_trials, num_channels, num_samples = data.shape

    # Initialize the output array for filterbank data
    # Shape: (Filterbank, trials, channels, samples)
    filtered_data_output = np.zeros((num_bands, num_trials, num_channels, num_samples))

    # Apply each filter in the filterbank across all trials and channels
    for band_idx, (low, high) in enumerate(filterbank_ranges):
        for trial_idx in range(num_trials):
            for channel_idx in range(num_channels):
                # Apply the bandpass filter to each channel in each trial for the current frequency range
                filtered_data_output[band_idx, trial_idx, channel_idx, :] = bandpass_filter(
                    data[trial_idx, channel_idx, :], low, high, fs
                )

    return filtered_data_output



def pad_trials_to_max_length(trials, max_length):
    """
    Pads each trial in the trials list to the max_length with zeros.
    
    Parameters:
    trials (list of np.ndarray): List of trials with varying lengths.
    max_length (int): The length to which all trials should be padded.
    
    Returns:
    np.ndarray: Array of shape (max_length, Channels, Trials) with padded trials.
    """
    padded_trials = []
    for trial in trials:
        # Get the current trial length and number of channels
        trial_length, num_channels = trial.shape
        
        # Create a zero-padded array of shape (max_length, num_channels)
        padded_trial = np.zeros((max_length, num_channels))
        
        # Copy the original trial data into the zero-padded array
        padded_trial[:trial_length, :] = trial
        padded_trials.append(padded_trial)
    
    # Stack along the third dimension (Trials) to get shape (max_length, Channels, Trials)
    return np.stack(padded_trials, axis=-1)

def concatenate_subject_data(subject_data):
    """
    Concatenates 'L' and 'R' trials for a single subject, pads to the longest trial length, 
    and formats the data as TimeStamp x Channels x Trials.

    Parameters:
    subject_data (dict): Dictionary containing data for a single subject with
                         structure subject_data['Idx'][Trial][Epoch].

    Returns:
    np.ndarray: Concatenated and padded array of shape (max_length, Channels, Trials).
    """
    all_trials = []  # Collect all trials here
    
    for idx in ['L', 'R']:
        trials = subject_data[idx]  # Access trials for the current index
        for trial in trials:
            all_trials.append(trial[0])  # Append the epoch data (TimeStamp x Channels)

    # Determine the maximum length across all trials for this subject
    max_length = max(trial.shape[0] for trial in all_trials)
    
    # Pad all trials to the maximum length and stack along the third axis (Trials)
    return pad_trials_to_max_length(all_trials, max_length)


def compute_csp_cnn_features(data, num_top_features=30, fs=256):
    """
    Computes CSP features for each filter bank, ranks and selects top spatial filters,
    and generates the final spatio-spectral feature representation.

    Parameters:
    - data (np.ndarray): Filtered EEG data of shape (Filterbank, Trials, Channels, Samples).
    - num_top_features (int): Number of top features to select based on mutual information (default is 30).
    - fs (int): Sampling frequency (default is 256 Hz).

    Returns:
    - G (np.ndarray): Spatio-spectral feature representation for CNN, shape (Trials, Top_Features, Samples).
    - labels (np.ndarray): Array of labels for each trial, with `0` for "Left" and `1` for "Right".
    """
    num_bands, num_trials, num_channels, num_samples = data.shape

    # Create labels: first half of trials labeled as 0 (Left) and second half as 1 (Right)
    labels = np.array([0] * (num_trials // 2) + [1] * (num_trials // 2))
    
    all_features = []  # List to store features from each filter band
    spatial_filters = []  # List to store spatial filters for each filter bank

    # Step 1: Compute CSP for each filter band
    for band_idx in range(num_bands):
        # Extract data for the current filter band
        band_data = data[band_idx, :, :, :]  # Shape: (Trials, Channels, Samples)

        # Initialize CSP
        csp = CSP(n_components=4, reg=None, log=True)  # Choose the number of CSP components, e.g., 4

        # Fit CSP on the data for this filter band
        band_features = csp.fit_transform(band_data, labels)  # Shape: (Trials, CSP_components)
        all_features.append(band_features)

        # Store the spatial filters for the current filter band
        spatial_filters.append(csp.filters_)  # Shape: (Channels, CSP_components)

    # Concatenate features from all filter bands to create a single feature matrix
    all_features = np.concatenate(all_features, axis=1)  # Shape: (Trials, Total_CSP_features_across_bands)

    # Standardize features to zero mean and unit variance (recommended for mutual information)
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)

    # Step 2: Select top features using mutual information
    mi_scores = mutual_info_classif(all_features, labels)
    selected_feature_indices = np.argsort(mi_scores)[-num_top_features:]  # Get indices of top features
    top_features = all_features[:, selected_feature_indices]  # Shape: (Trials, num_top_features)

    # Step 3: Generate the spatio-spectral representation G using the top spatial filters
    G = []  # To store the spatio-spectral representation
    for trial in range(num_trials):
        trial_features = []
        for idx in selected_feature_indices:
            # Determine which filter band and CSP component this index corresponds to
            band_idx = idx // 4  # Determine the band index
            filter_idx = idx % 4  # CSP component index within that band

            # Check that we have a valid band index and filter index
            if band_idx < num_bands and filter_idx < spatial_filters[band_idx].shape[1]:
                selected_filter = spatial_filters[band_idx][:, filter_idx]  # Get the filter from the correct band
                trial_data = data[band_idx, trial, :, :]  # Shape: (Channels, Samples)

                # Filter the trial data using the spatial filter
                filtered_trial_data = np.dot(selected_filter, trial_data)  # Shape: (Samples,)

                # Append the filtered signal as a feature for this trial
                trial_features.append(filtered_trial_data)
            else:
                print(f"Skipping invalid index: band_idx={band_idx}, filter_idx={filter_idx}")

        # Stack all features for this trial and add to the spatio-spectral representation
        G.append(np.stack(trial_features, axis=0))  # Shape: (num_top_features, Samples)

    # Convert G to a numpy array with shape (Trials, num_top_features, Samples)
    G = np.array(G)

    return G, labels

def add_labels_to_data(subject_data):
    """
    Adds a label vector to a subject's wavelet data.

    Parameters:
    - subject_data: List of numpy arrays, where each array corresponds to a subject's wavelet data.

    Returns:
    - subject_data: The original subject wavelet data.
    - labels: Numpy array containing the label vector for the subject.
    """
    num_trials = len(subject_data)  # Get the number of trials for the subject
    
    # Create a label vector: 1 for the first half, 2 for the second half
    label_vector = np.ones(num_trials)  # Start with all ones
    label_vector[num_trials // 2:] = 2  # Set second half to twos

    return subject_data, label_vector  # Return both the data and the label vector


def process_images_and_labels(img, labels, target_size=(64, 64), n_splits=5):
    """
    Processes color images and creates labels for k-fold cross-validation.
    
    Parameters:
    - img: List of image arrays (height x width or height x width x channels).
    - labels: List or array of labels corresponding to each image.
    - target_size: Desired output size for the images.
    - n_splits: Number of splits for K-Fold cross-validation.
    
    Returns:
    - X: Array of resized images.
    - Y: Array of one-hot encoded labels.
    - kf: KFold cross-validator object.
    """
    num_images = len(img)
    X = np.empty((num_images, target_size[0], target_size[1], 3))  # Assuming RGB color images
    Y = np.array(labels)
    
    for i, image_data in enumerate(img):
        # Ensure image has three dimensions: (height, width, channels)
        if image_data.ndim == 2:  # If grayscale (height, width), add a channel dimension
            image_data = np.expand_dims(image_data, axis=-1)
        if image_data.shape[-1] == 1:  # If single-channel, repeat to make it RGB
            image_data = np.repeat(image_data, 3, axis=-1)

        # Convert image to a Tensor, resize, and convert back to numpy
        image_tensor = tf.convert_to_tensor(image_data)
        resized_image = tf.image.resize(image_tensor, target_size).numpy().astype('uint8')
        
        X[i] = resized_image

    # Normalize pixel values
    X = X / 255.0
    
    # One-hot encode labels
    Y = tf.keras.utils.to_categorical(Y - 1, num_classes=2)  # Assuming labels are 1 and 2
    
    return X, Y

def build_model(input_shape, num_classes=2):
    model = Sequential()
    
    # Conv2D layer
    model.add(Conv2D(50, (3, 3), padding='same', input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    
    # Another Conv2D layer
    model.add(Conv2D(100, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    
    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(240, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    return model

def calculate_nscm(data):
    """
    Calculate the Normalized Sample Covariance Matrix (NSCM) for given EEG trials.
    
    Parameters:
        data (np.ndarray): 3D array of shape (trials, electrodes, samples) representing filtered EEG signals.
    
    Returns:
        G (np.ndarray): 3D array of shape (electrodes, electrodes, trials) representing the NSCM matrices for each trial.
    """
    trials, electrodes, samples = data.shape  # Unpack the shape
    G = np.zeros((electrodes, electrodes, trials))  # Initialize the 3D array for NSCMs

    # Calculate the NSCM for each trial
    for trial in range(trials):
        Zq = data[trial]  # Get the EEG data for the current trial
        
        # Calculate the mean for each electrode
        Z_bar = np.mean(Zq, axis=1)  # Mean across samples (shape: (electrodes,))
        
        # Compute the NSCM matrix for the current trial
        M = np.zeros((electrodes, electrodes))  # Initialize the NSCM matrix
        
        for e in range(electrodes):
            for t in range(electrodes):
                Zeq = Zq[e, :]  # Data for electrode e
                Zt = Zq[t, :]   # Data for electrode t
                
                Zeq_diff = Zeq - Z_bar[e]  # Difference from the mean for electrode e
                Zt_diff = Zt - Z_bar[t]     # Difference from the mean for electrode t
                
                # Calculate the normalized covariance
                numerator = np.sum(Zeq_diff * Zt_diff)
                denominator = np.sqrt(np.sum(Zeq_diff**2) * np.sum(Zt_diff**2))
                
                # Fill the NSCM matrix with the normalized covariance
                M[e, t] = numerator / denominator if denominator != 0 else 0  # Handle division by zero
        
        G[:, :, trial] = M  # Store the NSCM matrix for this trial
    
    return G

#%% Main
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Define the subject numbers
subject_numbers = [1 ,2, 5, 9, 21, 31, 34, 39]  # 

# Dictionary to hold the loaded data for each subject
subject_accuracies = {}

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    data = mat_contents[f'Subject{subject_number}']
    data = split_and_extend_dataset(data, chunk_size=256*5, fs=256, duration=5)
    data = apply_bandpass_and_select_channels(data)
    data = concatenate_subject_data(data) 
    data = np.reshape(data,(data.shape[2],22,1280)) #Output needs to be : Trials, Channels, Timesteps
    data = apply_filterbank(data, fs=256)
    features , labels = compute_csp_cnn_features(data, num_top_features=22, fs=256)
    features = calculate_nscm(features)
    features = np.reshape(features,(features.shape[2],features.shape[0],features.shape[0])) #Output needs to be : Trials, Channels, Timesteps
    del data
    
    X = features
    Y = labels
    Y = to_categorical(Y, num_classes=2)
    
    # Cross-validation setup
    input_shape = X.shape[1:]  # Input shape based on processed images
    skf = StratifiedKFold(n_splits=10)
    
    # Initialize a list to store accuracies for each fold
    fold_accuracies = []
    
    # K-Fold Cross-Validation
    for fold, (train_index, test_index) in enumerate(skf.split(X, np.argmax(Y, axis=1))):
        # Check class distribution in training and testing sets
        y_train_classes = np.argmax(Y[train_index], axis=1)
        y_test_classes = np.argmax(Y[test_index], axis=1)
        train_class_distribution = Counter(y_train_classes)
        test_class_distribution = Counter(y_test_classes)
        
        print(f"Fold {fold+1} Class Distribution:")
        print(f"  Training Set: {train_class_distribution}")
        print(f"  Testing Set: {test_class_distribution}")
        
        # Reinitialize model for each fold to prevent accumulated learning
        model = build_model(input_shape)
        
        # Split and normalize the data
        X_train, X_test = X[train_index] / 255., X[test_index] / 255.
        y_train, y_test = Y[train_index], Y[test_index]
        
        # Set up early stopping to avoid overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        
        # Train the model
        model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1, 
                  validation_data=(X_test, y_test), callbacks=[early_stopping])
        
        # Model Evaluation
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_true, y_pred)
        fold_accuracies.append(accuracy)
        
        # Print the confusion matrix and classification report for each fold
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))
    
        
    del features, X, Y
    # Calculate mean, max, and min accuracies for the subject
    mean_accuracy = np.mean(fold_accuracies)
    max_accuracy = np.max(fold_accuracies)
    min_accuracy = np.min(fold_accuracies)
    
    # Store results in the dictionary
    subject_accuracies[f'S{subject_number}'] = {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'max_accuracy': max_accuracy,
        'min_accuracy': min_accuracy
    }
    
    # Display results for each subject
    print(f"Subject {subject_number} Mean Testing Accuracy: {mean_accuracy:.4f}")
    print(f"Subject {subject_number} Max Testing Accuracy: {max_accuracy:.4f}")
    print(f"Subject {subject_number} Min Testing Accuracy: {min_accuracy:.4f}")

# Output the full accuracies for all subjects
print("\nOverall Subject Accuracies:")
for subject, metrics in subject_accuracies.items():
    print(f"{subject} - Mean: {metrics['mean_accuracy']:.4f}, Max: {metrics['max_accuracy']:.4f}, Min: {metrics['min_accuracy']:.4f}")



