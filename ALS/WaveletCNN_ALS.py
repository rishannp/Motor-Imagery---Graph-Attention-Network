# -*- coding: utf-8 -*-
"""
Rishan Patel, UCL, Bioelectronics Group.
WaveletCNN Implementation -  10.1109/ACCESS.2018.2889093
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
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Conv1D, BatchNormalization, MaxPool2D
from keras.optimizers import Adam 
import keras 
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from collections import Counter
#%% Functions

def split_and_extend_dataset(S1, chunk_size=256, fs=256, duration=5):
    """
    Splits the trials in S1 into chunks of size `chunk_size` using only the last `duration` seconds of each trial 
    and returns the extended dataset.

    Parameters:
    - S1: A dictionary containing the EEG data for classes 'L' and 'R'. Each entry is an array of trials.
    - chunk_size: The size of each chunk along the time dimension (default is 256).
    - fs: Sampling frequency (default is 256 Hz).
    - duration: The duration in seconds to retain from the end of each trial (default is 5 seconds).

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

            # Determine the starting point to keep only the last `points_to_keep` data points
            start_idx = max(0, N - points_to_keep)
            trial_array = trial_array[start_idx:, :]

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
            selected_channels = epoch_data[:, [8, 9, 10]]  # Select channels 8, 9, and 10

            # Apply bandpass filter to selected channels
            filtered_channels = bandpass_filter(selected_channels)
            
            # Overwrite the entire data with only the filtered channels
            subject_data[idx][trial][0] = filtered_channels

    return subject_data

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

def generate_wavelet_images(subject_data, subject_number, base_output_dir, sampling_rate=256, totalscal=64):
    """
    Generate wavelet transform images for the given subject data.

    Parameters:
    - subject_data: numpy array of shape (n_samples, n_channels, n_trials) for a single subject.
    - subject_number: identifier for the subject (e.g., 'S1').
    - base_output_dir: base directory to save the wavelet images.
    - sampling_rate: the sampling rate of the EEG data.
    - totalscal: number of scales to be used in the wavelet transform.
    """
    subject_output_dir = os.path.join(base_output_dir, subject_number)  # Create a subject-specific output directory
    os.makedirs(subject_output_dir, exist_ok=True)  # Create the subject output directory if it doesn't exist
    wavename = 'morl'  # Wavelet name
    
    wavelet_arrays = []  # List to hold the wavelet transform arrays
    
    # Calculate scales based on the central frequency
    fc = pywt.central_frequency(wavename)  # Central frequency
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(1, totalscal + 1)

    # Get the number of trials from the shape of the data
    n_trials = subject_data.shape[2]  # Shape is (n_samples, n_channels, n_trials)

    # Loop through each trial
    for i in range(n_trials):
        # Get data for the current trial, with shape (n_samples, n_channels)
        trial_data = subject_data[:, :, i]  

        # Accessing C3, Cz, and C4 channels
        dataC3 = trial_data[:, 0]  # C3 channel data (first channel)
        dataCz = trial_data[:, 1]  # Cz channel data (second channel)
        dataC4 = trial_data[:, 2]  # C4 channel data (third channel)

        # Wavelet transform for each channel
        cwtC3, _ = pywt.cwt(dataC3, scales, wavename, 1.0 / sampling_rate)
        cwtCz, _ = pywt.cwt(dataCz, scales, wavename, 1.0 / sampling_rate)
        cwtC4, _ = pywt.cwt(dataC4, scales, wavename, 1.0 / sampling_rate)

        # Concatenate the absolute values of the wavelet transforms
        cwtmat = np.concatenate([abs(cwtC3[7:30, :]), abs(cwtCz[7:30, :]), abs(cwtC4[7:30, :])], axis=0)  # C3, Cz, then C4


        # # Create the figure
        # fig = plt.figure()
        # plt.contourf(cwtmat)
        # plt.axis('off')  # Remove axes
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)

        # # Determine the label based on the trial index
        # if i < n_trials // 2:  # First half is Left Motor Imagery (label 1)
        #     label = 1
        # else:  # Second half is Right Motor Imagery (label 2)
        #     label = 2

        # # Save the figure
        # filepath = os.path.join(subject_output_dir, f"{label}_{i}.jpg")
        # fig.savefig(filepath)
        # plt.close(fig)  # Close the figure to avoid memory issues
        
        # Append the wavelet matrix for this trial to the list
        wavelet_arrays.append(cwtmat)
        
    print(f'Wavelet transform images generated for subject {subject_number}.')
    return wavelet_arrays

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
    model.add(Conv2D(16, (4,4), padding='same', input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(8,8)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (4,4), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(240, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0003, decay=1e-6),
                  metrics=['accuracy'])
    return model
#%% Main
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Define the subject numbers
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# Dictionary to hold the loaded data for each subject
data = {}
subject_accuracies = {}

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    data[f'S{subject_number}'] = split_and_extend_dataset(data[f'S{subject_number}'], chunk_size=256*5, fs=256, duration=5)
    data[f'S{subject_number}'] = apply_bandpass_and_select_channels(data[f'S{subject_number}'])
    data[f'S{subject_number}'] = concatenate_subject_data(data[f'S{subject_number}'])
    data[f'S{subject_number}'] = generate_wavelet_images(data[f'S{subject_number}'], subject_number=f'S{subject_number}', base_output_dir=r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Data')
    img, label = add_labels_to_data(data[f'S{subject_number}'])
    X, Y = process_images_and_labels(img, label)
    
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



