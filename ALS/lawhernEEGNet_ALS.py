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

# EEGNet-specific imports
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

#%%

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data,axis=0)
    return filtered_data

def aggregate_eeg_data(S1, chunk_size=1024):
    """
    Aggregate EEG data by selecting the middle chunk of fixed size from each trial and 
    merging 'L' and 'R' trials sequentially.

    Parameters:
        S1 (dict): Dictionary containing EEG data for each class. Keys are class labels,
                   values are arrays of shape (2, num_samples, num_channels), where the first dimension
                   corresponds to EEG data (index 0) and frequency data (index 1).
        chunk_size (int): The size of each chunk to be extracted from the middle of the trial.

    Returns:
        data (ndarray): Aggregated EEG data with shape (num_trials * 2, chunk_size, numElectrodes),
                        where 2 represents the 'L' and 'R' classes.
        labels (ndarray): Labels for each chunk with shape (num_trials * 2,)
                          where 0 represents 'L' and 1 represents 'R'.
    """
    numElectrodes = S1['L'][0, 1].shape[1]

    # Initialize lists to store aggregated EEG data and labels
    data_list = []
    labels_list = []

    # Loop through each trial and select the middle chunk of 'L' and 'R' trials
    for i in range(S1['L'].shape[1]):
        # Process 'L' trial
        l_trial = S1['L'][0, i]
        l_num_samples = l_trial.shape[0]

        if l_num_samples >= chunk_size:
            # Calculate the start and end indices for the middle chunk
            l_start = (l_num_samples - chunk_size) // 2
            l_end = l_start + chunk_size
            l_middle_chunk = l_trial[l_start:l_end, :]  # Select the middle chunk
            data_list.append(l_middle_chunk)
            labels_list.append(0)  # Label for 'L'

        # Process 'R' trial
        r_trial = S1['R'][0, i]
        r_num_samples = r_trial.shape[0]

        if r_num_samples >= chunk_size:
            # Calculate the start and end indices for the middle chunk
            r_start = (r_num_samples - chunk_size) // 2
            r_end = r_start + chunk_size
            r_middle_chunk = r_trial[r_start:r_end, :]  # Select the middle chunk
            data_list.append(r_middle_chunk)
            labels_list.append(1)  # Label for 'R'

    # Convert lists to numpy arrays
    data = np.stack(data_list, axis=0)  # Shape: (num_trials * 2, chunk_size, numElectrodes)
    labels = np.array(labels_list)      # Shape: (num_trials * 2,)

    return data, labels


# Function to save the results
def save_best_accuracies(best_accuracies, save_path='lawhernEEGNetHealthy.npy'):
    np.save(save_path, best_accuracies)

#%%

# Data directory
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Data'

# Define the subject numbers
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs = 256
idx = ['L', 'R']
# Initialize dictionary to store best accuracies
accuracies = {}


# Define k-fold cross-validation
k = 10  # Number of folds
kf = KFold(n_splits=k, shuffle=False)

# Iterate over each subject
for subject_number in subject_numbers:
    subject_data = {}
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}'][:,:-1]
    S1 = subject_data[f'S{subject_number}'][:, :]

    for key in idx:
        for i in range(S1.shape[1]):
            S1[key][0,i] = bandpass(S1[key][0,i], [8, 30], fs) # S1[key][0,1] i size TxN (T=Sample, N= Channel)
            
    # Prepare data for the subject
    data, labels = aggregate_eeg_data(S1, chunk_size=256*3)

    trials = data.shape[0]
    chans = data.shape[2]
    samples = data.shape[1]
    kernels = 1
    
    data = data.reshape(trials, chans, samples, kernels)  # N x C x T X K
    labels = labels.reshape(-1, 1)
    labels = OneHotEncoder(sparse_output=False).fit_transform(labels)
    
    # Initialize list to store accuracies for each fold
    fold_accuracies = []
    
    # Perform k-fold cross-validation

    # Define the learning rate and other hyperparameters
    initial_learning_rate = 0.0001
    #learning_rate_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
    # Perform k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        train_data, val_data = data[train_index, :, :, :], data[val_index, :, :, :]
        train_labels, val_labels = labels[train_index, :], labels[val_index, :]
    
        # Create EEGNet model
        model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                       dropoutRate=0.5, kernLength=128, F1=8, D=2, F2=16,
                       dropoutType='Dropout')
        
        # Compile the model with a lower learning rate
        optimizer = Adam(learning_rate=initial_learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Model checkpoint and early stopping
        checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.keras', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, verbose=1, restore_best_weights=True)
        
        # Train the model
        fittedModel = model.fit(train_data, train_labels, batch_size=32, epochs=250,
                                verbose=2, callbacks=[checkpointer, early_stopping],
                                validation_data=(val_data, val_labels))
        
        # Evaluate the model on the validation set
        probs = model.predict(val_data)
        preds = probs.argmax(axis=-1)
        acc = np.mean(preds == val_labels.argmax(axis=-1))
        print(f"Subject: S{subject_number}, Classification accuracy for fold {fold + 1}: {acc}")
        
        # Track the best validation accuracy for the fold
        best_acc = max(fittedModel.history['val_accuracy'])
        print(f"Subject: S{subject_number}, Best Accuracy for fold {fold + 1}: {best_acc}")
        
        # Store the fold accuracy
        fold_accuracies.append(best_acc)
        
        # # Plot learning curves for training and validation accuracy
        # plt.plot(fittedModel.history['accuracy'], label='Training Accuracy')
        # plt.plot(fittedModel.history['val_accuracy'], label='Validation Accuracy')
        # plt.title(f'Subject {subject_number}, Fold {fold + 1} - Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()
    
    # Store the fold accuracies for this subject
    accuracies[f'Subject_{subject_number}'] = fold_accuracies

# Save the accuracies dictionary in the current directory
def save_best_accuracies(best_accuracies, save_path='lawhernEEGNetHealthy.npy'):
    np.save(save_path, best_accuracies)

# Get the current working directory
current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'Lawhern2fold_accuracies.npy')
save_best_accuracies(accuracies, save_path=save_path)

print(f"Accuracies saved to {save_path}")

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
# Call the function to print mean, min, and max accuracies directly to the terminal
print_mean_min_max_accuracies(accuracies)