import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from os.path import join as pjoin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import mne
import os
from EEGModels import EEGNet

#%% MNE Preprocessing and Data Loading

# Directory containing the GDF files
data_dir = r'C:\Users\uceerjp\Desktop\PhD\IV2a\BCICIV_2a_gdf\Data'

# List of subject IDs
subjects = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# Event IDs for the four classes
event_id = {
    'left_hand': 7,   # LH
    'right_hand': 8,  # RH
    'feet': 9,        # Feet
    'tongue': 10      # Tongue
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

# Loop through each subject to preprocess
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
    
    # Remove the last three channels (EOG)
    all_channel_names = raw.info['ch_names']
    eog_channel_names = all_channel_names[-3:]  # Get the names of the last three channels
    raw.drop_channels(eog_channel_names)
    
    # Downsample to 128 Hz
    raw.resample(new_sfreq, npad='auto')
    
    # Get events and annotations
    events, _ = mne.events_from_annotations(raw)
    
    # Create epochs for all 4 classes
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                        baseline=None, preload=True, event_repeated='drop')
    
    # Extract the data as a numpy array
    data = epochs.get_data()
    
    # Create label vector
    labels = np.array([event[2] for event in events if event[2] in event_id.values()])
    
    # Store the data and labels in the dictionary with the key 'S1', 'S2', etc.
    subject_data[f'S{subject}'] = {'data': data, 'labels': labels}
    
    print(f'Stored data for subject S{subject} with shape {data.shape} and labels shape {labels.shape}')


#%% Define binary class permutations
class_pairs = [
    ('left_hand', 'right_hand'),
    ('left_hand', 'feet'),
    ('right_hand', 'feet'),
    ('feet', 'tongue'),
]

# Iterate over binary class pairs
for pair in class_pairs:
    print(f"Running classification for pair: {pair[0]} vs {pair[1]}")
    
    # Extract event IDs for the current pair
    binary_event_id = {pair[0]: event_id[pair[0]], pair[1]: event_id[pair[1]]}
    
    # Dictionary to store best accuracies for this pair
    accuracies = {}

    # Define k-fold cross-validation
    k = 10  # Number of folds
    kf = KFold(n_splits=k, shuffle=False)

    # Iterate over each subject
    for subject in subjects:
        S = subject_data[f'S{subject}']
        data = S['data']
        labels = S['labels']
        
        # Filter the data for only the selected classes
        binary_mask = np.isin(labels, list(binary_event_id.values()))
        binary_data = data[binary_mask]
        binary_labels = labels[binary_mask]

        # Re-label the selected classes as 0 and 1
        binary_labels = np.where(binary_labels == binary_event_id[pair[0]], 0, 1)

        # Reshape the data for the model (N x C x T x 1)
        trials = binary_data.shape[0]
        chans = binary_data.shape[1]
        samples = binary_data.shape[2]
        kernels = 1
        binary_data = binary_data.reshape(trials, chans, samples, kernels)

        # One-hot encode the labels
        binary_labels = binary_labels.reshape(-1, 1)
        binary_labels = OneHotEncoder(sparse_output=False).fit_transform(binary_labels)

        # Initialize list to store accuracies for each fold
        fold_accuracies = []

        # Perform k-fold cross-validation
        for fold, (train_index, val_index) in enumerate(kf.split(binary_data)):
            train_data, val_data = binary_data[train_index], binary_data[val_index]
            train_labels, val_labels = binary_labels[train_index], binary_labels[val_index]

            # Create EEGNet model for binary classification (2 classes)
            model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                           dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                           dropoutType='Dropout')

            # Compile the model
            optimizer = Adam(learning_rate=0.001)
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
            print(f"Subject: S{subject}, {pair[0]} vs {pair[1]} fold {fold + 1}: Accuracy = {acc}")

            # Store fold accuracy
            fold_accuracies.append(acc)

        # Store the accuracies for the subject
        accuracies[f'Subject_{subject}'] = fold_accuracies

    # Save accuracies for this pair
    save_path = f'{pair[0]}_vs_{pair[1]}_accuracies.npy'
    np.save(save_path, accuracies)
    print(f"Accuracies for {pair[0]} vs {pair[1]} saved to {save_path}")

#%%

import numpy as np
import os

# Directory where the accuracy files are saved
save_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Motor Imagery - Graph Attention Network\Healthy\ReplicatingEEGNet'

# List of the class pairs for which files were saved
class_pairs = [
    ('left_hand', 'right_hand'),
    ('left_hand', 'feet'),
    ('right_hand', 'feet'),
    ('feet', 'tongue'),
    # Add more pairs if needed
]

# Dictionary to store the mean accuracies for each class pair
mean_accuracies = {}

# Iterate through each class pair
for pair in class_pairs:
    # Construct the filename for the current pair's accuracy file
    filename = f'{pair[0]}_vs_{pair[1]}_accuracies.npy'
    filepath = os.path.join(save_dir, filename)
    
    # Load the accuracy file
    accuracies = np.load(filepath, allow_pickle=True).item()  # Load as a dictionary
    
    # Dictionary to store mean accuracies per subject
    subject_means = {}
    
    # Iterate through each subject's accuracies in the loaded file
    for subject, fold_accuracies in accuracies.items():
        # Calculate the mean accuracy across all folds for this subject
        mean_accuracy = np.mean(fold_accuracies)
        
        # Store the mean accuracy for this subject
        subject_means[subject] = mean_accuracy
    
    # Store the mean accuracies for this class pair
    mean_accuracies[f'{pair[0]}_vs_{pair[1]}'] = subject_means

# Print the mean accuracies
for pair, subject_means in mean_accuracies.items():
    print(f"\nMean accuracies for {pair}:")
    for subject, mean_acc in subject_means.items():
        print(f"  {subject}: {mean_acc:.4f}")

