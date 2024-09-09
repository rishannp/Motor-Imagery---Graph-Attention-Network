import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from os.path import join as pjoin
from sklearn.model_selection import KFold


def aggregate_eeg_data(S1, chunk_size=250):
    """
    Aggregate EEG data by splitting each trial into chunks of fixed size and discarding any remainder,
    merging 'L' and 'R' chunks sequentially.

    Parameters:
        S1 (dict): Dictionary containing EEG data for each class. Keys are class labels,
                   values are arrays of shape (2, num_samples, num_channels), where the first dimension
                   corresponds to EEG data (index 0) and frequency data (index 1).
        chunk_size (int): The size of each chunk to which the data will be split.

    Returns:
        data (ndarray): Aggregated EEG data with shape (chunk_size, numElectrodes, num_trials * num_chunks_per_trial).
        labels (ndarray): Labels for each chunk with shape (num_trials * num_chunks_per_trial,)
                          where 1 represents 'L' and 2 represents 'R'.
    """
    numElectrodes = S1['L'][0, 1].shape[1]

    # Initialize lists to store aggregated EEG data and labels
    data_list = []
    labels_list = []

    # Loop through each trial and interleave L and R trials
    for i in range(S1['L'].shape[1]):
        # Process 'L' trial
        l_trial = S1['L'][0, i]
        l_num_samples = l_trial.shape[0]
        l_num_chunks = l_num_samples // chunk_size
        
        for j in range(l_num_chunks):
            chunk = l_trial[j * chunk_size:(j + 1) * chunk_size, :]
            data_list.append(chunk)
            labels_list.append(0)  # Label for 'L'

        # Process 'R' trial
        r_trial = S1['R'][0, i]
        r_num_samples = r_trial.shape[0]
        r_num_chunks = r_num_samples // chunk_size
        
        for j in range(r_num_chunks):
            chunk = r_trial[j * chunk_size:(j + 1) * chunk_size, :]
            data_list.append(chunk)
            labels_list.append(1)  # Label for 'R'

    # Convert lists to numpy arrays
    data = np.stack(data_list, axis=0)  # Shape: (num_chunks_per_trial * num_trials * 2, chunk_size, numElectrodes)
    labels = np.array(labels_list)      # Shape: (num_chunks_per_trial * num_trials * 2,)

    return data, labels



class EEGDataset(Dataset):
    def __init__(self, data, labels):
        """
        Initialize the dataset with EEG data and labels.

        Parameters:
            data (ndarray): EEG data with shape (num_trials, num_samples, num_electrodes).
            labels (ndarray): Labels for each trial with shape (num_trials,).
        """
        # Permute data to the format expected by the model (e.g., (num_trials, num_electrodes, num_samples))
        self.data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)  # Shape: (num_trials, num_electrodes, num_samples)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Shape: (num_trials,)
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its label at the specified index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (sample, label) where sample has shape (num_electrodes, num_samples)
        """
        # Check index is within bounds
        if idx >= len(self):
            raise IndexError(f'Index {idx} out of bounds for dataset with length {len(self)}')

        sample = self.data[idx]  # Shape: (num_electrodes, num_samples)
        label = self.labels[idx]
        return sample, label

class EEGNet(nn.Module):
    def __init__(self, chunk_size, num_electrodes, F1=8, F2=16, D=2, num_classes=2, kernel_1=64, kernel_2=16, dropout=0.25):
        super(EEGNet, self).__init__()

        # First Conv2D Layer - Temporal Conv across the time dimension for each electrode
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=F1, 
            kernel_size=(1, kernel_1), 
            stride=1, 
            padding=0
        )  # Conv only on the time dimension
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Depthwise Convolution - Spanning the electrode dimension
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1, 
            out_channels=F1 * D, 
            kernel_size=(num_electrodes, 1),  # Depthwise convolution on the electrode dimension
            groups=F1,  # Depthwise separation
            padding=0  # valid padding, no padding
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        
        # Activation and pooling
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))  # Pool along the time dimension
        self.dropout1 = nn.Dropout(p=dropout)

        # Separable Convolution (spatial-temporal filtering)
        self.separable_conv = nn.Conv2d(
            in_channels=F1 * D, 
            out_channels=F2, 
            kernel_size=(1, kernel_2), 
            stride=1, 
            padding='same'
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)

        # Second Pooling Layer
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))  # Pool along the time dimension
        self.dropout2 = nn.Dropout(p=dropout)

        # Fully Connected Layer
        self.flatten = nn.Flatten()
        #self.fc = nn.Linear(F2 * (chunk_size // 32), num_classes)
        self.fc = nn.Linear(80, num_classes)


    def forward(self, x):
        #print(f"Input shape: {x.shape}")  # Should be (batch_size, num_electrodes, num_samples)

        x = x.view(x.size(0), 1, x.size(1), x.size(2))  # Reshape to (batch_size, 1, num_electrodes, num_samples)
        #print(f"After Reshape: {x.shape}")  # (batch_size, 1, num_electrodes, num_samples)

        # Conv2D Layer
        x = self.conv1(x)
        #print(f"After Conv2D (Temporal Conv): {x.shape}")  # Output size calculation
        x = self.batchnorm1(x)
        #print(f"After BatchNorm1: {x.shape}")

        x = self.elu(x)
        #print(f"After ELU (1st): {x.shape}")

        # Depthwise Convolution (Electrode Conv)
        x = self.depthwise_conv(x)
        #print(f"After Depthwise Conv2D (Spatial Conv): {x.shape}")
        x = self.batchnorm2(x)
        #print(f"After BatchNorm2: {x.shape}")

        x = self.elu(x)
        #print(f"After ELU (2nd): {x.shape}")

        # First Pooling Layer
        x = self.avgpool1(x)
        #print(f"After AvgPool2D (1st): {x.shape}")
        x = self.dropout1(x)
        #print(f"After Dropout1: {x.shape}")

        # Separable Convolution
        x = self.separable_conv(x)
        #print(f"After Separable Conv2D: {x.shape}")
        x = self.batchnorm3(x)
        #print(f"After BatchNorm3: {x.shape}")

        x = self.elu(x)
        #print(f"After ELU (3rd): {x.shape}")

        # Second Pooling Layer
        x = self.avgpool2(x)
        #print(f"After AvgPool2D (2nd): {x.shape}")
        x = self.dropout2(x)
        #print(f"After Dropout2: {x.shape}")

        # Flatten and Fully Connected
        x = self.flatten(x)
        #print(f"After Flatten: {x.shape}")
        x = self.fc(x)
        #print(f"After Fully Connected (FC): {x.shape}")

        return x



# Training and evaluation functions
def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    
    return accuracy

# Data directory
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Data'

# Define the subject numbers
subject_numbers = list(range(1, 10))  # This will generate [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Function to save the results
def save_best_accuracies(best_accuracies, save_path='EEGNetHealthy.npy'):
    np.save(save_path, best_accuracies)

# Initialize dictionary to store best accuracies
accuracies = {}
subject_data = {}

# Define k-fold cross-validation
k = 10  # Number of folds
kf = KFold(n_splits=k, shuffle=False)

for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'A{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents['subject']
    S1 = subject_data[f'S{subject_number}']
    S1 = S1[9,:].reshape(1, -1)
    fs =250
    idx = ['L','R']
    
    # Prepare data for the subject
    data, labels = aggregate_eeg_data(S1, chunk_size=250)
    
    # Initialize dictionary to store fold results
    subject_accuracies = {}

    # Perform k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        train_data, val_data = data[train_index], data[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]
        
        # Create datasets and dataloaders
        train_dataset = EEGDataset(train_data, train_labels)
        val_dataset = EEGDataset(val_data, val_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model, criterion, and optimizer
        chunk_size = data.shape[1]
        num_electrodes = data.shape[2]
        model = EEGNet(chunk_size=chunk_size, num_electrodes=num_electrodes, F1=8, F2=16, D=2, num_classes=2, kernel_1=64, kernel_2=16, dropout=0.5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train the model
        print(f'Training for Subject {subject_number}, Fold {fold+1}...')
        train(model, train_dataloader, criterion, optimizer, num_epochs=250)
        
        # Evaluate the model
        print(f'Evaluating for Subject {subject_number}, Fold {fold+1}...')
        accuracy = evaluate(model, val_dataloader)
        subject_accuracies[fold+1] = accuracy
    
    # Store the accuracies for this subject
    accuracies[subject_number] = subject_accuracies

# Save all accuracies to file
save_best_accuracies(accuracies)


def print_mean_min_max_accuracies(accuracies):
    # Iterate through each subject in the accuracies dictionary
    for subject_number, fold_accuracies in accuracies.items():
        # Get all fold accuracies for the current subject
        fold_values = list(fold_accuracies.values())
        
        # Calculate mean, min, and max of the fold accuracies
        mean_accuracy = np.mean(fold_values)
        min_accuracy = np.min(fold_values)
        max_accuracy = np.max(fold_values)
        
        # Print the mean, min, and max accuracy for the subject
        print(f'Subject {subject_number}: Mean Accuracy = {mean_accuracy:.4f}, '
              f'Min Accuracy = {min_accuracy:.4f}, Max Accuracy = {max_accuracy:.4f}')

# Example usage:
# Call the function to print mean, min, and max accuracies directly to the terminal
print_mean_min_max_accuracies(accuracies)
