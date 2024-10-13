# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.



https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""
import os
from os.path import dirname, join as pjoin
import scipy as sp
import scipy.io as sio
from scipy import signal
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import networkx as nx
import torch as torch
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from scipy.integrate import simps
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GAT, GraphNorm
from torch_geometric.nn import global_mean_pool
from torch import nn
 
#% Functions
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

def compute_plv(subject_data):
    idx = ['L', 'R']
    numElectrodes = subject_data['L'][0,1].shape[1]
    plv = {field: np.zeros((numElectrodes, numElectrodes, subject_data.shape[1])) for field in idx}
    for i, field in enumerate(idx):
        for j in range(subject_data.shape[1]):
            x = subject_data[field][0, j]
            plv[field][:, :, j] = plvfcn(x)
    l, r = plv['L'], plv['R']
    yl, yr = np.zeros((subject_data.shape[1], 1)), np.ones((subject_data.shape[1], 1))
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((yl, yr), axis=0)
    y = torch.tensor(y, dtype=torch.long)
    return img, y


def create_graphs(plv, threshold):
    
    graphs = []
    for i in range(plv.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv.shape[0]))
        for u in range(plv.shape[0]):
            for v in range(plv.shape[0]):
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        graphs.append(G)
    return graphs


def aggregate_eeg_data(S1,band): #%% This is to get the feat vector
    """
    Aggregate EEG data for each class.

    Parameters:
        S1 (dict): Dictionary containing EEG data for each class. Keys are class labels, 
                   values are arrays of shape (2, num_samples, num_channels), where the first dimension
                   corresponds to EEG data (index 0) and frequency data (index 1).

    Returns:
        l (ndarray): Aggregated EEG data for class 'L'.
        r (ndarray): Aggregated EEG data for class 'R'.
    """
    idx = ['L', 'R']
    numElectrodes = S1['L'][0,1].shape[1];
    max_sizes = {field: 0 for field in idx}

    # Find the maximum size of EEG data for each class
    for field in idx:
        for i in range(S1[field].shape[1]):
            max_sizes[field] = max(max_sizes[field], S1[field][0, i].shape[0])

    # Initialize arrays to store aggregated EEG data
    l = np.zeros((max_sizes['L'], numElectrodes, S1['L'].shape[1]))
    r = np.zeros((max_sizes['R'], numElectrodes, S1['R'].shape[1]))

    # Loop through each sample
    for i in range(S1['L'].shape[1]):
        for j, field in enumerate(idx):
            x = S1[field][0, i]  # EEG data for the current sample
            # Resize x to match the maximum size
            resized_x = np.zeros((max_sizes[field], 22))
            resized_x[:x.shape[0], :] = x
            # Add the resized EEG data to the respective array
            if field == 'L':
                l[:, :, i] += resized_x
            elif field == 'R':
                r[:, :, i] += resized_x

    l = l[..., np.newaxis]
    l = np.copy(l) * np.ones(len(band)-1)

    r = r[..., np.newaxis]
    r = np.copy(r) * np.ones(len(band)-1)
    
    return l, r

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data)
    return filtered_data


def bandpower(data,low,high):

    fs = 256
    # Define window length (2s)
    win = 2* fs
    freqs, psd = signal.welch(data, fs, nperseg=win)
    
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)
    
    # # Plot the power spectral density and fill the delta area
    # plt.figure(figsize=(7, 4))
    # plt.plot(freqs, psd, lw=2, color='k')
    # plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (uV^2 / Hz)')
    # plt.xlim([0, 40])
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram")
    
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    
    # Compute the absolute power by approximating the area under the curve
    power = simps(psd[idx_delta], dx=freq_res)
    
    return power

def bandpowercalc(l,band,fs):   
    x = np.zeros([l.shape[0],l.shape[3],l.shape[2]])
    for i in range(l.shape[0]): #node
        for j in range(l.shape[2]): #sample
            for k in range(0,l.shape[3]): #band
                data = l[i,:,j,k]
                low = band[k]
                high = band[k+1]
                x[i,k,j] = bandpower(data,low,high)

    return x

#%



# % Preparing Data
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Define the subject numbers
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
#subject_numbers = [21, 31, 34, 39]

# Dictionary to hold the loaded data for each subject
subject_data = {}

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    S1 = subject_data[f'S{subject_number}'][:,:]
    
    plv, y = compute_plv(S1)
    threshold = 0.3
    heads = 3
    ep = 500
    graphs = create_graphs(plv, threshold)
    numElectrodes = S1['L'][0,1].shape[1]
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)
    
    #% Initialize an empty list to store edge indices
    edge_indices = [] # % Edge indices are a list of source and target nodes in a graph. Think of it like the adjacency matrix

    # Iterate over the adjacency matrices
    for i in range(adj.shape[2]):
        # Initialize lists to store source and target nodes
        source_nodes = []
        target_nodes = []
        
        # Iterate through each element of the adjacency matrix
        for row in range(adj.shape[0]):
            for col in range(adj.shape[1]):
                # Check if there's an edge
                if adj[row, col, i] >= threshold:
                    # Add source and target nodes to the lists
                    source_nodes.append(row)
                    target_nodes.append(col)
                else:
                    # If no edge exists, add placeholder zeros to maintain size
                    source_nodes.append(0)
                    target_nodes.append(0)
        
        # Create edge index as a LongTensor
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        # Append edge index to the list
        edge_indices.append(edge_index)

    # Stack all edge indices along a new axis to create a 2D tensor
    edge_indices = torch.stack(edge_indices, dim=-1)

    del col,edge_index,i,row,source_nodes,target_nodes
    
    band = list(range(8, 41, 4))
    l,r = aggregate_eeg_data(S1,band)
    l,r = np.transpose(l,[1,0,2,3]),np.transpose(r,[1,0,2,3])
    fs = 256
    
    for i in range(l.shape[3]):
        bp = [band[i],band[i+1]]
        for j in range(l.shape[2]):
            l[:,:,j,i] = bandpass(l[:,:,j,i],bp,sample_rate=fs)
            r[:,:,j,i] = bandpass(r[:,:,j,i],bp,sample_rate=fs)
            #% Convert data from 22x1288x150xF to 22xFx150 where nodes x features x sample
            # features are BP of the band. 
    
    l = bandpowercalc(l,band,fs)
    r = bandpowercalc(r,band,fs)

    x = np.concatenate([l,r],axis=2)
    x = torch.tensor(x,dtype=torch.float32)

    del r,l,S1,i,j,G,bp 
    
    from torch_geometric.data import Data

    data_list = []
    for i in range(np.size(adj,2)):
        data_list.append(Data(x=x[:, :, i], edge_index=edge_indices[:,:,i], y=y[i, 0]))
        
    datal =[]
    datar =[]
    size = len(data_list)
    idx = size//2
    c = [0,idx,idx*2,idx*3]

    datal = data_list[c[0]:c[1]]
    datar = data_list[c[1]:c[2]]

    data_list = []

    for i in range(idx):
        x = [datal[i],datar[i]] #datare[i]]
        data_list.extend(x)


    size = len(data_list)
    
    
    # Initialize KFold with 5 splits
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # List to store highest test accuracies of each fold
    highest_test_accuracies = []
    
    # Iterate over each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
        # Create training and testing sets
        train = [data_list[i] for i in train_idx]
        test = [data_list[i] for i in test_idx]
        
        # Print the number of samples in train and test sets for each fold
        # print(f"Fold {fold + 1}")
        # print(f"Train size: {len(train)}")
        # print(f"Test size: {len(test)}")
        # print("=" * 20)
        
        torch.manual_seed(12345)
    
        train_loader = DataLoader(train, batch_size=32, shuffle=False)
        test_loader = DataLoader(test, batch_size=32, shuffle=False)
    
        #for step, data in enumerate(train_loader):
            #print(f'Step {step + 1}:')
            #print('=======')
            #print(f'Number of graphs in the current batch: {data.num_graphs}')
            #print(data)
            #print()
        
        # class GCN(torch.nn.Module):
        #     def __init__(self, hidden_channels):
        #         super(GCN, self).__init__()
        #         torch.manual_seed(12345)
        #         self.conv1 = GCNConv(8, hidden_channels)  # num node features
        #         self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #         self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #         self.lin = Linear(hidden_channels, 2)  # num of classes
        
        # class GAT(torch.nn.Module):
        #     def __init__(self, hidden_channels,heads):
        #         super(GAT, self).__init__()
        #         torch.manual_seed(12345)
        #         self.conv1 = GATv2Conv(8, hidden_channels, heads=heads, concat=True)  # num node features
        #         self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, concat=True)
        #         self.conv3 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads,concat=True)
        #         self.lin = Softmax()
        #         #self.lin = Linear(hidden_channels*heads, 2)  # num of classes
    
        #     def forward(self, x, edge_index, batch):
        #         # 1. Obtain node embeddings 
        #         x = self.conv1(x, edge_index)
        #         x = x.relu()
        #         x = self.conv2(x, edge_index)
        #         x = x.relu()
        #         x = self.conv3(x, edge_index)
    
        #         # 2. Readout layer
        #         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
    
        #         # 3. Apply a final classifier
        #         x = F.dropout(x, p=0.5, training=self.training)
        #         x = self.lin(x)
                
        #         return x
        
        class GAT(nn.Module):
            def __init__(self, hidden_channels, heads):
                super(GAT, self).__init__()
                
                # Define GAT convolution layers
                self.conv1 = GATv2Conv(8, hidden_channels, heads=heads, concat=True)  # num node features
                self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
                self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
                
                # Define GraphNorm layers
                self.gn1 = GraphNorm(hidden_channels * heads)
                self.gn2 = GraphNorm(hidden_channels * heads)
                self.gn3 = GraphNorm(hidden_channels * heads)
                
                # Define the final linear layer
                self.lin = nn.Linear(hidden_channels * heads, 2)  # num of classes
                
                # Store attention weights from the last layer
                self.last_attention_weights = None
        
            def forward(self, x, edge_index, batch):
                # Apply first GAT layer
                x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
                x = F.relu(x)
                x = self.gn1(x, batch)
                
                # Apply second GAT layer
                x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
                x = F.relu(x)
                x = self.gn2(x, batch)
                
                # Apply third GAT layer and store attention weights
                x, attn3 = self.conv3(x, edge_index, return_attention_weights=True)
                self.last_attention_weights = attn3[1].detach().cpu().numpy()  # Store last layer's attention weights
                x = self.gn3(x, batch)
                
                # Global pooling and classification
                x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                x = F.dropout(x, p=0.50, training=self.training)
                x = self.lin(x)
                
                return x
    
        model = GAT(hidden_channels=22,heads=3)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
    
        def train(model, train_loader, optimizer, criterion):
            model.train()
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
        def test(model, loader):
            model.eval()
            correct = 0
            for data in loader:
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
            
            # Store last layer's attention weights
            attention_weights = model.last_attention_weights
            model.last_attention_weights = None  # Clear stored attention weights after using them
            return correct / len(loader.dataset), attention_weights
        
        # Function to plot dual heatmaps for training and testing attention scores
        def plot_dual_attention_heatmaps(train_attention, test_attention, electrode_labels, subject_number, fold):
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Create a figure with two subplots
        
            # Plot for training attention scores
            axes[0].imshow(train_attention, cmap="Blues", aspect='auto')
            axes[0].set_title(f'Subject {subject_number} Fold {fold+1} - Training Attention')
            axes[0].set_xticks(np.arange(len(electrode_labels)))
            axes[0].set_yticks(np.arange(len(electrode_labels)))
            axes[0].set_xticklabels(electrode_labels)
            axes[0].set_yticklabels(electrode_labels)
            axes[0].set_xlabel("Electrodes")
            axes[0].set_ylabel("Electrodes")
            axes[0].colorbar = plt.colorbar(axes[0].imshow(train_attention, cmap="Blues", aspect='auto'), ax=axes[0])
        
            # Plot for testing attention scores
            axes[1].imshow(test_attention, cmap="Blues", aspect='auto')
            axes[1].set_title(f'Subject {subject_number} Fold {fold+1} - Testing Attention')
            axes[1].set_xticks(np.arange(len(electrode_labels)))
            axes[1].set_yticks(np.arange(len(electrode_labels)))
            axes[1].set_xticklabels(electrode_labels)
            axes[1].set_yticklabels(electrode_labels)
            axes[1].set_xlabel("Electrodes")
            axes[1].set_ylabel("Electrodes")
            axes[1].colorbar = plt.colorbar(axes[1].imshow(test_attention, cmap="Blues", aspect='auto'), ax=axes[1])
        
            plt.tight_layout()
            plt.show()
            
            
        optimal = [0, 0, 0]
        best_train_attention = None
        best_test_attention = None
        
        for epoch in range(1, 500):
            train(model, train_loader, optimizer, criterion)
            train_acc, train_attention = test(model, train_loader)
            test_acc, test_attention = test(model, test_loader)
            av_acc = np.mean([train_acc, test_acc])
    
            # Store only if current test accuracy is higher
            if test_acc > optimal[2]:
                optimal = [av_acc, train_acc, test_acc]
                best_train_attention = train_attention  # Save best training attention
                best_test_attention = test_attention   # Save best testing attention
    
        highest_test_accuracies.append(optimal[2])
    
        # Plot both training and testing attention scores if the best attention data is available
        if best_train_attention is not None and best_test_attention is not None:
            # Reshape and process the attention scores
            trainbatch = best_train_attention.shape[0] // 22 // 22
            testbatch = best_test_attention.shape[0] // 22 // 22
            
            best_train_attention = best_train_attention[:,-1].reshape(22, 22, trainbatch)
            best_train_attention = np.mean(best_train_attention, axis=2)
            best_train_attention = best_train_attention[:-3, :-3]  # Remove the last 3 rows/columns
    
            best_test_attention = best_test_attention[:,-1].reshape(22, 22, testbatch)
            best_test_attention = np.mean(best_test_attention, axis=2)
            best_test_attention = best_test_attention[:-3, :-3]  # Remove the last 3 rows/columns
    
            # Plot the dual attention heatmaps
            plot_dual_attention_heatmaps(best_train_attention, best_test_attention, 
                                         electrode_labels=["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", 
                                                           "T7", "C3", "CZ", "C4", "T8", "P7", "P3", 
                                                           "PZ", "P4", "P8", "O1", "O2"],
                                         subject_number=subject_number, fold=fold)
    
    
    # Calculate and print the mean of the highest test accuracies across all folds
    meanhigh = np.mean(highest_test_accuracies)
    maxhigh = np.max(highest_test_accuracies)
    minhigh = np.min(highest_test_accuracies)
    print(f'S{subject_number}: Mean: {meanhigh:.4f}, Max: {maxhigh:.4f}, Min: {minhigh:.4f}')