

als_patients = {
    "EEGNet": {
        "S1": {"Mean": 65.69, "Max": 74.19, "Min": 56.25},
        "S2": {"Mean": 67.06, "Max": 81.82, "Min": 60.61},
        "S5": {"Mean": 56.93, "Max": 70.97, "Min": 41.94},
        "S9": {"Mean": 65.59, "Max": 83.87, "Min": 48.39},
        "S21": {"Mean": 55.69, "Max": 62.50, "Min": 45.16},
        "S31": {"Mean": 62.58, "Max": 73.33, "Min": 45.16},
        "S34": {"Mean": 57.34, "Max": 65.52, "Min": 43.33},
        "S39": {"Mean": 61.99, "Max": 70.00, "Min": 51.72},
        "Average": {"Mean": 61.61, "Max": 72.78, "Min": 49.07}
    },
    "DeepConvNet": {
        "S1": {"Mean": 66.56, "Max": 71.88, "Min": 58.06},
        "S2": {"Mean": 60.12, "Max": 66.67, "Min": 51.52},
        "S5": {"Mean": 63.34, "Max": 77.42, "Min": 48.39},
        "S9": {"Mean": 67.84, "Max": 90.32, "Min": 51.61},
        "S21": {"Mean": 58.28, "Max": 67.74, "Min": 48.39},
        "S31": {"Mean": 63.83, "Max": 77.42, "Min": 58.06},
        "S34": {"Mean": 62.06, "Max": 73.33, "Min": 50.00},
        "S39": {"Mean": 61.99, "Max": 72.41, "Min": 55.17},
        "Average": {"Mean": 63.00, "Max": 74.65, "Min": 52.65}
    },
    "ShallowConvNet": {
        "S1": {"Mean": 61.82, "Max": 74.19, "Min": 43.75},
        "S2": {"Mean": 68.57, "Max": 78.79, "Min": 54.55},
        "S5": {"Mean": 58.87, "Max": 70.97, "Min": 50.00},
        "S9": {"Mean": 66.54, "Max": 93.55, "Min": 51.61},
        "S21": {"Mean": 59.85, "Max": 65.62, "Min": 45.16},
        "S31": {"Mean": 55.73, "Max": 64.52, "Min": 35.48},
        "S34": {"Mean": 63.75, "Max": 72.41, "Min": 55.17},
        "S39": {"Mean": 63.01, "Max": 76.67, "Min": 51.72},
        "Average": {"Mean": 62.27, "Max": 74.59, "Min": 48.43}
    },
    "WaveletCNN": {
        "S1": {"Mean": 61.82, "Max": 74.19, "Min": 43.75},
        "S2": {"Mean": 68.57, "Max": 78.79, "Min": 54.55},
        "S5": {"Mean": 58.87, "Max": 70.97, "Min": 50.00},
        "S9": {"Mean": 66.54, "Max": 93.55, "Min": 51.61},
        "S21": {"Mean": 59.85, "Max": 65.62, "Min": 45.16},
        "S31": {"Mean": 55.73, "Max": 64.52, "Min": 35.48},
        "S34": {"Mean": 63.75, "Max": 72.41, "Min": 55.17},
        "S39": {"Mean": 63.01, "Max": 76.67, "Min": 51.72},
        "Average": {"Mean": 62.27, "Max": 74.59, "Min": 48.43}
    },
    "CSP": {
        "S1": {"Mean": 51.57, "Max": 65.62, "Min": 34.38},
        "S2": {"Mean": 65.21, "Max": 76.47, "Min": 58.82},
        "S5": {"Mean": 50.64, "Max": 67.74, "Min": 37.50},
        "S9": {"Mean": 88.31, "Max": 93.75, "Min": 81.25},
        "S21": {"Mean": 82.97, "Max": 93.75, "Min": 74.19},
        "S31": {"Mean": 75.40, "Max": 93.75, "Min": 64.52},
        "S34": {"Mean": 74.76, "Max": 96.67, "Min": 51.72},
        "S39": {"Mean": 83.16, "Max": 93.33, "Min": 72.41},
        "Average": {"Mean": 72.27, "Max": 83.90, "Min": 58.95}
    },
    "GCN 1": {
        "S1": {"Mean": 69.37, "Max": 78.12, "Min": 59.38},
        "S2": {"Mean": 77.59, "Max": 91.43, "Min": 64.71},
        "S5": {"Mean": 66.64, "Max": 71.88, "Min": 61.29},
        "S9": {"Mean": 66.67, "Max": 84.38, "Min": 56.25},
        "S21": {"Mean": 65.31, "Max": 78.12, "Min": 56.25},
        "S31": {"Mean": 67.19, "Max": 81.25, "Min": 59.38},
        "S34": {"Mean": 60.67, "Max": 76.67, "Min": 53.33},
        "S39": {"Mean": 68.00, "Max": 76.67, "Min": 60.00},
        "Average": {"Mean": 67.68, "Max": 79.82, "Min": 58.83}
    },
    "GCN 2": {
        "S1": {"Mean": 65.94, "Max": 71.88, "Min": 59.38},
        "S2": {"Mean": 68.34, "Max": 80.00, "Min": 54.29},
        "S5": {"Mean": 62.88, "Max": 71.88, "Min": 58.06},
        "S9": {"Mean": 64.80, "Max": 71.88, "Min": 56.25},
        "S21": {"Mean": 63.12, "Max": 71.88, "Min": 53.12},
        "S31": {"Mean": 65.00, "Max": 75.00, "Min": 53.12},
        "S34": {"Mean": 65.00, "Max": 70.00, "Min": 56.67},
        "S39": {"Mean": 61.67, "Max": 66.67, "Min": 53.33},
        "Average": {"Mean": 64.59, "Max": 72.40, "Min": 55.53}
    },
    # "GAT (T=0, H=1)": {
    #     "S1": {"Mean": 70.31, "Max": 78.12, "Min": 65.62},
    #     "S2": {"Mean": 76.77, "Max": 85.29, "Min": 68.57},
    #     "S5": {"Mean": 72.30, "Max": 78.12, "Min": 61.29},
    #     "S9": {"Mean": 71.71, "Max": 78.12, "Min": 62.50},
    #     "S21": {"Mean": 71.88, "Max": 78.12, "Min": 65.62},
    #     "S31": {"Mean": 72.19, "Max": 81.25, "Min": 65.62},
    #     "S34": {"Mean": 70.67, "Max": 76.67, "Min": 60.00},
    #     "S39": {"Mean": 74.33, "Max": 86.67, "Min": 66.67},
    #     "Average": {"Mean": 72.52, "Max": 80.30, "Min": 64.49}
    # },
    ## Adjust the mins for this model
    "GAT (T=0.3, H=3)": {
        "S1": {"Mean": 70.31, "Max": 78.12, "Min": 62.50},
        "S2": {"Mean": 77.05, "Max": 82.35, "Min": 71.43},
        "S5": {"Mean": 71.36, "Max": 81.25, "Min": 65.62},
        "S9": {"Mean": 74.19, "Max": 84.38, "Min": 64.52},
        "S21": {"Mean": 75.62, "Max": 84.38, "Min": 68.75},
        "S31": {"Mean": 75.31, "Max": 81.25, "Min": 68.75},
        "S34": {"Mean": 70.67, "Max": 80.00, "Min": 63.33},
        "S39": {"Mean": 78.00, "Max": 86.67, "Min": 73.33},
        "Average": {"Mean": 74.06, "Max": 82.30, "Min": 67.28}
    }
}


healthy_patients = {
    "EEGNet": {
        "S1": {"Mean": 56.61, "Max": 67.86, "Min": 44.83},
        "S2": {"Mean": 65.23, "Max": 79.31, "Min": 55.17},
        "S3": {"Mean": 61.25, "Max": 75.00, "Min": 50.00},
        "S4": {"Mean": 61.25, "Max": 75.00, "Min": 50.00},
        "S5": {"Mean": 65.23, "Max": 75.86, "Min": 51.72},
        "S6": {"Mean": 73.26, "Max": 86.21, "Min": 65.52},
        "S7": {"Mean": 71.48, "Max": 86.21, "Min": 60.71},
        "S8": {"Mean": 65.65, "Max": 72.41, "Min": 58.62},
        "S9": {"Mean": 69.43, "Max": 86.21, "Min": 62.07},
        "Average": {"Mean": 65.34, "Max": 76.78, "Min": 54.88}
    },
    "DeepConvNet": {
        "S1": {"Mean": 64.22, "Max": 72.41, "Min": 58.62},
        "S2": {"Mean": 69.78, "Max": 79.31, "Min": 58.62},
        "S3": {"Mean": 64.98, "Max": 71.43, "Min": 55.17},
        "S4": {"Mean": 70.42, "Max": 79.17, "Min": 66.67},
        "S5": {"Mean": 69.79, "Max": 75.86, "Min": 58.62},
        "S6": {"Mean": 68.73, "Max": 79.31, "Min": 62.07},
        "S7": {"Mean": 70.47, "Max": 75.86, "Min": 67.86},
        "S8": {"Mean": 62.54, "Max": 68.97, "Min": 48.28},
        "S9": {"Mean": 67.62, "Max": 82.76, "Min": 53.57},
        "Average": {"Mean": 67.57, "Max": 76.85, "Min": 57.80}
    },
    "ShallowConvNet": {
        "S1": {"Mean": 69.43, "Max": 79.31, "Min": 62.07},
        "S2": {"Mean": 65.63, "Max": 75.86, "Min": 55.17},
        "S3": {"Mean": 62.48, "Max": 68.97, "Min": 51.72},
        "S4": {"Mean": 65.42, "Max": 79.17, "Min": 54.17},
        "S5": {"Mean": 63.14, "Max": 68.97, "Min": 46.43},
        "S6": {"Mean": 67.78, "Max": 82.14, "Min": 51.72},
        "S7": {"Mean": 70.46, "Max": 79.31, "Min": 62.07},
        "S8": {"Mean": 67.03, "Max": 79.31, "Min": 55.17},
        "S9": {"Mean": 66.28, "Max": 79.31, "Min": 57.14},
        "Average": {"Mean": 66.75, "Max": 76.80, "Min": 54.52}
    },
    "CSP": {
        "S1": {"Mean": 77.71, "Max": 86.67, "Min": 64.29},
        "S2": {"Mean": 61.90, "Max": 85.71, "Min": 42.86},
        "S3": {"Mean": 91.71, "Max": 100.00, "Min": 80.00},
        "S4": {"Mean": 70.76, "Max": 80.00, "Min": 35.71},
        "S5": {"Mean": 63.24, "Max": 73.33, "Min": 53.33},
        "S6": {"Mean": 66.62, "Max": 86.67, "Min": 42.86},
        "S7": {"Mean": 70.86, "Max": 85.71, "Min": 46.67},
        "S8": {"Mean": 95.76, "Max": 100.00, "Min": 85.71},
        "S9": {"Mean": 86.90, "Max": 100.00, "Min": 73.33},
        "Average": {"Mean": 77.50, "Max": 88.53, "Min": 58.63}
    },
    "GCN 1": {
        "S1": {"Mean": 63.19, "Max": 79.31, "Min": 55.17},
        "S2": {"Mean": 63.55, "Max": 72.41, "Min": 48.28},
        "S3": {"Mean": 65.30, "Max": 75.86, "Min": 51.72},
        "S4": {"Mean": 65.83, "Max": 83.33, "Min": 58.33},
        "S5": {"Mean": 62.86, "Max": 71.43, "Min": 51.72},
        "S6": {"Mean": 63.17, "Max": 72.41, "Min": 53.57},
        "S7": {"Mean": 64.90, "Max": 72.41, "Min": 57.14},
        "S8": {"Mean": 65.96, "Max": 79.31, "Min": 53.57},
        "S9": {"Mean": 63.87, "Max": 75.86, "Min": 57.14},
        "Average": {"Mean": 64.25, "Max": 76.84, "Min": 54.59}
    },
    "GCN 2": {
        "S1": {"Mean": 65.53, "Max": 79.31, "Min": 51.72},
        "S2": {"Mean": 67.71, "Max": 79.31, "Min": 58.62},
        "S3": {"Mean": 69.77, "Max": 86.21, "Min": 60.71},
        "S4": {"Mean": 65.00, "Max": 75.00, "Min": 50.00},
        "S5": {"Mean": 67.73, "Max": 75.86, "Min": 55.17},
        "S6": {"Mean": 68.76, "Max": 79.31, "Min": 62.07},
        "S7": {"Mean": 67.32, "Max": 82.76, "Min": 57.14},
        "S8": {"Mean": 70.14, "Max": 75.86, "Min": 62.07},
        "S9": {"Mean": 62.54, "Max": 75.86, "Min": 51.72},
        "Average": {"Mean": 67.06, "Max": 78.16, "Min": 55.06}
    },
    
    "GAT (T=0.3, H=1)": {
        "S1": {"Mean": 68.34, "Max": 82.76, "Min": 57.14},
        "S2": {"Mean": 71.51, "Max": 75.86, "Min": 65.52},
        "S3": {"Mean": 71.18, "Max": 79.31, "Min": 62.07},
        "S4": {"Mean": 71.67, "Max": 83.33, "Min": 58.33},
        "S5": {"Mean": 73.65, "Max": 82.14, "Min": 65.52},
        "S6": {"Mean": 73.33, "Max": 85.71, "Min": 65.52},
        "S7": {"Mean": 73.61, "Max": 79.31, "Min": 65.52},
        "S8": {"Mean": 72.88, "Max": 79.31, "Min": 64.29},
        "S9": {"Mean": 70.87, "Max": 78.57, "Min": 62.07},
        "Average": {"Mean": 71.89, "Max": 80.70, "Min": 62.89}
    }
    # ## Adjust the mins for this model...
    # "GAT (T=0, H=3)": {
    #     "S1": {"Mean": 69.10, "Max": 79.31, "Min": 58.62},
    #     "S2": {"Mean": 71.18, "Max": 75.86, "Min": 62.07},
    #     "S3": {"Mean": 73.28, "Max": 78.57, "Min": 65.52},
    #     "S4": {"Mean": 65.42, "Max": 79.17, "Min": 54.17},
    #     "S5": {"Mean": 67.34, "Max": 82.76, "Min": 55.17},
    #     "S6": {"Mean": 63.58, "Max": 71.43, "Min": 55.17},
    #     "S7": {"Mean": 74.31, "Max": 86.21, "Min": 58.62},
    #     "S8": {"Mean": 71.18, "Max": 75.86, "Min": 65.52},
    #     "S9": {"Mean": 68.40, "Max": 75.86, "Min": 55.17},
    #     "Average": {"Mean": 69.31, "Max": 78.34, "Min": 58.89}
    # }
}

#%% Plots

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set font sizes globally
plt.rcParams.update({
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 14,     # Axis labels font size
    'xtick.labelsize': 12,    # X-ticks font size
    'ytick.labelsize': 12,    # Y-ticks font size
    'legend.fontsize': 12,    # Legend font size
    'figure.titlesize': 18    # Figure title font size
})

# Prepare data for the Healthy cohort with error bars
data_healthy = {
    'Model': [],
    'Mean Accuracy': [],
    'Min Accuracy': [],
    'Max Accuracy': []
}

for model, metrics in healthy_patients.items():
    data_healthy['Model'].append(model)
    data_healthy['Mean Accuracy'].append(metrics['Average']['Mean'])
    data_healthy['Min Accuracy'].append(metrics['Average']['Min'])
    data_healthy['Max Accuracy'].append(metrics['Average']['Max'])

df_healthy = pd.DataFrame(data_healthy)

# Create bar plot for Healthy cohort with error bars
plt.figure(figsize=(15, 4))
barplot = sns.barplot(x='Model', y='Mean Accuracy', data=df_healthy, palette='Blues_d', ci=None)

# Add error bars
for i, row in df_healthy.iterrows():
    barplot.errorbar(x=i, y=row['Mean Accuracy'],
                     yerr=[[row['Mean Accuracy'] - row['Min Accuracy']],
                           [row['Max Accuracy'] - row['Mean Accuracy']]], 
                     fmt='o', color='black', capsize=5)

plt.title('Model Average Accuracy for Healthy Cohort')
plt.ylabel('Mean Accuracy')
plt.xticks(rotation=30)

# Set y-axis ticks from 0 to 100 in increments of 10
plt.yticks(range(0, 101, 10))

plt.tight_layout()
plt.show()

# Prepare data for the ALS cohort with error bars
data_als = {
    'Model': [],
    'Mean Accuracy': [],
    'Min Accuracy': [],
    'Max Accuracy': []
}

for model, metrics in als_patients.items():
    data_als['Model'].append(model)
    data_als['Mean Accuracy'].append(metrics['Average']['Mean'])
    data_als['Min Accuracy'].append(metrics['Average']['Min'])
    data_als['Max Accuracy'].append(metrics['Average']['Max'])

df_als = pd.DataFrame(data_als)

# Create bar plot for ALS cohort with error bars
plt.figure(figsize=(15, 4))
barplot = sns.barplot(x='Model', y='Mean Accuracy', data=df_als, palette='Reds_d', ci=None)

# Add error bars
for i, row in df_als.iterrows():
    barplot.errorbar(x=i, y=row['Mean Accuracy'],
                     yerr=[[row['Mean Accuracy'] - row['Min Accuracy']],
                           [row['Max Accuracy'] - row['Mean Accuracy']]],
                     fmt='o', color='black', capsize=5)

plt.title('Model Average Accuracy for ALS Cohort')
plt.ylabel('Mean Accuracy')
plt.xticks(rotation=35)

# Set y-axis ticks from 0 to 100 in increments of 10
plt.yticks(range(0, 101, 10))

plt.tight_layout()
plt.show()


#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to flatten nested accuracy data for a single cohort (e.g., ALS or Healthy patients)
def flatten_data(model_dict):
    records = []
    for model, subjects in model_dict.items():
        for subject, values in subjects.items():
            if subject != 'cohort_averages':  # We exclude the cohort average entry for individual subject plots
                records.append({
                    'Subject': subject,
                    'Model': model,
                    'Mean': values['Mean']  # Assuming values is a dict with 'Mean' as a key
                })
    return pd.DataFrame(records)

# Flatten ALS and Healthy data dictionaries
als_df = flatten_data(als_patients)
healthy_df = flatten_data(healthy_patients)

# Plotting function for grouped bar plots
def plot_grouped_bar(data, title):
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=data, kind="bar",
        x="Subject", y="Mean", hue="Model", 
        palette="bright",  # Use 'bright' palette for more vibrant colors
        alpha=.8, height=6, aspect=3  # Adjusted aspect ratio for wider plot
    )
    
    # Add horizontal line at 70% accuracy
    plt.axhline(70, color='red', linestyle='--', label="70% Threshold")
    
    g.despine(left=True)
    g.set_axis_labels("Subjects", "Accuracy")
    
    # Move the legend outside the plot
    g.legend.set_title("Models")

    # Set y-axis ticks from 0 to 100 in increments of 10
    plt.yticks(range(0, 101, 10))
 
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Add extra margin for the legend

# Plot ALS subjects with updated model names
plot_grouped_bar(als_df, 'ALS Patients - Model Comparison')

# Plot Healthy subjects with existing model names
plot_grouped_bar(healthy_df, 'Healthy Patients - Model Comparison')

plt.show()



#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting function for boxplots
def plot_boxplot(data, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5, 6))
    sns.boxplot(x="Model", y="Mean", data=data, palette="pastel")  # Using pastel palette
    plt.axhline(70, color='red', linestyle='--', label="70% Threshold")  # 70% threshold line
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Models", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

# Plot boxplot for ALS subjects
plot_boxplot(als_df, 'ALS Patients - Model Average Accuracies')

# Plot boxplot for Healthy subjects
plot_boxplot(healthy_df, 'Healthy Participants - Model Average Accuracies')

plt.show()

#%% F-Test Healthy

import numpy as np
from scipy.stats import f

# Accuracy values for each model and subject (transcribed from your table)
models = {
    'FBCSP': [91.33, 56.88, 93.05, 62.83, 88.2, 58.26, 92.01, 95.85, 92.03],
    'CSP': [77.71, 61.9, 91.71, 70.76, 63.24, 66.56, 70.86, 95.76, 86.9],
    'CSSP': [71.8, 59.36, 33.77, 59.91, 52.18, 73.26, 65.91, 96.16, 93.41],
    'EEGNet': [56.61, 65.23, 61.25, 61.25, 65.23, 73.68, 71.48, 65.65, 69.43],
    'Deep ConvNet': [64.22, 69.78, 64.98, 70.42, 69.79, 69.98, 70.47, 65.54, 67.62],
    'Shallow ConvNet': [69.43, 65.63, 67.22, 65.42, 63.14, 70.48, 70.46, 67.03, 66.28],
    'Wavelet CNN': [76.67, 72, 90.98, 73.33, 83.33, 89.9, 82.67, 80, 80.1],
    'Ego CNN': [86.1, 98.9, 95.46, 66.31, 100, 72.59, 85.7, 98.3, 100],
    'CSP CNN': [90.26, 48.62, 90.97, 78.47, 91.32, 89.9, 94.8, 97.2, 94.1],
    'GCN 1': [63.19, 63.55, 66.97, 65.83, 62.86, 62.39, 64.87, 65.96, 63.87],
    'GCN 2': [65.53, 67.71, 67.09, 65.01, 67.73, 65.77, 63.42, 70.14, 62.54],
    'Our Work': [68.34, 71.51, 71.18, 71.67, 73.65, 73.33, 73.61, 72.88, 70.87]
}

# Function to perform F-test and return F-statistic and p-value
def f_test(var1, var2, n1, n2):
    f_statistic = var1 / var2 if var1 > var2 else var2 / var1
    dfn = n1 - 1  # Degrees of freedom for GAT (first sample)
    dfd = n2 - 1  # Degrees of freedom for other model (second sample)
    p_value = 1 - f.cdf(f_statistic, dfn, dfd)
    return f_statistic, p_value

# Get variance and sample size for each model
model_variances = {model: np.var(accuracies, ddof=1) for model, accuracies in models.items()}
model_samples = {model: len(accuracies) for model, accuracies in models.items()}

# Set the significance level
alpha = 0.05

# Perform pairwise F-tests comparing 'Our Work' to all other models
our_var = model_variances['Our Work']
n_our = model_samples['Our Work']

for model_name, accuracies in models.items():
    if model_name != 'Our Work':  # Compare only non-'Our Work' models
        other_var = model_variances[model_name]
        n_other = model_samples[model_name]
        
        # Perform F-test
        f_stat, p_value = f_test(our_var, other_var, n_our, n_other)
        
        # Print results with significance check
        if p_value < alpha:
            print(f"Variance of 'Our Work' is statistically significantly different from {model_name} (F-statistic: {f_stat}, p-value: {p_value})")
        else:
            print(f"No significant difference in variance between 'Our Work' and {model_name} (F-statistic: {f_stat}, p-value: {p_value})")

#%%

# Accuracy values for each model and subject (transcribed from your table)
# Data from the image
models = {
    "CSP": [51.57, 65.21, 50.64, 88.31, 82.97, 75.40, 74.76, 83.16],
    "EEGNet": [65.69, 67.06, 63.93, 66.54, 58.28, 62.58, 62.06, 61.99],
    "Deep ConvNet": [66.56, 60.12, 63.34, 66.54, 59.85, 63.83, 62.06, 63.01],
    "Shallow ConvNet": [61.82, 68.57, 58.87, 66.67, 65.31, 55.73, 65.00, 68.00],
    "GCN 1": [69.37, 77.59, 66.64, 66.54, 61.31, 67.19, 65.00, 61.67],
    "GCN 2": [65.94, 63.84, 62.88, 64.80, 63.12, 65.00, 65.00, 61.67],
    "Our Work": [70.31, 77.05, 71.36, 74.19, 75.62, 75.31, 70.67, 78.00],
}

# Function to perform F-test and return F-statistic and p-value
def f_test(var1, var2, n1, n2):
    f_statistic = var1 / var2 if var1 > var2 else var2 / var1
    dfn = n1 - 1  # Degrees of freedom for GAT (first sample)
    dfd = n2 - 1  # Degrees of freedom for other model (second sample)
    p_value = 1 - f.cdf(f_statistic, dfn, dfd)
    return f_statistic, p_value

# Get variance and sample size for each model
model_variances = {model: np.var(accuracies, ddof=1) for model, accuracies in models.items()}
model_samples = {model: len(accuracies) for model, accuracies in models.items()}

# Set the significance level
alpha = 0.05

# Perform pairwise F-tests comparing 'Our Work' to all other models
our_var = model_variances['Our Work']
n_our = model_samples['Our Work']

for model_name, accuracies in models.items():
    if model_name != 'Our Work':  # Compare only non-'Our Work' models
        other_var = model_variances[model_name]
        n_other = model_samples[model_name]
        
        # Perform F-test
        f_stat, p_value = f_test(our_var, other_var, n_our, n_other)
        
        # Print results with significance check
        if p_value < alpha:
            print(f"Variance of 'Our Work' is statistically significantly different from {model_name} (F-statistic: {f_stat}, p-value: {p_value})")
        else:
            print(f"No significant difference in variance between 'Our Work' and {model_name} (F-statistic: {f_stat}, p-value: {p_value})")


