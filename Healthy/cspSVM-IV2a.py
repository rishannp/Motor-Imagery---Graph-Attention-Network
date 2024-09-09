import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score
from mne.decoding import CSP

# Define the directory containing your .mat files
directory = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Data'

# Create a dictionary to store the concatenated data and labels by subject
data_by_subject = {}

# Define the list of subject IDs you're interested in
subject_ids = list(range(1, 10))  # Subject IDs 1 through 9

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.mat') and filename.startswith('A') and filename[1:-4].isdigit():
        subject_number_str = filename[1:-4]
        
        if int(subject_number_str) in subject_ids:
            print(f"Processing file: {filename}")
            file_path = os.path.join(directory, filename)
            mat_data = loadmat(file_path)
            
            subject_variable_name = 'subject'
            if subject_variable_name in mat_data:
                void_array = mat_data[subject_variable_name]
                
                subject_id = f'S{subject_number_str}'
                if subject_id not in data_by_subject:
                    data_by_subject[subject_id] = {'L': [], 'R': []}
                
                for item in void_array[0]:
                    L_data = item['L']
                    R_data = item['R']
                    
                    data_by_subject[subject_id]['L'].append(L_data)
                    data_by_subject[subject_id]['R'].append(R_data)

fs = 250

def bandpass_filter_trials(data_split, low_freq, high_freq, sfreq):
    filtered_data_split = {}
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')

    for subject in data_split:
        subject_data = data_split[subject]
        filtered_subject_data = {}

        for direction in ['L', 'R']:
            trials = subject_data[direction]
            filtered_trials = []

            for trial_data in trials:
                trial_data = np.array(trial_data, dtype=np.float64)  # Ensure float64
                filtered_trial_data = np.zeros_like(trial_data)
                for ch in range(trial_data.shape[1]):
                    filtered_trial_data[:, ch] = filtfilt(b, a, trial_data[:, ch])
                filtered_trials.append(filtered_trial_data)

            filtered_subject_data[direction] = filtered_trials

        filtered_data_split[subject] = filtered_subject_data

    return filtered_data_split

filtered_data_split = bandpass_filter_trials(data_by_subject, low_freq=8, high_freq=30, sfreq=fs)
del data_by_subject

def combine_L_R_data(data):
    for subject in data.keys():
        if 'L' in data[subject] and 'R' in data[subject]:
            L_data = np.array(data[subject]['L'], dtype=np.float64)
            R_data = np.array(data[subject]['R'], dtype=np.float64)
            num_L_trials = L_data.shape[0]
            num_R_trials = R_data.shape[0]
            min_trials = min(num_L_trials, num_R_trials)

            combined_data = []
            label_vector = []

            for i in range(min_trials):
                combined_data.append(L_data[i])
                combined_data.append(R_data[i])
                label_vector.append(0)
                label_vector.append(1)

            combined_data = np.array(combined_data, dtype=np.float64)
            combined_data = np.transpose(combined_data, (0, 2, 1))  # [trials, samples, channels]
            label_vector = np.array(label_vector, dtype=np.float64)

            data[subject] = {'data': combined_data, 'label': label_vector}

    return data

data_combined = combine_L_R_data(filtered_data_split)

def train_and_test_csp(train_data, test_data, n_components=10):
    csp = CSP(n_components=n_components, log=True, norm_trace=True, component_order='mutual_info')
    
    X_train = np.array(train_data['data'], dtype=np.float64)
    y_train = np.array(train_data['labels'], dtype=np.float64)
    
    X_test = np.array(test_data['data'], dtype=np.float64)
    y_test = np.array(test_data['labels'], dtype=np.float64)
    
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    try:
        csp.fit(X_train, y_train)
        X_train_transformed = csp.transform(X_train)
        X_test_transformed = csp.transform(X_test)
        
        scaler = StandardScaler()
        X_train_transformed = scaler.fit_transform(X_train_transformed)
        X_test_transformed = scaler.transform(X_test_transformed)
        
        classifier = SVC(class_weight='balanced')
        classifier.fit(X_train_transformed, y_train)
        
        y_pred = classifier.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        return accuracy, kappa
    
    except Exception as e:
        print(f"Error during CSP processing: {e}")
        return None, None

def cross_validate(data, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = {}
    kappas = {}

    for subject, subject_data in data.items():
        X = np.array(subject_data['data'], dtype=np.float64)
        y = np.array(subject_data['label'], dtype=np.float64)
        
        accuracies[subject] = []
        kappas[subject] = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_data = {'data': X_train, 'labels': y_train}
            test_data = {'data': X_test, 'labels': y_test}
            
            accuracy, kappa = train_and_test_csp(train_data, test_data)
            if accuracy is not None:
                accuracies[subject].append(accuracy)
                kappas[subject].append(kappa)
        
        mean_accuracy = np.mean(accuracies[subject]) if accuracies[subject] else 0
        mean_kappa = np.mean(kappas[subject]) if kappas[subject] else 0
        print(f'Subject: {subject}, Mean Accuracy: {mean_accuracy:.4f}, Mean Kappa: {mean_kappa:.4f}')
    
    return accuracies, kappas

accuracies, kappas = cross_validate(data_combined, n_folds=10)

def compute_mean_accuracies(accuracies, kappas):
    subject_stats = {}
    
    # Calculate mean, min, and max accuracy and kappa for each subject
    for subject, acc_list in accuracies.items():
        subject_mean_acc = np.mean(acc_list)
        subject_min_acc = np.min(acc_list)
        subject_max_acc = np.max(acc_list)

        subject_mean_kappa = np.mean(kappas[subject])
        subject_min_kappa = np.min(kappas[subject])
        subject_max_kappa = np.max(kappas[subject])

        subject_stats[subject] = {
            'mean_accuracy': subject_mean_acc,
            'min_accuracy': subject_min_acc,
            'max_accuracy': subject_max_acc,
            'mean_kappa': subject_mean_kappa,
            'min_kappa': subject_min_kappa,
            'max_kappa': subject_max_kappa
        }
        print(f"Subject: {subject}")
        print(f"  Mean accuracy: {subject_mean_acc:.4f}")
        print(f"  Min accuracy: {subject_min_acc:.4f}")
        print(f"  Max accuracy: {subject_max_acc:.4f}")
        print(f"  Mean kappa: {subject_mean_kappa:.4f}")
        print(f"  Min kappa: {subject_min_kappa:.4f}")
        print(f"  Max kappa: {subject_max_kappa:.4f}")
    
    # Calculate the mean of min and max values across all subjects
    min_accuracies = [stats['min_accuracy'] for stats in subject_stats.values()]
    max_accuracies = [stats['max_accuracy'] for stats in subject_stats.values()]
    mean_accuracies = [stats['mean_accuracy'] for stats in subject_stats.values()]

    min_kappas = [stats['min_kappa'] for stats in subject_stats.values()]
    max_kappas = [stats['max_kappa'] for stats in subject_stats.values()]
    mean_kappas = [stats['mean_kappa'] for stats in subject_stats.values()]

    mean_min_acc = np.mean(min_accuracies)
    mean_max_acc = np.mean(max_accuracies)
    mean_cohort_acc = np.mean(mean_accuracies)
    
    mean_min_kappa = np.mean(min_kappas)
    mean_max_kappa = np.mean(max_kappas)
    mean_cohort_kappa = np.mean(mean_kappas)
    
    print(f"\nCohort mean accuracy: {mean_cohort_acc:.4f}")
    print(f"Cohort mean of min accuracies: {mean_min_acc:.4f}")
    print(f"Cohort mean of max accuracies: {mean_max_acc:.4f}")
    print(f"\nCohort mean kappa: {mean_cohort_kappa:.4f}")
    print(f"Cohort mean of min kappa: {mean_min_kappa:.4f}")
    print(f"Cohort mean of max kappa: {mean_max_kappa:.4f}")
    
    return subject_stats, mean_cohort_acc, mean_min_acc, mean_max_acc, mean_cohort_kappa, mean_min_kappa, mean_max_kappa

# Example: computing the mean accuracies, min, max, and Cohen's kappa
subject_stats, mean_cohort_acc, mean_min_acc, mean_max_acc, mean_cohort_kappa, mean_min_kappa, mean_max_kappa = compute_mean_accuracies(accuracies, kappas)
