from load_data import *

original_BOLD =  np.load(f'/home/vipul/{dataset_name}/{atlas_name}/X.npz')
BOLD_atlas, label = get_atlas_data(original_BOLD)


# BOLD signal for the atlas, labels
def select_time(X, Y, start, end, region=True):
    X_atlas=[]   
    for i in range(len(Y)):
        if region == True:
            bold_matrix = X[i][start:end,:] #RxR
            X_atlas.append(bold_matrix)
        else:
            bold_matrix = np.transpose(X[i][start:end,:]) #TxT
    return X_atlas, Y

X_atlas, labels = select_time(X_loaded, Y_loaded, start, end, region=True)



# creating frequency graph
def compute_coherence_matrix_PSD(signals, fs, nperseg=256):
    """
    Compute the coherence matrix for a set of signals.
    
    Parameters:
        signals (ndarray): 2D array where each row is a signal.
        fs (int): Sampling frequency.
        nperseg (int): Length of each segment for Welch's method.
    
    Returns:
        coherence_matrix (ndarray): NxN coherence matrix.
    """
    num_signals = signals.shape[0]
    coherence_matrix = np.zeros((num_signals, num_signals))

    for i in range(num_signals):
        for j in range(i, num_signals):
            # Compute PSD for signals i and j
            f, Pxx = welch(signals[i], fs=fs, nperseg=nperseg)
            _, Pyy = welch(signals[j], fs=fs, nperseg=nperseg)
            _, Pxy = csd(signals[i], signals[j], fs=fs, nperseg=nperseg)

            # Compute coherence: Cxy(f) = |Pxy(f)|^2 / (Pxx(f) * Pyy(f))
            Cxy = (np.abs(Pxy) ** 2) / (Pxx * Pyy)
            coherence_value = np.mean(Cxy)  # Average coherence over frequencies

            # Store in the matrix (symmetric)
            coherence_matrix[i, j] = coherence_value
            coherence_matrix[j, i] = coherence_value

    # Remove self-coherence (diagonal elements)
    np.fill_diagonal(coherence_matrix, 0)
    
    return coherence_matrix



def compute_psd(time_series_data, fs, nperseg):
    """Compute the Power Spectral Density (PSD) for a time series."""
    # Dynamically adjust nperseg based on data length
    nperseg = min(len(time_series_data), nperseg)
    freqs, psd = welch(time_series_data, fs=fs, nperseg=nperseg)
    return freqs, psd




def dataset_time(X, Y, start, end, pert=False):
    config = Config()  # Instantiate the config with default settings
    X_featgraph, X_adjgraph = [], []
    
    for i in range(len(Y)):
        data_slice = X[i][start:end, :]
        if pert == True:
            data_slice = DataTransform_TD_bank(data_slice, config)
        
        window_data = np.corrcoef(data_slice)
        knn_graph = compute_KNN_graph(window_data)
        
        X_featgraph.append(window_data)
        X_adjgraph.append(knn_graph)



def dataset_freq(X, Y, fs, nperseg=256, pert=False):
    config = Config()  # Instantiate the config with default settings
    X_featgraph, X_adjgraph = [], []
    
    for i in range(len(Y)):
        signals = X[i]
        if pert == True:
            signals = DataTransform_FD(signals, config)
        
        coherence_matrix = compute_coherence_matrix_PSD(signals, fs, nperseg)
        psd_data = np.array([compute_psd(signal, fs, nperseg)[1] for signal in signals])
        
        X_featgraph.append(psd_data)
        X_adjgraph.append(coherence_matrix)
    
    return X_featgraph, X_adjgraph, Y        
    
    return X_featgraph, X_adjgraph, Y



def dataset_freq_random(X, Y, fs, nperseg=256, pert=False):
    X_featgraph, X_adjgraph = [], []
    
    for i in range(len(Y)):
        signals = X[i]
        if pert == True:
            signals = random_frequency_perturbation_one_hot(signals)
        
        coherence_matrix = compute_coherence_matrix_PSD(signals, fs, nperseg)
        psd_data = np.array([compute_psd(signal, fs, nperseg)[1] for signal in signals])
        
        X_featgraph.append(psd_data)
        X_adjgraph.append(coherence_matrix)
    
    return X_featgraph, X_adjgraph, Y

# creating time graph
X_featgraph_time, X_adjgraph_time, labels = dataset_time(BOLD_atlas, label, start, end, pert=False)
dataset_time = to_tensor(X_featgraph_time, X_adjgraph_time, labels)
print('1st X_featgraph is:', X_featgraph[0].shape)
print('1st X_adjgraph is:', X_adjgraph[0].shape)
print('1st dataset_time is:', dataset_time[0].shape)

#creating freq graph
X_featgraph_freq, X_adjgraph_freq, labels_freq = dataset_freq(BOLD_atlas, label, fs, nperseg, pert=True)
dataset_freq = to_tensor(X_featgraph_freq, X_adjgraph_freq, labels)
print('1st X_featgraph is:', X_featgraph_freq[0].shape)
print('1st X_adjgraph is:', X_adjgraph_freq[0].shape)
print('1st dataset_time is:', dataset_freq[0].shape)
