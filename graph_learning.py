from load_data import *
from augmentations import *
import numpy as np
from scipy.signal import coherence
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import welch, csd
from joblib import Parallel, delayed
import time
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import coherence
from load_data import *
from augmentations import *
import numpy as np
from scipy.signal import coherence, welch
from joblib import Parallel, delayed
import time




fs = 0.5    
nperseg = 200

def pad_psd(psd, target_size=101):
    current_size = len(psd)
    if current_size == target_size:
        return psd  # Do nothing if already 101
    elif current_size < target_size:
        return np.pad(psd, (0, target_size - current_size), mode='constant')  # Append zeros
    else:
        return psd[:target_size]  # If larger than 101 (unlikely), truncate


    
def compute_fft(time_series_data, fs):
    """
    Compute FFT and return frequencies and FFT data.

    Args:
        time_series_data (ndarray): 1D time-domain signal.
        fs (int): Sampling frequency in Hz.

    Returns:
        freqs (ndarray): Frequency bins (only positive frequencies).
        fft_data (ndarray): Complex FFT values (for further processing like CSD).
    """
    N = len(time_series_data)  # Signal length
    fft_data = np.fft.fft(time_series_data) / N  # Compute FFT and normalize
    freqs = np.fft.fftfreq(N, d=1/fs)  # Frequency bins

    # Return only the positive half of the spectrum (for real signals)
    freqs = freqs[:N // 2]
    fft_data = fft_data[:N // 2]

    return freqs, fft_data


def compute_coherence_matrix_FFT(fft_data):
    """
    Compute coherence matrix from precomputed FFT spectra.

    Args:
        fft_data (ndarray): Complex-valued FFT data of shape (N, F),
                            where N is the number of signals and F is the number of frequency bins.

    Returns:
        coherence_matrix (ndarray): N × N coherence matrix.
    """
    N, F = fft_data.shape  # N signals, F frequency bins
    coherence_matrix = np.zeros((N, N))  # Initialize coherence matrix

    for i in range(N):
        for j in range(i + 1, N):  # Compute only upper triangle
            Pxx = np.abs(fft_data[i]) ** 2  # PSD of signal i
            Pyy = np.abs(fft_data[j]) ** 2  # PSD of signal j
            Pxy = fft_data[i] * np.conj(fft_data[j])  # CSD of (i, j)

            # Compute coherence
            Cxy_f = np.abs(Pxy) ** 2 / (Pxx * Pyy)
            coherence_value = np.mean(Cxy_f)  # Average across all frequencies

            # Fill coherence matrix symmetrically
            coherence_matrix[i, j] = coherence_value
            coherence_matrix[j, i] = coherence_value

    # Apply threshold: Set values < 0.3 to 0 in a single operation
    coherence_matrix[coherence_matrix < 0.3] = 0

    # Remove self-coherence
    np.fill_diagonal(coherence_matrix, 0)

    return coherence_matrix



def compute_psd(time_series_data, fs, nperseg):
    """
    Compute the Power Spectral Density (PSD) for a time series.

    Args:
        time_series_data (ndarray): 1D time-domain signal.
        fs (int): Sampling frequency in Hz.
        nperseg (int): Segment length for Welch’s method.

    Returns:
        freqs (ndarray): Frequency bins.
        psd (ndarray): Power Spectral Density values.
    """
    # Ensure nperseg is valid
    nperseg = min(len(time_series_data), nperseg)
    if nperseg < len(time_series_data) // 2:
        nperseg = len(time_series_data) // 2  # Prevent excessive shrinking

    freqs, psd = welch(time_series_data, fs=fs, nperseg=nperseg)
    return freqs, psd


def dataset_time(X, Y, pert=False):
    config = Config_augmentation()  
    X_featgraph, X_adjgraph = [], []
    for i in range(len(Y)):
        signals = X[i]
        if pert == True: 
            signals = DataTransform_TD_bank(signals, config )
        window_data = np.corrcoef(signals)
        knn_graph = compute_KNN_graph(window_data)

        X_featgraph.append(window_data)
        X_adjgraph.append(knn_graph)
        
    return X_featgraph, X_adjgraph, Y



def dataset_freq(X, Y, fs, nperseg, pert=False):
    config = Config_augmentation()  # Ensure this is defined elsewhere
    X_featgraph, X_adjgraph = [], []

    total_start_time = time.time()  # Start timing

    for i in range(len(Y)):  # Loop over each subject
        signals = X[i] 
        fft_data = np.array([compute_fft(signal, fs=fs)[1] for signal in signals])
        if pert == True: 
            fft_data = DataTransform_FD(fft_data, config)  # Apply transformation         
            if isinstance(fft_data, torch.Tensor):
                fft_data = fft_data.cpu().detach().numpy()
        coherence_matrix = compute_coherence_matrix_FFT(fft_data)
        #psd_data = np.array([compute_psd(signal, fs=fs, nperseg=nperseg)[1] for signal in signals])
        psd_data = np.array([pad_psd(compute_psd(signal, fs=fs, nperseg=nperseg)[1], 101) for signal in signals])
        
        X_featgraph.append(psd_data)
        X_adjgraph.append(coherence_matrix)
        print(f"Coherence matrix calculation done for subject {i+1}/{len(Y)}")
    total_elapsed_time = time.time() - total_start_time  # Compute total time taken
    print(f"Total time taken for processing {len(Y)} subjects: {total_elapsed_time:.2f} seconds")

    return X_featgraph, X_adjgraph, Y


