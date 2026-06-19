import numpy as np

def segment_signal(data, window_size=2.0, overlap=0.0, fs=128):
    """
    Segments EEG data into fixed-length windows.
    
    Args:
        data (np.ndarray): EEG data array (channels, time).
        window_size (float): Window size in seconds.
        overlap (float): Overlap between windows (0.0 to 1.0).
        fs (int): Sampling frequency.
        
    Returns:
        segments (np.ndarray): Segmented data array (num_segments, channels, samples_per_window).
    """
    n_channels, n_samples = data.shape
    window_samples = int(window_size * fs)
    step_samples = int(window_samples * (1 - overlap))
    
    segments = []
    
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        segment = data[:, start:end]
        segments.append(segment)
        
    if not segments:
        return np.array([])
        
    return np.array(segments)

def normalize_data(data):
    """
    Z-score normalization per channel.
    
    Args:
        data (np.ndarray): Input data (..., channels, time).
        
    Returns:
        normalized_data (np.ndarray): Normalized data.
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return (data - mean) / (std + 1e-8)
