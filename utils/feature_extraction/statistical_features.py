import numpy as np

def compute_statistical_features(data):
    """
    Computes statistical features for each channel.
    Features: Mean, Std, Var, Min, Max.
    
    Args:
        data (np.ndarray): EEG segment (channels, samples).
        
    Returns:
        stats (np.ndarray): Array of shape (channels, 5).
    """
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    var = np.var(data, axis=1)
    min_val = np.min(data, axis=1)
    max_val = np.max(data, axis=1)
    
    return np.stack([mean, std, var, min_val, max_val], axis=1)
