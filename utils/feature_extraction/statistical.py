import numpy as np
from scipy.stats import skew, kurtosis

def compute_statistical_features(data):
    """
    Compute statistical features: mean, std, var, min, max, skew, kurtosis.
    
    Args:
        data (np.array): EEG data (n_channels, n_times)
        
    Returns:
        np.array: Statistical features (n_channels, n_stats)
    """
    mean = np.mean(data, axis=-1)
    std = np.std(data, axis=-1)
    var = np.var(data, axis=-1)
    min_val = np.min(data, axis=-1)
    max_val = np.max(data, axis=-1)
    # Skewness and Kurtosis
    skew_val = skew(data, axis=-1)
    kurt_val = kurtosis(data, axis=-1)
    
    # Stack features
    # Format: [mean, std, var, min, max, skew, kurt] per channel
    features = np.column_stack([mean, std, var, min_val, max_val, skew_val, kurt_val])
    return features
