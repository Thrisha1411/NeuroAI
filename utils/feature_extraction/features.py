import numpy as np
from .frequency_features import compute_band_power
from .statistical_features import compute_statistical_features
from .connectivity_features import compute_connectivity_features

def extract_features(data, fs=128):
    """
    Extracts all features for a single EEG segment.
    
    Args:
        data (np.ndarray): EEG segment (channels, samples).
        fs (int): Sampling frequency.
        
    Returns:
        feature_vector (np.ndarray): 1D array of concatenated features.
    """
    # 1. Frequency Features (Channels x 5 Bands) -> Flatten
    freq_feats = compute_band_power(data, fs).flatten()
    
    # 2. Statistical Features (Channels x 5 Stats) -> Flatten
    stat_feats = compute_statistical_features(data).flatten()
    
    # 3. Connectivity Features (Flattened Upper Triangle)
    conn_feats = compute_connectivity_features(data)
    
    # Concatenate all
    return np.concatenate([freq_feats, stat_feats, conn_feats])
