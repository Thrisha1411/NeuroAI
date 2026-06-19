import numpy as np

def compute_connectivity_features(data):
    """
    Compute channel-to-channel connectivity (Pearson correlation).
    
    Args:
        data (np.array): EEG data (n_channels, n_times)
        
    Returns:
        np.array: Flattened upper triangle of correlation matrix.
    """
    # Correlation matrix (n_channels, n_channels)
    corr_matrix = np.corrcoef(data)
    
    # Extract upper triangle indices (excluding diagonal)
    # k=1 offset excludes diagonal
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    
    features = corr_matrix[upper_tri_indices]
    
    return features
