import numpy as np

def compute_connectivity_features(data):
    """
    Computes Pearson correlation matrix between channels.
    Flattened upper triangle is returned to avoid redundancy.
    
    Args:
        data (np.ndarray): EEG segment (channels, samples).
        
    Returns:
        connectivity (np.ndarray): Flattened correlation coefficients.
    """
    # Pearson correlation
    corr_matrix = np.corrcoef(data)
    
    # Get upper triangle indices (excluding diagonal) to avoid duplicates and self-correlation
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    
    return corr_matrix[upper_tri_indices]
