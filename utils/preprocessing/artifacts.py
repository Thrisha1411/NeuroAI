import numpy as np
from sklearn.decomposition import FastICA

def remove_artifacts_ica(data, n_components=None):
    """
    Apply ICA to remove artifacts (blind source separation).
    Note: For a production system, visual inspection or automatic identification 
    of artifact components (e.g. via correlation with EOG) is preferred.
    Here we implement a placeholder for ICA decomposition.
    
    Args:
        data (np.array): EEG data of shape (n_channels, n_samples)
        
    Returns:
        np.array: Cleaned data
    """
    # Transpose to (n_samples, n_channels) for sklearn
    data_t = data.T
    
    if n_components is None:
        n_components = data.shape[0]
        
    ica = FastICA(n_components=n_components, random_state=42)
    components = ica.fit_transform(data_t)
    
    # TODO: Implement automatic component rejection logic here.
    # For now, we return the reconstructed signal (identity op)
    # in a real scenario, we would zero out artifact components.
    
    restored = ica.inverse_transform(components)
    return restored.T

def z_score_rejection(data, threshold=3.0):
    """
    Reject segments or channels with high z-scores.
    Alternative: Clip values > threshold.
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    z_scores = (data - mean) / (std + 1e-8)
    
    # Simple clipping for now
    clean_data = np.clip(data, mean - threshold*std, mean + threshold*std)
    return clean_data
