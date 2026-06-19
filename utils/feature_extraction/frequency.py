import numpy as np
from scipy.signal import welch

def compute_band_power(data, fs, bands):
    """
    Compute relative band power for specified frequency bands using Welch's method.
    
    Args:
        data (np.array): Segmented EEG data (n_channels, n_times)
        fs (int): Sampling rate
        bands (dict): Dictionary of band names and (low, high) tuples
        
    Returns:
        np.array: Band powers of shape (n_channels, n_bands)
    """
    n_channels, n_times = data.shape
    # Compute PSD using Welch's method
    freqs, psd = welch(data, fs, nperseg=min(n_times, fs*2))
    
    # Total power for normalization (relative power)
    total_power = np.sum(psd, axis=-1, keepdims=True)
    
    features = []
    
    # Order bands to ensure consistent feature vector
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for band in band_names:
        if band in bands:
            low, high = bands[band]
            # Find closest indices
            idx_min = int(np.argmax(freqs >= low))
            idx_max = int(np.argmax(freqs >= high))
            if idx_max == 0: idx_max = len(freqs) - 1
            
            band_power = np.sum(psd[:, idx_min:idx_max], axis=-1)  # type: ignore
            # Relative power
            rel_power = band_power / (total_power.flatten() + 1e-8)
            features.append(rel_power)
            
    # Stack features: (n_bands, n_channels) -> (n_channels, n_bands)
    return np.column_stack(features)
