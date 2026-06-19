import numpy as np
from scipy.signal import welch

def compute_band_power(data, fs=128):
    """
    Computes absolute band power for Delta, Theta, Alpha, Beta, Gamma bands.
    Args:
        data (np.ndarray): EEG segment (channels, samples).
        fs (int): Sampling frequency.
        
    Returns:
        band_powers (np.ndarray): Array of shape (channels, 5) containing power in 5 bands.
    """
    # Define frequency bands
    # Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-45)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    
    n_channels = data.shape[0]
    features = np.zeros((n_channels, len(bands)))
    
    for ch in range(n_channels):
        freqs, psd = welch(data[ch], fs, nperseg=fs*2) # 2s window for welch
        
        for i, (low, high) in enumerate(bands):
            # Find indices of freq range
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Integration (simulated by mean for simplicity or simps)
            # Standard is mean or sum. Let's use mean power.
            if np.any(idx_band):
                features[ch, i] = float(np.mean(psd[idx_band]))  # type: ignore
            else:
                features[ch, i] = 0.0
                
    return features
