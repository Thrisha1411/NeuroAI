import numpy as np
from scipy.signal import butter, lfilter, iirnotch

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=128, order=5):
    """
    Apply bandpass filter to EEG data.
    
    Args:
        data (np.ndarray): EEG signal array (channels, time).
        lowcut (float): Low frequency cutoff.
        highcut (float): High frequency cutoff.
        fs (int): Sampling frequency.
        order (int): Filter order.
        
    Returns:
        filtered_data (np.ndarray): Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y

def notch_filter(data, freq=50.0, fs=128, quality=30.0):
    """
    Apply notch filter to remove power line noise (50Hz/60Hz).
    
    Args:
        data (np.ndarray): EEG signal array.
        freq (float): Frequency to remove.
        fs (int): Sampling frequency.
        quality (float): Quality factor.
        
    Returns:
        filtered_data (np.ndarray): Signal with noise removed.
    """
    b, a = iirnotch(freq, quality, fs)
    y = lfilter(b, a, data, axis=-1)
    return y
