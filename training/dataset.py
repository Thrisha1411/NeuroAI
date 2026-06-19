import torch
from torch.utils.data import Dataset
import numpy as np

class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG Data.
    Expects pre-segmented data and labels.
    """
    def __init__(self, data, emotion_labels, focus_labels, transform=None):
        """
        Args:
            data (np.ndarray): Shape (Samples, Channels, Time)
            emotion_labels (np.ndarray): Shape (Samples,)
            focus_labels (np.ndarray): Shape (Samples,)
            transform (callable, optional): Transform to apply to data.
        """
        self.data = torch.FloatTensor(data)
        self.emotion_labels = torch.LongTensor(emotion_labels)
        self.focus_labels = torch.FloatTensor(focus_labels).unsqueeze(1) # For BCEWithLogits if binary
        # Or LongTensor if CrossEntropy
        # Prompt says "CrossEntropy focus loss" and "Binary classification".
        # If CrossEntropy, must be LongTensor (0, 1).
        # Let's use CrossEntropy for consistency as requested ("CrossEntropy focus loss").
        self.focus_labels = torch.LongTensor(focus_labels)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        emotion = self.emotion_labels[idx]
        focus = self.focus_labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, emotion, focus
