import torch
import numpy as np
import os
from collections import Counter
from utils.preprocessing.filters import apply_bandpass_filter, apply_notch_filter
from utils.preprocessing.segmentation import segment_signal, normalize_signal
from models.hybrid_model import HybridModel

class Predictor:
    def __init__(self, model_path, device='cuda', config=None):
        self.device = device
        self.config = config
        self.model = self._load_model(model_path)
        
    def _load_model(self, path):
         # Assuming model params from config or defaults
         input_channels = 32 # Data loader default
         # Input samples depends on window size. 2s * 128Hz = 256
         input_samples = 256
         
         model = HybridModel(input_channels=input_channels, input_samples=input_samples)
         try:
            model.load_state_dict(torch.load(path, map_location=self.device))
         except Exception as e:
            print(f"Error loading model weights: {e}. Using random weights for demo.")
            
         model.to(self.device)
         model.eval()
         return model
         
    def preprocess_input(self, data, fs=128):
        # 1. Filter
        filtered = apply_bandpass_filter(data, 0.5, 45, fs)
        filtered = apply_notch_filter(filtered, 50, fs) # Assuming 50Hz mains
        
        # 2. Segment
        # Window 2s, overlap 0.5
        segments = segment_signal(filtered, window_size_sec=2, overlap=0.5, fs=fs)
        
        # 3. Normalize
        norm_segments = normalize_signal(segments, method='zscore')
        
        return norm_segments
        
    def predict(self, raw_data, fs=128):
        """
        Predict emotion and focus from raw EEG data (n_channels, n_timepoints).
        Uses majority voting across windows.
        """
        segments = self.preprocess_input(raw_data, fs) # (n_seg, n_ch, n_time)
        
        if len(segments) == 0:
            return "Unknown", "Unknown", {}
            
        tensor_in = torch.FloatTensor(segments).to(self.device)
        
        with torch.no_grad():
            out_em, out_fo = self.model(tensor_in)
            
            _, pred_em = torch.max(out_em, 1) # (n_seg,)
            _, pred_fo = torch.max(out_fo, 1) # (n_seg,)
            
        # Majority Voting
        em_counts = Counter(pred_em.cpu().numpy())
        fo_counts = Counter(pred_fo.cpu().numpy())
        
        final_em = em_counts.most_common(1)[0][0]
        final_fo = fo_counts.most_common(1)[0][0]
        
        # Map to labels
        emotions = {0: 'Happy', 1: 'Calm', 2: 'Sad', 3: 'Angry'}
        focus_states = {0: 'Distracted', 1: 'Focused'}
        
        return emotions.get(final_em, 'Unknown'), focus_states.get(final_fo, 'Unknown'), {
            'emotion_dist': {k: v for k, v in em_counts.items()},
            'focus_dist': {k: v for k, v in fo_counts.items()},
            'segments_count': len(segments)
        }
