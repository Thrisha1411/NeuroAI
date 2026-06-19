import torch
import numpy as np
import os
import logging
from data.loader import load_deap_raw
from utils.preprocessing.filters import bandpass_filter, notch_filter
from utils.preprocessing.segmentation import segment_signal, normalize_data
from utils.preprocessing.artifacts import z_score_rejection
from models.hybrid_model import HybridModel
from utils.logger import get_session_logger

# Initialize logger
logger = get_session_logger()

def predict_eeg(file_path, model_path='models/checkpoints/best_model.pth', device='cpu'):
    """
    Runs inference on a raw EEG file.
    
    Args:
        file_path (str): Path to raw EEG file.
        model_path (str): Path to saved model weights.
        device (str): 'cpu' or 'cuda'.
        
    Returns:
        results (dict): Predictions and distribution.
            - emotion_mapped: Majority vote emotion.
            - focus_mapped: Majority vote focus.
            - em_probs: Mean probability distribution for emotion.
            - fo_probs: Mean probability distribution for focus.
    """
    # 1. Load Data
    # Support both DEAP/SEED logic or generic. Assuming DEAP/MNE BDF for demo.
    try:
        raw = load_deap_raw(file_path)
        # extracting data from MNE Raw object
        if hasattr(raw, 'get_data'):
            data = raw.get_data() # (Channels, Time)
            fs = int(raw.info['sfreq'])
        else:
            # Assume it's numpy or dict from loader
            if isinstance(raw, dict):
                # Try to find data key
                keys = [k for k in raw.keys() if 'data' in k]
                if keys:
                    data = raw[keys[0]]
                    if data.ndim == 3: # (trials, ch, time)
                        data = data.reshape(-1, data.shape[-1]) # flatten trials
                else:
                     raise ValueError("Could not find data in loaded dictionary.")
                fs = 128 # Default assumption
            else:
                data = raw
                fs = 128

    except Exception as e:
        return {"error": str(e)}

    # 2. Preprocessing
    logger.info(f"Preprocessing file: {file_path}")
    
    # Filter
    data = bandpass_filter(data, fs=fs)
    data = notch_filter(data, fs=fs)
    
    # Artifact Removal (Simple)
    data = z_score_rejection(data)
    
    # Segment
    # Window size should match training. Assuming 2s.
    segments = segment_signal(data, window_size=2.0, overlap=0.5, fs=fs)
    
    if len(segments) == 0:
        logger.error("Signal too short to segment.")
        return {"error": "Signal too short to segment."}
        
    # Normalize
    segments = normalize_data(segments) # (N_seg, Channels, Time)
    logger.info(f"Generated {len(segments)} segments.")
    
    # 3. Model Inference
    # Model config must match training.
    # Assuming 32 channels. If mismatch, need to handle.
    n_channels = segments.shape[1]
    n_samples = segments.shape[2]
    
    model = HybridModel(input_channels=n_channels, input_samples=n_samples)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        # If no model, return dummy or error? 
        # For demo purposes, we proceed with random weights but warn.
        logger.warning("Model checkpoint not found. Using random weights.")
    
    model.to(device)
    model.eval()
    
    inputs = torch.FloatTensor(segments).to(device)
    
    with torch.no_grad():
        em_logits, fo_logits = model(inputs)
        
        # Probabilities
        em_probs = torch.softmax(em_logits, dim=1).cpu().numpy()
        fo_probs = torch.softmax(fo_logits, dim=1).cpu().numpy()
        
        # Predictions
        em_preds = np.argmax(em_probs, axis=1)
        fo_preds = np.argmax(fo_probs, axis=1)
        
    # 4. Aggregation (Majority Vote)
    from scipy.stats import mode
    
    final_em = mode(em_preds, keepdims=True)[0][0]
    final_fo = mode(fo_preds, keepdims=True)[0][0]
    
    avg_em_probs = np.mean(em_probs, axis=0)
    avg_fo_probs = np.mean(fo_probs, axis=0)
    
    logger.info(f"Prediction Complete. Emotion: {final_em}, Focus: {final_fo}")
    
    return {
        'emotion_label': int(final_em),
        'focus_label': int(final_fo),
        'emotion_probs': avg_em_probs,
        'focus_probs': avg_fo_probs,
        'predictions_per_segment': {
            'emotion': em_preds,
            'focus': fo_preds
        }
    }
