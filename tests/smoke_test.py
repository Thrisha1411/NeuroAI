import sys
import os
import torch
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    print("Testing imports...")
    try:
        from data.loader import load_deap_raw
        from utils.preprocessing.filters import bandpass_filter
        from utils.preprocessing.segmentation import segment_signal
        from utils.feature_extraction.features import extract_features
        from models.hybrid_model import HybridModel
        from training.trainer import train_model
        from utils.metrics import calculate_metrics
        # Streamlit pages are scripts, not modules to import usually, but logic is in inference
        from inference.predict import predict_eeg
        print("  -> Imports successful.")
    except Exception as e:
        print(f"  -> Import failed: {e}")
        sys.exit(1)

def test_model_instantiation():
    print("Testing model instantiation...")
    try:
        from models.hybrid_model import HybridModel
        model = HybridModel(input_channels=32, input_samples=256) # 2s at 128Hz
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 32, 256) # Batch=2
        em_out, fo_out = model(dummy_input)
        
        print(f"  -> Forward pass output shapes: Emotion {em_out.shape}, Focus {fo_out.shape}")
        
        assert em_out.shape == (2, 4)
        assert fo_out.shape == (2, 2)
        print("  -> Model test successful.")
    except Exception as e:
        print(f"  -> Model test failed: {e}")
        sys.exit(1)

def test_preprocessing():
    print("Testing preprocessing functions...")
    try:
        from utils.preprocessing.segmentation import segment_signal
        dummy_data = np.random.randn(32, 1280) # 10s data
        segments = segment_signal(dummy_data, window_size=2.0, overlap=0.0, fs=128)
        # 10s / 2s = 5 segments
        print(f"  -> Segmented shape: {segments.shape}")
        assert segments.shape == (5, 32, 256)
        print("  -> Preprocessing test successful.")
    except Exception as e:
        print(f"  -> Preprocessing test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("--- Starting Smoke Test ---")
    test_imports()
    test_preprocessing()
    test_model_instantiation()
    print("--- Smoke Test Passed ---")
