# Configuration Constants

# Data Parameters
SAMPLING_RATE_DEAP = 128
SAMPLING_RATE_SEED = 200  # Example, check actual SEED sampling rate if used

# Preprocessing
FILTER_LOW = 0.5
FILTER_HIGH = 45.0
FILTER_ORDER = 4

# Segmentation
WINDOW_SIZE_SECONDS = 2
OVERLAP = 0.5  # 50% overlap

# Feature Extraction
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Label Mapping (DEAP)
VALENCE_THRESHOLD = 5.0
AROUSAL_THRESHOLD = 5.0

# Model
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
HIDDEN_SIZE_LSTM = 64
ATTENTION_HEADS = 4

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "outputs/logs/"
