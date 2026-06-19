import json
import os
from datetime import datetime

HISTORY_FILE = "data/history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_result(filename, dataset, emotion, focus, metrics):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "dataset": dataset,
        "prediction_emotion": emotion,
        "prediction_focus": focus,
        "metrics": metrics
    }
    
    history = load_history()
    history.insert(0, entry) # Prepend
    
    if not os.path.exists("data"):
        os.makedirs("data")
        
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
