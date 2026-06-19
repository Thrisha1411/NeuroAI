import numpy as np

def map_emotion_label(valence, arousal, threshold=5.0):
    """
    Maps Valence and Arousal values to 4 emotion classes.
    Classes: 
    0: Happy (HV, HA)
    1: Calm (HV, LA)
    2: Sad (LV, LA)
    3: Angry (LV, HA)
    
    Args:
        valence (float): Valence score (1-9).
        arousal (float): Arousal score (1-9).
        threshold (float): Midpoint threshold (usually 5.0 for DEAP).
        
    Returns:
        int: Emotion class index (0-3).
    """
    if valence >= threshold and arousal >= threshold:
        return 0 # Happy
    elif valence >= threshold and arousal < threshold:
        return 1 # Calm
    elif valence < threshold and arousal < threshold:
        return 2 # Sad
    else:
        return 3 # Angry

def get_emotion_name(label_idx):
    mapping = {0: "Happy", 1: "Calm", 2: "Sad", 3: "Angry"}
    return mapping.get(label_idx, "Unknown")

def map_focus_label(arousal, dominance, threshold=5.0):
    """
    Maps Arousal and Dominance to Focus level.
    High Arousal + High Dominance -> Focused (1)
    Else -> Distracted (0)
    
    Args:
        arousal (float): Arousal score.
        dominance (float): Dominance score.
        
    Returns:
        int: Focus class (0 or 1).
    """
    # Simple heuristic: High arousal and control implies focus
    if arousal >= threshold and dominance >= threshold:
        return 1 # Focused
    else:
        return 0 # Distracted

def get_focus_name(label_idx):
    return "Focused" if label_idx == 1 else "Distracted"
