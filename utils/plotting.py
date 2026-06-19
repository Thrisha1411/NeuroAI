import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Plot confusion matrix using Seaborn.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    return fig

def plot_training_history(history):
    """
    Plot training and validation loss/accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy (Emotion)
    ax2.plot(history['train_acc_emotion'], label='Train Acc (Emotion)')
    ax2.plot(history['val_acc_emotion'], label='Val Acc (Emotion)')
    ax2.plot(history['train_acc_focus'], label='Train Acc (Focus)', linestyle='--')
    ax2.plot(history['val_acc_focus'], label='Val Acc (Focus)', linestyle='--')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    return fig

def plot_prediction_distribution(counts, classes, title="Display"):
    """
    Plot bar chart of prediction counts.
    """
    fig, ax = plt.subplots()
    labels = list(counts.keys())
    values = list(counts.values())
    
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']
    selected_colors = [colors[i % len(colors)] for i in range(len(labels))]
    ax.bar(labels, values, color=selected_colors)
    ax.set_title(title)
    ax.set_ylabel('Count')
    return fig
