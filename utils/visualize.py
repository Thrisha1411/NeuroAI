import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_confusion_matrix(cm, classes, title, save_path=None):
    """
    Plots a confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    # plt.show() # In Streamlit we return figure usually or save.
    return plt.gcf()

def plot_training_curves(history, save_path=None):
    """
    Plots training and validation loss/accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy (Emotion)
    ax2.plot(epochs, history['train_acc_em'], label='Train Acc (Emotion)')
    ax2.plot(epochs, history['val_acc_em'], label='Val Acc (Emotion)')
    # Accuracy (Focus) - Optional to add
    ax2.plot(epochs, history['train_acc_fo'], label='Train Acc (Focus)', linestyle='--')
    ax2.plot(epochs, history['val_acc_fo'], label='Val Acc (Focus)', linestyle='--')
    
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig
