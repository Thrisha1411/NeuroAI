import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import numpy as np

def train_model(model, train_dataset, val_dataset, config):
    """
    Full training loop.
    
    Args:
        model (nn.Module): The model to train.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        config (dict): Training configuration.
            - batch_size
            - learning_rate
            - num_epochs
            - device
            - save_dir
    
    Returns:
        history (dict): Training history (loss, acc).
    """
    device = config.get('device', 'cpu')
    batch_size = config.get('batch_size', 32)
    lr = config.get('learning_rate', 0.001)
    epochs = config.get('num_epochs', 10)
    save_dir = config.get('save_dir', 'models/checkpoints')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Loss & Optimizer
    from .loss import CombinedLoss
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Training Loop
    model.to(device)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc_em': [], 'val_acc_em': [],
        'train_acc_fo': [], 'val_acc_fo': []
    }
    
    best_val_loss = float('inf')
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_em = 0
        correct_fo = 0
        total_samples = 0
        
        for inputs, em_labels, fo_labels in train_loader:
            inputs = inputs.to(device)
            em_labels = em_labels.to(device)
            fo_labels = fo_labels.to(device)
            
            optimizer.zero_grad()
            
            em_out, fo_out = model(inputs)
            
            loss, loss_em, loss_fo = criterion(em_out, fo_out, em_labels, fo_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Metrics
            _, pred_em = torch.max(em_out, 1)
            _, pred_fo = torch.max(fo_out, 1)
            
            correct_em += torch.eq(pred_em, em_labels).sum().item()
            correct_fo += torch.eq(pred_fo, fo_labels).sum().item()
            total_samples += inputs.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc_em = correct_em / total_samples
        epoch_acc_fo = correct_fo / total_samples
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct_em = 0
        val_correct_fo = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, em_labels, fo_labels in val_loader:
                inputs = inputs.to(device)
                em_labels = em_labels.to(device)
                fo_labels = fo_labels.to(device)
                
                em_out, fo_out = model(inputs)
                loss, _, _ = criterion(em_out, fo_out, em_labels, fo_labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, pred_em = torch.max(em_out, 1)
                _, pred_fo = torch.max(fo_out, 1)
                
                val_correct_em += torch.eq(pred_em, em_labels).sum().item()
                val_correct_fo += torch.eq(pred_fo, fo_labels).sum().item()
                val_total += inputs.size(0)
                
        val_epoch_loss = val_loss / val_total
        val_acc_em = val_correct_em / val_total
        val_acc_fo = val_correct_fo / val_total
        
        # Logging
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} "
              f"Acc Em: {epoch_acc_em:.2f} | Acc Fo: {epoch_acc_fo:.2f}")
        
        # Save History
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['train_acc_em'].append(epoch_acc_em)
        history['val_acc_em'].append(val_acc_em)
        history['train_acc_fo'].append(epoch_acc_fo)
        history['val_acc_fo'].append(val_acc_fo)
        
        # Checkpoint
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("  -> Saved best model.")
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print("Training complete.")
    
    return history
