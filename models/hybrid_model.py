import torch
import torch.nn as nn
from models.attention import Attention

class HybridModel(nn.Module):
    """
    Hybrid CNN-LSTM + Attention Model for EEG Emotion and Focus Detection.
    Takes raw EEG signals (Channels x Time) as input.
    """
    def __init__(self, input_channels=32, input_samples=256, num_emotion_classes=4, num_focus_classes=2):
        super(HybridModel, self).__init__()
        
        # 1. 1D CNN Block for Spatial/Temporal Feature Extraction
        # Input: (Batch, Channels, Time)
        # Note: We treat 'Channels' as the input channels for Conv1d.
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # /2
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # /4
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # /8
        )
        
        # CNN Output: (Batch, 256, Time/8)
        self.cnn_output_channels = 256
        
        # 2. LSTM Block for Temporal Dependencies
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # LSTM Output: (Batch, Seq_Len, Hidden*2)
        self.lstm_out_dim = self.lstm_hidden_size * 2
        
        # 3. Attention Mechanism
        self.attention = Attention(self.lstm_out_dim)
        
        # 4. Classification Heads
        # Shared FC layer
        self.fc_shared = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Emotion Head
        self.emotion_head = nn.Linear(128, num_emotion_classes)
        
        # Focus Head
        self.focus_head = nn.Linear(128, num_focus_classes)
        
    def forward(self, x):
        # x: (Batch, Channels, Time)
        
        # CNN Feature Extraction
        x = self.cnn(x) # (Batch, 256, Time/8)
        
        # Reshape for LSTM: (Batch, Time, Channels) AKA (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1) # (Batch, Time/8, 256)
        
        # LSTM
        lstm_out, _ = self.lstm(x) # (Batch, Seq_Len, 256)
        
        # Attention Pooling
        # context: (Batch, 256)
        context, attn_weights = self.attention(lstm_out)
        
        # Shared FC
        features = self.fc_shared(context)
        
        # Outputs
        emotion_logits = self.emotion_head(features)
        focus_logits = self.focus_head(features)
        
        return emotion_logits, focus_logits
