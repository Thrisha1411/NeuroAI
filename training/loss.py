import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    """
    Combined Loss for Multi-Task Learning.
    Loss = alpha * EmotionLoss + beta * FocusLoss
    """
    def __init__(self, alpha=1.0, beta=1.0, emotion_weight=None, focus_weight=None):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
        # CrossEntropyLoss expects class indices (LongTensor)
        self.emotion_criterion = nn.CrossEntropyLoss(weight=emotion_weight)
        self.focus_criterion = nn.CrossEntropyLoss(weight=focus_weight)
        
    def forward(self, emotion_pred, focus_pred, emotion_target, focus_target):
        """
        Args:
            emotion_pred: (Batch, Num_Emotion_Classes)
            focus_pred: (Batch, Num_Focus_Classes)
            emotion_target: (Batch) Class indices
            focus_target: (Batch) Class indices
        """
        loss_emotion = self.emotion_criterion(emotion_pred, emotion_target)
        loss_focus = self.focus_criterion(focus_pred, focus_target)
        
        total_loss = self.alpha * loss_emotion + self.beta * loss_focus
        
        return total_loss, loss_emotion, loss_focus
