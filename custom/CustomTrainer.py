from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        pt = torch.gather(probs, 1, targets.unsqueeze(1))
        focal_loss = -self.alpha * (torch.pow(1 - pt, self.gamma)) * log_probs
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        targets = (1 - self.smoothing) * targets + self.smoothing / num_classes
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(log_probs * targets, dim=1)
        return loss.mean()
    
class FocalWithLabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.0):
        super(FocalWithLabelSmoothingLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=self.smoothing)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        label_smoothing_loss = self.label_smoothing_loss(inputs, targets)
        loss = focal_loss + label_smoothing_loss
        return loss


class CustomTrainer(Trainer):
    def __init__(self, loss_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_fn == 'focal':
            self.loss_fn = FocalLoss()
        elif loss_fn == 'label_smoothing':
            self.loss_fn = LabelSmoothingLoss()
        elif loss_fn == 'focal_with_label_smoothing':
            self.loss_fn = FocalWithLabelSmoothingLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs['logits']
        labels = inputs['labels']
        
        loss = self.loss_fn(logits, labels)
        # print(loss, outputs['loss'])
        
        return (loss, outputs) if return_outputs else loss