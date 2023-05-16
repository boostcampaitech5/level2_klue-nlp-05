from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import LabelSmoother
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # inputs : [batch_size, num_classes]
        # targets : [batch_size]

        # 1. 모든 클래스에 대해 1 - smoothing의 가중치가 있는 확률 분포를 생성합니다.
        # 예를 들어, num_classes가 10이고 smoothing이 0.1이면 분포는 [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1].
        p = 1.0 - self.smoothing
        q = self.smoothing / (torch.numel(inputs) - 1)
        one_hot = torch.full(inputs.size(), q, dtype=torch.float, device=inputs.device)
        one_hot[targets] = p

        # 2. 로그 손실 함수를 사용하여 분포에 대해 예측을 손실합니다.
        loss = -(one_hot * inputs).log()

        return loss.mean()
    
# class FocalWithLabelSmoothingLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1):
#         super(FocalWithLabelSmoothingLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.smoothing = smoothing
#         self.focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma)
#         self.label_smoothing_loss = LabelSmoothingLoss(smoothing=self.smoothing)

#     def forward(self, inputs, targets):
#         focal_loss = self.focal_loss(inputs, targets)
#         label_smoothing_loss = self.label_smoothing_loss(inputs, targets)
#         loss = focal_loss + label_smoothing_loss
#         return loss


class CustomTrainer(Trainer):
    def __init__(self, loss_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs['logits']
        labels = inputs['labels']
        focal_loss_fn = FocalLoss()
        label_smoothing_loss_fn = LabelSmoother(epsilon=0.1)
        default_loss_fn = nn.CrossEntropyLoss()
        
        if self.loss_fn == 'focal':
            loss = focal_loss_fn(logits, labels)
        elif self.loss_fn == 'label_smoothing':
            loss = label_smoothing_loss_fn(outputs, labels)
        # elif self.loss_fn == 'focal_with_label_smoothing':
        #     torch.autograd.set_detect_anomaly(True)
        #     focal_loss = focal_loss_fn(logits, labels)
        #     label_smoothing_loss = label_smoothing_loss_fn(outputs, labels)
        #     loss = focal_loss + label_smoothing_loss
        elif self.loss_fn == 'penalty_loss':
            weights = torch.ones(30).to(device)
            weights[11] = 2
            loss_fun = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fun(logits, labels)
        else:
            loss = default_loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss