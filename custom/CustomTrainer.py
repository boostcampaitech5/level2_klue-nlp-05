from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=2.0, num_classes=30, reduction='mean'):
#         nn.Module.__init__(self)
#         self.weight = weight
#         self.gamma = gamma
#         self.num_classes = num_classes
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         log_prob = F.log_softmax(input_tensor, dim=-1)
#         prob = torch.exp(log_prob)
#         return F.nll_loss(
#             ((1 - prob) ** self.gamma) * log_prob,
#             target_tensor,
#             weight=self.weight,
#             reduction=self.reduction
#         )


class CustomTrainer(Trainer):
    def __init__(self, loss_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.loss_fn = FocalLoss() if loss_fn == 'focal' else torch.nn.CrossEntropyLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs)
        logits = outputs['logits']
        labels = inputs['labels']
        
        loss = self.loss_fn(logits, labels)
        # print(loss, outputs['loss'])
        
        return (loss, outputs) if return_outputs else loss