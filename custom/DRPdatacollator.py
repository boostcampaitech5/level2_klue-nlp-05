from transformers import DataCollatorWithPadding
import torch
import copy

class DRPDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = {}
        batch['input_tensor'] = [feature['input_tensor'] for feature in features]
        batch['att_mask_tensor'] = [feature['att_mask_tensor'] for feature in features]
        batch['no_predict_tensor'] = [feature['no_predict_tensor'] for feature in features]
        batch['labels'] = [feature['labels'] for feature in features]
        
        return batch