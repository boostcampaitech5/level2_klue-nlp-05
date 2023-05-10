from transformers import Trainer, TrainingArguments

class CustomTrainer(Trainer):
    def __init__(self, loss_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, inputs["labels"])
        
        return (loss, outputs) if return_outputs else loss