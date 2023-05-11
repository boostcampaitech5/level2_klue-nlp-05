from transformers import AutoModel, BertPreTrainedModel
import torch
from copy import deepcopy

class SepecialEntityBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config
        
        
        # subject entity type 에 맞는 special token의 id에 대한 dictionary
        sub_tokens = ['[S:PER]', '[S:ORG]', '[S:LOC]', '[/S:PER]', '[/S:ORG]', '[/S:LOC]']
        token_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        self.sub_entity_token = dict(zip(sub_tokens, token_ids))
        
        # object entity type 에 맞는 special token의 id에 대한 dictionary
        obj_tokens = ['[O:PER]', '[O:ORG]', '[O:POH]', '[O:LOC]', '[O:NOH]', '[O:DAT]',
                      '[/O:PER]', '[/O:ORG]', '[/O:POH]', '[/O:LOC]', '[/O:NOH]', '[/O:DAT]']
        token_ids = tokenizer.convert_tokens_to_ids(obj_tokens)
        self.obj_entity_token = dict(zip(obj_tokens, token_ids))
        
        # 이 classifier는 각 entity special token 마다 적용된다.
        self.classifier = torch.nn.Sequential(
            self.model.pooler, # hidden_size -> hidden_size
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels , bias=True)
        )
        
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(4)])
        # 각 classifier layer를 통과한 hidden state를 가중합하고 그 가중치를 학습시킨다.
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[0.25]], [[0.25]], [[0.25]], [[0.25]]]))
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None, output_attentions=False):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = outputs.last_hidden_state
                     
        batch_size = len(input_ids)
        
        special_idx = list()
        
        # entity type을 forward에서  안 받고, 그냥 문장 내에서 special token을 찾는 방법도 써보았으나, 학습시간이 거의 2배로 증가한다.
        for i in range(batch_size):
            sub_start, sub_end = self.sub_entity_token[f'[S:{subject_type[i]}]'], self.sub_entity_token[f'[/S:{subject_type[i]}]']
            obj_start, obj_end = self.obj_entity_token[f'[O:{object_type[i]}]'], self.obj_entity_token[f'[/O:{object_type[i]}]']
            
            sub_start_idx = torch.nonzero(input_ids[i] == sub_start)[0][0] # entity type에 맞는 special token의 index
            sub_end_idx = torch.nonzero(input_ids[i] == sub_end)[0][0]
            obj_start_idx = torch.nonzero(input_ids[i] == obj_start)[0][0]
            obj_end_idx = torch.nonzero(input_ids[i] == obj_end)[0][0]
            
            special_idx.append([sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx])
        
        # (batch_size, hidden_size) 가 4개인 list
        pooled_output = [torch.stack([special_outputs[i, special_idx[i][j], :] for i in range(batch_size)]) for j in range(4)]

        logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(4)], dim=0) # (4, batch, num_label)
        logits = torch.sum(self.weight_parameter*logits, dim=0) # (batch_size, num_label)
        
        loss = None
        
        if labels is not None: # 실제로 inference에서 label은 None이 아니라 100이지만 그냥 return 할 때, 필요하므로 냅두었다.
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        # attention 은 num_layer * (batch_size, num_attention_head, sequence_length, sequence_length)
        if output_attentions:    
            outputs = {"loss" : loss, "logits": logits, "attentions": outputs.attentions[0]}
        else:
            outputs = {"loss" : loss, "logits": logits}
        
        return outputs # (loss), logits, (attentions)   