from transformers import AutoModel, BertPreTrainedModel
import torch
from copy import deepcopy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

w = [1.0, 1.7999, 4.1223, 4.2224, 2.5114, 2.9772, 1.9814, 3.0767, 5.2281, 6.2914, 4.4455, 4.8999, 4.8999, 4.9155, 3.8822, 
    3.0446, 5.2499, 3.4842, 4.0533, 5.5776, 2.6310, 3.9087, 5.9729, 5.7559, 4.1271, 3.1326, 5.0506, 6.4737, 5.1519, 5.5982]

w = torch.tensor(w).to(device)


def DWBL(logits, labels):
    
    sum_ = torch.tensor(0.0, requires_grad=True)
    logits = torch.nn.functional.softmax(logits, dim=-1)
    labels = torch.nn.functional.one_hot(labels, num_classes=30)
    
    for i in range(30):
        sum_ = sum_ + (w[i]**(1-logits[:,i]))*labels[:,i]*torch.log(logits[:,i]) - logits[:,i]*(1-logits[:,i])
        
    sum_ = -sum_.mean()
    
    return sum_

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
        
        '''
        token_ids = tokenizer.convert_tokens_to_ids(['[SS]', '[SE]', '[OS]', '[OE]'])
        self.sub_start, self.sub_end, self.obj_start, self.obj_end = token_ids[0], token_ids[1], token_ids[2], token_ids[3]
        '''
        
        # 이 classifier는 각 entity special token 마다 적용된다.
        self.classifier = torch.nn.Sequential(
            self.model.pooler, # hidden_size -> hidden_size
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels , bias=True)
        )
        
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(4)])
        # 각 classifier layer를 통과한 hidden state를 가중합하고 그 가중치를 학습시킨다.
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[0.25]], [[0.25]], [[0.25]], [[0.25]]]))
        
        '''
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=4*config.hidden_size, out_features=4*config.hidden_size, bias=True),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=4*config.hidden_size, out_features=config.num_labels, bias=True),
            )
        '''
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None, output_attentions=False):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = outputs.last_hidden_state # batch, seqlen, hidden_size
                        
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
        
        '''
        pooled_output = torch.stack([special_outputs[i, special_idx[i], :].view(-1, 4*self.config.hidden_size).squeeze() for i in range(batch_size)], dim=0) # batch, 4*hidden_size
        
        logits = self.classifier(pooled_output) # batch, num_labels
        
        '''
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
        
class SepecialPunctBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config
        
        ids = tokenizer.convert_tokens_to_ids(['@', '#'])
        self.sub_ids, self.obj_ids = ids[0], ids[1]
        
        
        # 이 classifier는 각 entity special token 마다 적용된다.
        self.classifier = torch.nn.Sequential(
            #self.model.pooler, # hidden_size -> hidden_size
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels , bias=True)
        )
        
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(2)])
        # 각 classifier layer를 통과한 hidden state를 가중합하고 그 가중치를 학습시킨다.
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[0.5]], [[0.5]]]))
        
        '''
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(4)])
        # 각 classifier layer를 통과한 hidden state를 가중합하고 그 가중치를 학습시킨다.
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[0.25]], [[0.25]], [[0.25]], [[0.25]]]))
        '''
        '''
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*config.hidden_size, out_features=2*config.hidden_size, bias=True),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=2*config.hidden_size, out_features=config.num_labels, bias=True),
            )
        '''
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, output_attentions=False):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = outputs.last_hidden_state
                        
        batch_size = len(input_ids)
        
        special_idx = list()
        
        # entity type을 forward에서  안 받고, 그냥 문장 내에서 special token을 찾는 방법도 써보았으나, 학습시간이 거의 2배로 증가한다.
        for i in range(batch_size):
            
            sub_start_idx = torch.nonzero(input_ids[i] == self.sub_ids)[0][0] # entity type에 맞는 special token의 index
            # sub_end_idx = torch.nonzero(input_ids[i] == self.sub_ids)[1][0]
            obj_start_idx = torch.nonzero(input_ids[i] == self.obj_ids)[0][0]
            # obj_end_idx = torch.nonzero(input_ids[i] == self.obj_ids)[1][0]
            
            special_idx.append([sub_start_idx, obj_start_idx])
            # special_idx.append([sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx])
        
        '''
        pooled_output = torch.stack([special_outputs[i, special_idx[i], :].view(-1, 2*self.config.hidden_size).squeeze() for i in range(batch_size)], dim=0) # batch, 2*hidden_size
        
        logits = self.classifier(pooled_output) # batch, num_labels
        '''
        
        # (batch_size, hidden_size) 가 2개인 list
        pooled_output = [torch.stack([special_outputs[i, special_idx[i][j], :] for i in range(batch_size)]) for j in range(2)]

        #logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(2)], dim=0) # (2, batch, num_label)
        logits = torch.stack([self.special_classifier[i](pooled_output[i]) for i in range(2)], dim=0) # (2, batch, num_label)
        logits = torch.sum(self.weight_parameter*logits, dim=0) # (batch_size, num_label)
        
        '''
        pooled_output = [torch.stack([special_outputs[i, special_idx[i][j], :] for i in range(batch_size)]) for j in range(4)]

        logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(4)], dim=0) # (4, batch, num_label)
        logits = torch.sum(self.weight_parameter*logits, dim=0) # (batch_size, num_label)
        '''
        
        loss = None
        
        if labels is not None:
            weights = torch.ones(30).to(device)
            weights[0] = 2
            loss_fun = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
            #loss = DWBL(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        # attention 은 num_layer * (batch_size, num_attention_head, sequence_length, sequence_length)    
        if output_attentions:    
            outputs = {"loss" : loss, "logits": logits, "attentions": outputs.attentions[0]}
        else:
            outputs = {"loss" : loss, "logits": logits}
        
        return outputs # (loss), logits, (attentions)