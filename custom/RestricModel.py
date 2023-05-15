from transformers import AutoModel, BertPreTrainedModel
import torch
from copy import deepcopy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RestrictPunctBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config
        
        ids = tokenizer.convert_tokens_to_ids(['@', '#'])
        self.sub_ids, self.obj_ids = ids[0], ids[1]
        
        '''
        self.type_label = [
            [0, 6, 10, 12, 13, 14, 16, 17, 21], # 'PER, PER'
            [0, 4, 6, 8, 15, 23, 29], # 'PER, ORG'
            [0, 4, 6, 8, 10, 12, 13, 16, 17, 21, 29], # 'PER, POH'
            [0, 24, 25], # 'PER, DAT'
            [0, 6, 11, 15, 26, 27], # 'PER, LOC'
            [0], # 'PER, NOH'
            [0, 1, 28], # 'ORG, PER'
            [0, 2, 5, 7, 19, 20], # 'ORG, ORG'
            [0, 1, 3, 5, 19, 20], # 'ORG, POH'
            [0, 18, 22], # 'ORG, DAT'
            [0, 2, 7, 20], # 'ORG, LOC'
            [0, 9] # 'ORG, NOH'
            ]
        '''
        self.type_label = [[0, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 21, 24, 26, 27], # 'PER, PER' 
            [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 23, 24, 26, 27, 29], # 'PER, ORG'
            [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 21, 27, 29], # 'PER, POH'
            [0, 4, 6, 10, 11, 14, 15, 17, 21, 24, 25, 26, 27], # 'PER, DAT'
            [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 21, 23, 24, 26, 27, 29], # 'PER, LOC'
            [0, 4, 6, 10, 12, 15, 21, 24, 25], # 'PER, NOH'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, PER'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, ORG'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, POH'
            [0, 2, 5, 7, 18, 19, 20, 22], # 'ORG, DAT'
            [0, 1, 2, 3, 5, 7, 19, 20], # 'ORG, LOC'
            [0, 1, 2, 3, 5, 7, 9, 20], # 'ORG, NOH'
            [0], [0], [0], [0], [0], [0]] # LOC
        
        self.classifier = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(in_features=2*config.hidden_size, out_features=2*config.hidden_size, bias=True),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(in_features=2*config.hidden_size, out_features=len(self.type_label[i]), bias=True)
            )
        for i in range(12)])
        
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, restrict_num=None, output_attentions=False):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = outputs.last_hidden_state # batch, seqlen, hidden_size
                        
        batch_size = len(input_ids)
        
        new_output = list()
        
        for i in range(batch_size):
            
            sub_start_idx = torch.nonzero(input_ids[i] == self.sub_ids)[0][0] # entity type에 맞는 token의 index
            obj_start_idx = torch.nonzero(input_ids[i] == self.obj_ids)[0][0]
            
            if restrict_num[i] == -1:
                pooled_output = torch.tensor([1.0]+[0.0 for _ in range(self.config.num_labels)]).to(device)
            else:
                pooled_output = special_outputs[i, [sub_start_idx, obj_start_idx], :].view(-1, 2*self.config.hidden_size) # 1, 2*hidden_size
                pooled_output = self.classifier[restrict_num[i]](pooled_output).squeeze() # restrict

                pooled_output_2 = torch.zeros(self.config.num_labels).to(device)
                pooled_output_2[self.type_label[restrict_num[i]]] = pooled_output # 30
            
            new_output.append(pooled_output_2)
            
        logits = torch.stack(new_output, dim=0) # batch, 30
        
        loss = None
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        # attention 은 num_layer * (batch_size, num_attention_head, sequence_length, sequence_length)    
        if output_attentions:    
            outputs = {"loss" : loss, "logits": logits, "attentions": outputs.attentions[0]}
        else:
            outputs = {"loss" : loss, "logits": logits}
        
        return outputs # (loss), logits, (attentions)
    
class RestrictPunctBERT2(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config
        
        ids = tokenizer.convert_tokens_to_ids(['@', '#'])
        self.sub_ids, self.obj_ids = ids[0], ids[1]
        
        '''
        self.type_label = [
            [0, 6, 10, 12, 13, 14, 16, 17, 21], # 'PER, PER'
            [0, 4, 6, 8, 15, 23, 29], # 'PER, ORG'
            [0, 4, 6, 8, 10, 12, 13, 16, 17, 21, 29], # 'PER, POH'
            [0, 24, 25], # 'PER, DAT'
            [0, 6, 11, 15, 26, 27], # 'PER, LOC'
            [0], # 'PER, NOH'
            [0, 1, 28], # 'ORG, PER'
            [0, 2, 5, 7, 19, 20], # 'ORG, ORG'
            [0, 1, 3, 5, 19, 20], # 'ORG, POH'
            [0, 18, 22], # 'ORG, DAT'
            [0, 2, 7, 20], # 'ORG, LOC'
            [0, 9] # 'ORG, NOH'
            ]
        '''
        self.type_label = [[0, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 21, 24, 26, 27], # 'PER, PER' 
            [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 23, 24, 26, 27, 29], # 'PER, ORG'
            [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 21, 27, 29], # 'PER, POH'
            [0, 4, 6, 10, 11, 14, 15, 17, 21, 24, 25, 26, 27], # 'PER, DAT'
            [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 21, 23, 24, 26, 27, 29], # 'PER, LOC'
            [0, 4, 6, 10, 12, 15, 21, 24, 25], # 'PER, NOH'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, PER'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, ORG'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, POH'
            [0, 2, 5, 7, 18, 19, 20, 22], # 'ORG, DAT'
            [0, 1, 2, 3, 5, 7, 19, 20], # 'ORG, LOC'
            [0, 1, 2, 3, 5, 7, 9, 20], # 'ORG, NOH'
            [0], [0], [0], [0], [0], [0]] # LOC
        
        # 이 classifier는 각 entity special token 마다 적용된다.
        self.classifier = torch.nn.Sequential(
            self.model.pooler, # hidden_size -> hidden_size
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels , bias=True)
        )
        
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(2)])
        # 각 classifier layer를 통과한 hidden state를 가중합하고 그 가중치를 학습시킨다.
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[0.5]], [[0.5]]]))
        
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, restrict_num=None, output_attentions=False):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = outputs.last_hidden_state # batch, seqlen, hidden_size
                        
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
        
        # (batch_size, hidden_size) 가 2개인 list
        pooled_output = [torch.stack([special_outputs[i, special_idx[i][j], :] for i in range(batch_size)]) for j in range(2)]
        
        new_output = []
        
        for i in range(2):
            new_lst = []
            for j in range(batch_size):
                tmp = torch.zeros(30).to(device).detach()
                tmp[self.type_label[restrict_num[j]]] = self.special_classifier[i](pooled_output[i][j].unsqueeze(0).unsqueeze(1))[0, self.type_label[restrict_num[j]]]
                new_lst.append(tmp)
                
            new_output.append(torch.stack(new_lst, dim=0))
        
        logits = torch.stack(new_output, dim=0) # (2, batch, num_label)
        logits = torch.sum(self.weight_parameter*logits, dim=0) # (batch_size, num_label)
        
        loss = None
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        # attention 은 num_layer * (batch_size, num_attention_head, sequence_length, sequence_length)    
        if output_attentions:    
            outputs = {"loss" : loss, "logits": logits, "attentions": outputs.attentions[0]}
        else:
            outputs = {"loss" : loss, "logits": logits}
        
        return outputs # (loss), logits, (attentions)