from transformers import AutoModel, BertPreTrainedModel
import torch
from copy import deepcopy
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']

sub_type_dict = {'PER': '인물', 'ORG': '단체', 'LOC': '장소'}
obj_type_dict = {'PER': '인물', 'ORG': '단체', 'POH': '명칭', 'DAT': '날짜', 'LOC': '장소', 'NOH': '수'}
'''   

class DRPBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config
        
        self.splt = tokenizer.convert_tokens_to_ids('@')
        self.no = tokenizer.convert_tokens_to_ids('#')
        self.sep = tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id
        self.cls = tokenizer.cls_token_id
        
        self.weight = torch.nn.Linear(in_features = config.hidden_size, out_features=1, bias=True)
        
    def forward(self, input_tensor, att_mask_tensor=None, no_predict_tensor=None, labels=None):
        
        batch_size = len(input_tensor)
        batch_output = []
        
        for i in range(batch_size):
            '''
            input_id = input_ids[i]
            
            sublists = [[self.cls]+list(group)+[self.sep] for key, group in groupby(input_id, lambda x: x == self.splt) if not key]
            sublists.pop()
            sublists[0].pop(0)
            
            no_predict_idx = [1 if sublist == [self.cls, self.no, self.sep] else 0 for index, sublist in enumerate(sublists)]
            no_predict_idx = torch.tensor(no_predict_idx).unsqueeze(1).to(device) # (30, 1)
            
            padded_tensor = pad_sequence([torch.tensor(seq) for seq in sublists], batch_first=True, padding_value=self.pad) # 30, max_seqlen
            att_mask = torch.where(padded_tensor == self.pad, torch.tensor(0.0), torch.tensor(1.0))
            '''
            
            outputs = self.model(input_tensor[i], attention_mask=att_mask_tensor[i].to(device))
            outputs = outputs.last_hidden_state[:, 0] # 30, hidden_size
            outputs = outputs*(1-no_predict_tensor[i]) # 30, hidden_size
            
            outputs = self.weight(outputs) # 30, 1
            batch_output.append(outputs)
            
        logits = torch.stack(batch_output, dim=0).squeeze() # batch, 30
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
            
            return loss, logits
           
        return logits 