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
        # input_tensor : batch, 30, 512   att_mask_tensor : batch 30 512  no_predict_tensor : batch 30
        
        batch_size = len(input_tensor)
        logits = []
        '''
        input_tensor = input_tensor.view(-1, input_tensor.size(-1)) # (batch_size*30, 512)
        att_mask_tensor = att_mask_tensor.view(-1, att_mask_tensor.size(-1)) # (batch_size*30, 512)
        no_predict_tensor = no_predict_tensor.view(-1) # (batch_size*30, )
        
        outputs = self.model(input_tensor, attention_mask=att_mask_tensor)
        outputs = outputs.last_hiden_state[:, 0].detach() # (batch_size * 30, hidden_size)
        outputs = outputs * (1 - no_predict_tensor.unsqueeze(-1)) # batch_size*30, hidden_size
        
        outputs = self.weight(outputs)
        logits = outputs.view(batch_size, 30)
        '''
        for i in range(batch_size):
            new_input = input_tensor[i][(no_predict_tensor[i]==0).squeeze()].detach()
            new_attention = att_mask_tensor[i][(no_predict_tensor[i]==0).squeeze()].detach()
            
            outputs = self.model(new_input, attention_mask=new_attention)
            outputs = outputs.last_hidden_state[:, 0].detach() # restric_num, hidden_size
            
            #outputs = outputs*(1-no_predict_tensor[i]) # 30, hidden_size
            outputs = self.weight(outputs) # restric_num, 1
            new_outputs = torch.zeros((30, 1)).to(device)
            new_outputs[(no_predict_tensor[i]==0).squeeze()]=outputs
            
            logits.append(new_outputs.squeeze()) # 30
            
        logits = torch.stack(logits, dim=0) # batch, 30    
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
            
            return loss, logits
           
        return logits 