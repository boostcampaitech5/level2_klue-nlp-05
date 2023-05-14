from transformers import AutoModel, BertPreTrainedModel, AutoModelForSequenceClassification
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
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
        
        outputs = (logits,) + outputs[2:]
        
        if labels is not None: # 실제로 inference에서 label은 None이 아니라 100이지만 그냥 return 할 때, 필요하므로 냅두었다.
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
            
            outputs = (loss,) + outputs
        
        return outputs # (loss), logits, (hidden_states), (attentions)                
        
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
            self.model.pooler, # hidden_size -> hidden_size
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels , bias=True)
        )
        
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(2)])
        # 각 classifier layer를 통과한 hidden state를 가중합하고 그 가중치를 학습시킨다.
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[0.5]], [[0.5]]]))
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        special_outputs = outputs.last_hidden_state
                        
        batch_size = len(input_ids)
        
        special_idx = list()
        
        # entity type을 forward에서  안 받고, 그냥 문장 내에서 special token을 찾는 방법도 써보았으나, 학습시간이 거의 2배로 증가한다.
        for i in range(batch_size):
            
            sub_start_idx = torch.nonzero(input_ids[i] == self.sub_ids)[0][0] # entity type에 맞는 special token의 index
            obj_start_idx = torch.nonzero(input_ids[i] == self.obj_ids)[0][0]
            
            special_idx.append([sub_start_idx, obj_start_idx])
        
        # (batch_size, hidden_size) 가 2개인 list
        pooled_output = [torch.stack([special_outputs[i, special_idx[i][j], :] for i in range(batch_size)]) for j in range(2)]

        logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(2)], dim=0) # (2, batch, num_label)
        logits = torch.sum(self.weight_parameter*logits, dim=0) # (batch_size, num_label)
        
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
            
            outputs = (loss,) + outputs
        
        return outputs # (loss), logits, (hidden_states), (attentions)

class CLS_SpecialEntityBERT(BertPreTrainedModel) :
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config

        self.subj = tokenizer.convert_tokens_to_ids('[SUBJ]')
        self.obj = tokenizer.convert_tokens_to_ids('[OBJ]')
        
        # 이 classifier는 각 entity special token 마다 적용된다.
        self.pool_special_linear_block = torch.nn.Sequential(
                torch.nn.Linear(
                    5 * self.model.config.hidden_size, self.model.config.hidden_size
                ),  # 5 for 1 [CLS], 2 [SUBJ], 2 [OBJ]
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(self.model.config.hidden_size, config.num_labels),
            )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None) :
        # 이 형태로 들어오는 이유 : dataset이 __getitem__을 보면 딕셔너리 형태로 들어가니까 **kwargs 로 들어갈텐데,
        # 그러니까 x가 들어오는것이 아니라 이런 형태로 넣어줘야 한다.
        batch_size = len(input_ids)
        output = self.model(input_ids = input_ids , attention_mask=attention_mask, 
        token_type_ids=token_type_ids, output_hidden_states=True)
        # output은 3가지의 output
        # 0번째 : last hidden state > size (batch , sequence_length, hidden_size)
        # 1번째 : pooler_output > CLS 토큰에 대한 정보가 담긴 tensor, size (batch, hidden_size)
        # 2번째 : hidden states, 이 전 layer의 hidden state들의 리스트. shape > (batch, sequence_length, hidden_size)
        # 첫 번째 요소는 last hidden state와 동일함

        before_output = []
        
        for i in range(batch_size) :
            ''' 마지막에 [subj], [obj] 토큰의 hidden state를 갖고오기 위해서 어떤 부분에 위치해 있는지 미리 찾기 '''
            each_sentence = []
            each_sentence.append(output.last_hidden_state[i][0]) # CLS 

            subj_idx = int((input_ids[i]==self.subj).nonzero())
            #subj_idx = input_ids[i].find(subj) 
            subj_type_idx = subj_idx + 1 # subject 유형
            each_sentence.append(output.last_hidden_state[i][subj_idx]) # subj 
            each_sentence.append(output.last_hidden_state[i][subj_type_idx]) # subj type

            obj_idx = int((input_ids[i]==self.obj).nonzero())
            #obj_idx = input_ids[i].find(obj)
            obj_type_idx = obj_idx + 1
            each_sentence.append(output.last_hidden_state[i][obj_idx]) # obj
            each_sentence.append(output.last_hidden_state[i][obj_type_idx]) # obj type

            before_input = torch.cat(each_sentence)
            after_classifier = self.pool_special_linear_block(before_input)
            before_output.append(after_classifier)

        before_output = torch.stack(before_output) # tensor 화
        # real_output = (befroe_output,) + output[2:] # 여기서 output[2:] 는 hidden_state들!
        # bert 모델을 찾아보면 forward 하면 나와야 하는 값이
        # # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(before_output.view(-1, self.config.num_labels), labels.view(-1))
            real_output = (loss,) + (before_output,)
        else :
            real_output = (before_output,)
        
        return real_output

class sangmin_SpecialEntityBERT(BertPreTrainedModel) :
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config

        self.subj = tokenizer.convert_tokens_to_ids('[SUBJ]')
        self.obj = tokenizer.convert_tokens_to_ids('[OBJ]')
        
        # 이 classifier는 각 entity special token 마다 적용된다.
        self.pool_special_linear_block = torch.nn.Sequential(
                torch.nn.Linear(
                    4 * self.model.config.hidden_size, self.model.config.hidden_size
                ),  # 5 for 1 [CLS], 2 [SUBJ], 2 [OBJ]
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(self.model.config.hidden_size, config.num_labels),
            )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None) :
        # 이 형태로 들어오는 이유 : dataset이 __getitem__을 보면 딕셔너리 형태로 들어가니까 **kwargs 로 들어갈텐데,
        # 그러니까 x가 들어오는것이 아니라 이런 형태로 넣어줘야 한다.
        batch_size = len(input_ids)
        output = self.model(input_ids = input_ids , attention_mask=attention_mask, 
        token_type_ids=token_type_ids, output_hidden_states=True)
        # output은 3가지의 output
        # 0번째 : last hidden state > size (batch , sequence_length, hidden_size)
        # 1번째 : pooler_output > CLS 토큰에 대한 정보가 담긴 tensor, size (batch, hidden_size)
        # 2번째 : hidden states, 이 전 layer의 hidden state들의 리스트. shape > (batch, sequence_length, hidden_size)
        # 첫 번째 요소는 last hidden state와 동일함

        before_output = []
        
        for i in range(batch_size) :
            ''' 마지막에 [subj], [obj] 토큰의 hidden state를 갖고오기 위해서 어떤 부분에 위치해 있는지 미리 찾기 '''
            each_sentence = []
            # each_sentence.append(output.last_hidden_state[i][0]) # CLS 

            subj_idx = int((input_ids[i]==self.subj).nonzero())
            #subj_idx = input_ids[i].find(subj) 
            subj_type_idx = subj_idx + 1 # subject 유형
            each_sentence.append(output.last_hidden_state[i][subj_idx]) # subj 
            each_sentence.append(output.last_hidden_state[i][subj_type_idx]) # subj type

            obj_idx = int((input_ids[i]==self.obj).nonzero())
            #obj_idx = input_ids[i].find(obj)
            obj_type_idx = obj_idx + 1
            each_sentence.append(output.last_hidden_state[i][obj_idx]) # obj
            each_sentence.append(output.last_hidden_state[i][obj_type_idx]) # obj type

            before_input = torch.cat(each_sentence)
            after_classifier = self.pool_special_linear_block(before_input)
            before_output.append(after_classifier)

        before_output = torch.stack(before_output) # tensor 화
        # real_output = (befroe_output,) + output[2:] # 여기서 output[2:] 는 hidden_state들!
        # bert 모델을 찾아보면 forward 하면 나와야 하는 값이
        # # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(before_output.view(-1, self.config.num_labels), labels.view(-1))
            real_output = (loss,) + (before_output,)
        else :
            real_output = (before_output,)
        
        return real_output