from transformers import AutoModel, BertPreTrainedModel, AutoModelForSequenceClassification, AutoConfig
import torch
from copy import deepcopy
import numpy as np

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
    
class SequentialDoubleBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer, model_type, device):
        super().__init__(config)

        self.tokenizer = tokenizer
        self.dv = device
        self.model_type = model_type

        self.model_config1 = AutoConfig.from_pretrained(model_name)
        self.model_config2 = AutoConfig.from_pretrained(model_name)
        self.model_config1.num_labels = 2
        self.model_config2.num_labels = 30

        if self.model_type == 'base':
            self.plm1 = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config1)
            self.plm2 = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config2)
        elif self.model_type == 'entity_special':
            self.plm1 = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config1)
            self.plm2 = AutoModel.from_pretrained(model_name, config=self.model_config2)

            self.plm1.resize_token_embeddings(len(self.tokenizer))
            self.plm2.resize_token_embeddings(len(self.tokenizer))

            # 사용할 subject special token list
            sub_tokens = ['[S:PER]', '[S:ORG]', '[S:LOC]', '[/S:PER]', '[/S:ORG]', '[/S:LOC]']
            token_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
            self.sub_entity_token = dict(zip(sub_tokens, token_ids))

            # 사용할 object special token list
            obj_tokens = ['[O:PER]', '[O:ORG]', '[O:POH]', '[O:LOC]', '[O:NOH]', '[O:DAT]',
                  '[/O:PER]', '[/O:ORG]', '[/O:POH]', '[/O:LOC]', '[/O:NOH]', '[/O:DAT]']
            token_ids = tokenizer.convert_tokens_to_ids(obj_tokens)
            self.obj_entity_token = dict(zip(obj_tokens, token_ids))
            
            self.net = torch.nn.Sequential(
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size , bias=True),
                torch.nn.ReLU()
            )

            self.classifier = torch.nn.Linear(in_features=config.hidden_size*4, out_features=config.num_labels , bias=True)
        elif self.model_type == 'entity_punct':
            self.plm1 = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config1)
            self.plm2 = AutoModel.from_pretrained(model_name, config=self.model_config2)

            self.plm1.resize_token_embeddings(len(self.tokenizer))
            self.plm2.resize_token_embeddings(len(self.tokenizer))

            ids = tokenizer.convert_tokens_to_ids(['@', '#'])
            self.sub_ids, self.obj_ids = ids[0], ids[1]

            self.net = torch.nn.Sequential(
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size , bias=True),
                torch.nn.ReLU()
            )

            # 계산에 들어가는 punctuation token의 개수만큼 hidden_size에 곱해준다.
            self.classifier = torch.nn.Linear(in_features=config.hidden_size*2, out_features=config.num_labels , bias=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None, subject_words=None, object_words=None):
        batch_size = len(input_ids)

        DRP_dict = {0 : "는(은) 관계가 없습니다.",
                    1 : "는(은) 관계가 있습니다."}
            
        if self.model_type == 'base':
            outputs1 = self.plm1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits1 = outputs1['logits']
            logits1_cpu = logits1.detach().cpu().numpy()
            result1 = np.argmax(logits1_cpu, axis=-1)

            new_sentences = list()
            tmp_binary_labels = list()

            for i in range(batch_size):
                sentence = self.tokenizer.decode(input_ids[i])[6:].replace('[PAD]', '').rstrip() + " "
                drp_sentence = subject_words[i] + "와(과) " + object_words[i] + DRP_dict[result1[i]] + ": " + f"{object_words[i]}는 {subject_words[i]}의 {object_type[i]}이다."
                new_sentences.append(sentence+drp_sentence)

                if labels != None:
                    label = 0 if labels[i] == 0 else 1
                    tmp_binary_labels.append(label)

            new_tokenized_sentence = self.tokenizer(
                new_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True, 
            )

            new_input_ids = new_tokenized_sentence['input_ids'].view(batch_size, -1).to(self.dv)
            new_attention_mask = new_tokenized_sentence['attention_mask'].view(batch_size, -1).to(self.dv)
            new_token_type_ids = new_tokenized_sentence['token_type_ids'].view(batch_size, -1).to(self.dv)
            tmp_binary_labels = torch.tensor(tmp_binary_labels).view(batch_size, -1).to(self.dv)

            outputs2 = self.plm2(new_input_ids, attention_mask=new_attention_mask, token_type_ids=new_token_type_ids)
            logits2 = outputs2['logits']

            outputs = (logits2,) + outputs2[2:]

            if labels is not None:
                loss_fun1 = torch.nn.BCELoss()
                loss_fun2 = torch.nn.CrossEntropyLoss()
                
                loss1 = loss_fun1(torch.argmax(logits1.view(-1, self.model_config1.num_labels), dim=1).float(), tmp_binary_labels.view(-1).float())
                loss2 = loss_fun2(logits2.view(-1, self.model_config2.num_labels), labels.view(-1))

                loss = loss1 + loss2
            
                outputs = (loss,) + outputs

            return outputs
        elif self.model_type == 'entity_special':
            outputs1 = self.plm1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits1 = outputs1['logits']
            logits1_cpu = logits1.detach().cpu().numpy()
            result1 = np.argmax(logits1_cpu, axis=-1)

            new_sentences = list()
            tmp_binary_labels = list()

            for i in range(batch_size):
                sentence = self.tokenizer.decode(input_ids[i])[6:].replace('[PAD]', '').rstrip() + " "
                drp_sentence = subject_words[i] + "와(과) " + object_words[i] + DRP_dict[result1[i]] + ": " + f"{object_words[i]}는 {subject_words[i]}의 {object_type[i]}이다."
                new_sentences.append(sentence+drp_sentence)

                if labels != None:
                    label = 0 if labels[i] == 0 else 1
                    tmp_binary_labels.append(label)

            new_tokenized_sentence = self.tokenizer(
                new_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True, 
            )

            new_input_ids = new_tokenized_sentence['input_ids'].view(batch_size, -1).to(self.dv)
            new_attention_mask = new_tokenized_sentence['attention_mask'].view(batch_size, -1).to(self.dv)
            new_token_type_ids = new_tokenized_sentence['token_type_ids'].view(batch_size, -1).to(self.dv)
            tmp_binary_labels = torch.tensor(tmp_binary_labels).view(batch_size, -1).to(self.dv)

            outputs2 = self.plm2(new_input_ids, attention_mask=new_attention_mask, token_type_ids=new_token_type_ids)
            hidden_states = outputs2.last_hidden_state
            batch_size, seq_length, hidden_size = hidden_states.shape

            special_hs_list = list()
 
            for i in range(batch_size):
                sub_start, sub_end = self.sub_entity_token[f'[S:{subject_type[i]}]'], self.sub_entity_token[f'[/S:{subject_type[i]}]']
                obj_start, obj_end = self.obj_entity_token[f'[O:{object_type[i]}]'], self.obj_entity_token[f'[/O:{object_type[i]}]']

                sub_start_cond = (new_input_ids[i] == sub_start).bool()
                obj_start_cond = (new_input_ids[i] == obj_start).bool()
                sub_end_cond = (new_input_ids[i] == sub_end).bool()
                obj_end_cond = (new_input_ids[i] == obj_end).bool()

                special_hs_list.append(hidden_states[i][sub_start_cond+obj_start_cond+sub_end_cond+obj_end_cond].view(-1, hidden_size))

            special_hs = torch.stack(special_hs_list, dim=0).view(batch_size, -1).to(self.dv)

            logits2 = self.net(special_hs).view(batch_size, -1)
            logits2 = self.classifier(logits2)

            outputs = (logits2,) + outputs2[2:]

            if labels is not None:
                loss_fun1 = torch.nn.BCELoss()
                loss_fun2 = torch.nn.CrossEntropyLoss()
                
                loss1 = loss_fun1(torch.argmax(logits1.view(-1, self.model_config1.num_labels), dim=1).float(), tmp_binary_labels.view(-1).float())
                loss2 = loss_fun2(logits2.view(-1, self.model_config2.num_labels), labels.view(-1))

                loss = loss1 + loss2
            
                outputs = (loss,) + outputs

            return outputs 
        elif self.model_type == 'entity_punct':
            outputs1 = self.plm1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits1 = outputs1['logits']
            logits1_cpu = logits1.detach().cpu().numpy()
            result1 = np.argmax(logits1_cpu, axis=-1)

            new_sentences = list()
            tmp_binary_labels = list()

            for i in range(batch_size):
                sentence = self.tokenizer.decode(input_ids[i])[6:].replace('[PAD]', '').rstrip() + " "
                drp_sentence = subject_words[i] + "와(과) " + object_words[i] + DRP_dict[result1[i]] + ": " + f"{object_words[i]}는 {subject_words[i]}의 {object_type[i]}이다."
                new_sentences.append(sentence+drp_sentence)

                if labels != None:
                    label = 0 if labels[i] == 0 else 1
                    tmp_binary_labels.append(label)

            new_tokenized_sentence = self.tokenizer(
                new_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True, 
            )

            new_input_ids = new_tokenized_sentence['input_ids'].view(batch_size, -1).to(self.dv)
            new_attention_mask = new_tokenized_sentence['attention_mask'].view(batch_size, -1).to(self.dv)
            new_token_type_ids = new_tokenized_sentence['token_type_ids'].view(batch_size, -1).to(self.dv)
            tmp_binary_labels = torch.tensor(tmp_binary_labels).view(batch_size, -1).to(self.dv)

            outputs2 = self.plm2(new_input_ids, attention_mask=new_attention_mask, token_type_ids=new_token_type_ids)
            hidden_states = outputs2.last_hidden_state
            batch_size, seq_length, hidden_size = hidden_states.shape

            special_hs_list = list()
 
            for i in range(batch_size):
                sub_start, sub_end = self.sub_entity_token[f'[S:{subject_type[i]}]'], self.sub_entity_token[f'[/S:{subject_type[i]}]']
                obj_start, obj_end = self.obj_entity_token[f'[O:{object_type[i]}]'], self.obj_entity_token[f'[/O:{object_type[i]}]']

                sub_start_cond = (new_input_ids[i] == self.sub_ids).bool()
                obj_start_cond = (new_input_ids[i] == self.obj_ids).bool()

                special_hs_list.append(hidden_states[i][sub_start_cond+obj_start_cond].view(-1, hidden_size))

            special_hs = torch.stack(special_hs_list, dim=0)

            logits2 = self.net(special_hs).view(batch_size, -1)
            logits2 = self.classifier(logits2)

            outputs = (logits2,) + outputs2[2:]

            if labels is not None:
                loss_fun1 = torch.nn.BCELoss()
                loss_fun2 = torch.nn.CrossEntropyLoss()
                
                loss1 = loss_fun1(torch.argmax(logits1.view(-1, self.model_config1.num_labels), dim=1).float(), tmp_binary_labels.view(-1).float())
                loss2 = loss_fun2(logits2.view(-1, self.model_config2.num_labels), labels.view(-1))

                loss = loss1 + loss2
            
                outputs = (loss,) + outputs

            return outputs 