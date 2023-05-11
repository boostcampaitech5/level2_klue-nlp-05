import pickle as pickle
import os
import pandas as pd
import torch
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence

sub_type_dict = {'PER': '인물', 'ORG': '단체', 'LOC': '장소'}
obj_type_dict = {'PER': '인물', 'ORG': '단체', 'POH': '명칭', 'DAT': '날짜', 'LOC': '장소', 'NOH': '수'}

drp_discription = [
    '주체와 객체는 관계가 없다.', # 0 no relation, everything
    '주체와 객체는 관계가 있다: 객체 인물은 주체 단체의 대표자 또는 구성원이다.', # 1 org:top_members/employees, org, per
    '주체와 객체는 관계가 있다: 객체 단체는 주체 단체에 속한다.', # 2 org:members, org, org
    '주체와 객체는 관계가 있다: 객체 명칭은 주체 단체가 생산한 제품 또는 상품이다.', # 3 org:product, org, poh
    '주체와 객체는 관계가 있다: 객체 명칭은 주체 인물의 직업적 위치를 나타내는 공식 또는 비공식 명칭이다.', # 4 per:title, per, poh
    '주체와 객체는 관계가 있다: 객체 명칭은 주체 단체를 지칭하기 위해 공식 이름 대신 부르는 대체 이름이다.', # 5 org:alternate_names, org, poh
    '주체와 객체는 관계가 있다: 객체 단체는 주체 인물이 근무하는 조직이다.', # 6 per:employee_of, per, org
    '주체와 객체는 관계가 있다: 객체 장소는 주체 단체의 본사가 위치한 장소이다.', # 7 org:place_of_headquarters, org, loc
    '주체와 객체는 관계가 있다: 객체 명칭은 주체 인물이 생산한 제품 또는 예술 작품이다.', # 8 per:product, per, poh
    '주체와 객체는 관계가 있다: 객체 수는 주체 단체에 소속된 총 구성원 수이다.', # 9 org:number_of_employees/members, org, noh
    '주체와 객체는 관계가 있다: 객체 인물은 주체 인물의 자녀이다.', # 10 per:children, per, per
    '주체와 객체는 관계가 있다: 객체 장소는 주체 인물이 살고 있는 장소이다.', # 11 per:place_of_residence, per, loc
    '주체와 객체는 관계가 있다: 객체 명칭은 주체 인물을 지칭하기 위해 공식 이름 대친 부르는 대체 이름이다.', # 12 per:alternate_names, per, poh
    '주체와 객체는 관계가 있다: 객체 인물은 주체 인물의 부모, 자녀, 형제자매 및 배우자를 제외한 가족 구성원이다.', # 13 per:other_family, per, per
    '주체와 객체는 관계가 있다: 객체 인물은 주체 인물과 함께 일하는 사람들이다.', # 14 per:colleagues, per, per
    '주체와 객체는 관계가 있다: 객체 장소 또는 명칭은 주체 인물의 출신지 또는 국적이다.', # 15 per:origin, per, loc/poh
    '주체와 객체는 관계가 있다: 객체 인물은 주체 인물의 형제자매들이다.', # 16 per:siblings, per, per
    '주체와 객체는 관계가 있다: 객체 인물은 주체 인물의 배우자이다.', # 17 per:spouse, per, per
    '주체와 객체는 관계가 있다: 객체 날짜는 주체 단체가 설립된 날짜이다.', # 18 org:founded, org, dat
    '주체와 객체는 관계가 있다: 객체 단체는 주체 단체가 소속된 정치/종교 단체이다.', # 19 org:political/religious_affilication, org, org
    '주체와 객체는 관계가 있다: 객체 단체은 주체 단체가 속한 단체이다.', # 20 org:member_of, org, org
    '주체와 객체는 관계가 있다: 객체 인물은 주체 인물의 부모이다.', # 21 per:parents, per, per
    '주체와 객체는 관계가 있다: 객체 날짜는 주체 단체가 해산된 날짜이다.', # 22 org:dissovled, org, dat
    '주체와 객체는 관계가 있다: 객체 명칭은 주체 인물이 재학 중인 학교이다.', # 23 per:schools_attend, per, poh
    '주체와 객체는 관계가 있다: 객체 날짜는 주체 인물이 사망한 날짜이다.', # 24 per:date_of_death, per, dat
    '주체와 객체는 관계가 있다: 객체 날짜는 주체 인물의 생년월일이다.', # 25 per:date_of_birth, per, dat
    '주체와 객체는 관계가 있다: 객체 장소는 주체 인물이 태어난 장소이다.', # 26 per:place_of_birth, per, loc
    '주체와 객체는 관계가 있다: 객체 장소는 주체 인물이 사망한 장소이다.', # 27 per:place_of_death, per, loc
    '주체와 객체는 관계가 있다: 객체 인물 또는 단체는 주체 단체를 설립한 인물 또는 단체이다.', # 28 org:founded_by, org, per/org
    '주체와 객체는 관계가 있다: 객체 명칭은 주체 인물이 믿는 종교이다.', # 29 per:religion, per, poh
]
        
type_nums = {
    'PER, PER': [0, 10, 13, 14, 16, 17, 21], 'PER, ORG': [0, 6], 'PER, POH': [0, 4, 8, 12, 15, 23, 29], 'PER, DAT': [0, 24, 25], 'PER, LOC': [0, 11, 15, 26, 27], 'PER, NOH': [0],
    'ORG, PER': [0, 1, 28], 'ORG, ORG': [0, 2, 19, 20, 28], 'ORG, POH': [0, 3, 4, 5], 'ORG, DAT': [0, 18, 22], 'ORG, LOC': [0, 7], 'ORG, NOH': [0, 9],
    'LOC, PER': [0], 'LOC, ORG': [0], 'LOC, POH': [0], 'LOC, DAT': [0], 'LOC, LOC': [0], 'LOC, NOH': [0],
    }
        
class DRP_RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, att_mask_tensor, no_predict_tensor, labels):
      self.input_tensor = input_tensor
      self.att_mask_tensor = att_mask_tensor
      self.no_predict_tensor = no_predict_tensor
      self.labels = labels
      
    def __getitem__(self, idx):
      item = dict()
      item['input_tensor'] = self.input_tensor[idx]
      item['att_mask_tensor'] = self.att_mask_tensor[idx]
      item['no_predict_tensor'] = self.no_predict_tensor[idx]
      item['labels'] = torch.tensor(self.labels[idx])
      
      return item
    
    def __len__(self):
        return len(self.labels)      

def drp_preprocessing_dataset(dataset):
  type = []
  sentences = []  
    
  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity, obj_entity = eval(sub_entity), eval(obj_entity)
    sub_word, obj_word = sub_entity['word'], obj_entity['word']
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_start, sub_end, obj_start, obj_end = '[u1]', '[u2]', '[u3]', '[u4]'
    
    if sub_idx[0] < obj_idx[0]:
        sentence = (sentence[:sub_idx[0]] + '[u1]' + sub_word + f'(주체 {sub_type_dict[sub_type]})' + '[u2]'
                    + sentence[sub_idx[1]+1: obj_idx[0]] + '[u3]' + obj_word + f'(객체 {obj_type_dict[obj_type]})' + '[u4]'
                    + sentence[obj_idx[1]+1:])
    else:
        sentence = (sentence[:obj_idx[0]] + '[u3]' + obj_word + f'(객체 {obj_type_dict[obj_type]})' + '[u4]'
                    + sentence[obj_idx[1]+1: sub_idx[0]] + '[u1]' + sub_word + f'(주체 {sub_type_dict[sub_type]})' + '[u2]'
                    + sentence[sub_idx[1]+1:])
    
    type = sub_type + ', ' + obj_type
    restrict_label = type_nums[type]
    
    drp_sentence = ''
    
    for i in range(30):
        if i in restrict_label:
            drp_sentence += sentence + '[SEP]' + drp_discription[i] + '@'
        else:
            drp_sentence += '#@'
        
    sentences.append(drp_sentence)
    
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'],})
  
  return out_dataset
   
    
def drp_load_data(dataset_dir, model_type):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
      
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = drp_preprocessing_dataset(pd_dataset)

  return dataset

def drp_tokenized_dataset(dataset, tokenizer):
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        padding=False,
        add_special_tokens=True
        )
    
    splt = tokenizer.convert_tokens_to_ids('@')
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id
    cls = tokenizer.cls_token_id
    no = tokenizer.convert_tokens_to_ids('#')
    
    input_tensor = []
    att_mask_tensor = []
    no_predict_tensor = []
    
    for i in range(len(tokenized_sentences['input_ids'])):
            
      sublists = [[cls]+list(group)+[sep] for key, group in groupby(tokenized_sentences['input_ids'][i], lambda x: x == splt) if not key]
      sublists.pop()
      sublists[0].pop(0)
            
      no_predict_idx = [1 if sublist == [cls, no, sep] else 0 for index, sublist in enumerate(sublists)]
      no_predict_idx = torch.tensor(no_predict_idx).unsqueeze(1) # (30, 1)
      no_predict_tensor.append(no_predict_idx)
           
      padded_tensor = pad_sequence([torch.tensor(seq) for seq in sublists], batch_first=True, padding_value=pad) # 30, max_seqlen
      att_mask = torch.where(padded_tensor == pad, torch.tensor(0.0), torch.tensor(1.0)) # 30, max_seqlen

      input_tensor.append(padded_tensor)
      att_mask_tensor.append(att_mask)
      
    return input_tensor, att_mask_tensor,no_predict_tensor 