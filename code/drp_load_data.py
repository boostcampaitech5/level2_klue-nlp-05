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
    '주체와 객체는 관계가 있다: 객체 인물/명칭/단체는 주체 단체의 대표자 또는 구성원이다.', # 1 org:top_members/employees, org, per/poh/org/                      loc/noh -> 4195, 52, 22, 13, 1
    '주체와 객체는 관계가 있다: 객체 단체/장소/명칭은 주체 단체에 속한다.', # 2 org:members, org, org/loc/poh/                                          dat -> 285, 97, 32, 1
    '주체와 객체는 관계가 있다: 객체 명칭/단체는 주체 단체가 생산한 제품 또는 상품이다.', # 3 org:product, org, poh/org/                            loc/noh -> 304, 48, 24, 1
    '주체와 객체는 관계가 있다: 객체 명칭/단체/장소는 주체 인물의 직업적 위치를 나타내는 공식 또는 비공식 명칭이다.', # 4 per:title, per, poh/org/loc/              noh/per/dat -> 1845, 141, 81, 19, 14, 3
    '주체와 객체는 관계가 있다: 객체 단체/명칭/인물은 주체 단체를 지칭하기 위해 공식 이름 대신 부르는 대체 이름이다.', # 5 org:alternate_names, org, org/poh/per/   loc/dat -> 1154, 106, 31, 23, 1
    '주체와 객체는 관계가 있다: 객체 단체/인물/장소/명칭은 주체 인물이 근무하는 조직이다.', # 6 per:employee_of, per, org/per/loc/poh/                                 dat/noh -> 2857, 391, 165, 135, 19, 6
    '주체와 객체는 관계가 있다: 객체 장소/단체/명칭은 주체 단체의 본사가 위치한 장소이다.', # 7 org:place_of_headquarters, org, loc/org/poh/                      dat -> 894, 254, 39, 4
    '주체와 객체는 관계가 있다: 객체 명칭/단체/인물은 주체 인물이 생산한 제품 또는 예술 작품이다.', # 8 per:product, per, poh/org/per/                             loc -> 120, 11, 6, 2
    '주체와 객체는 관계가 있다: 객체 수는 주체 단체에 소속된 총 구성원 수이다.', # 9 org:number_of_employees/members, org, noh -> 48
    '주체와 객체는 관계가 있다: 객체 인물/명칭은 주체 인물의 자녀이다.', # 10 per:children, per, per/poh/                                                    loc/noh/org/dat -> 276, 22, 3, 2, 1, 1
    '주체와 객체는 관계가 있다: 객체 장소/단체는 주체 인물이 살고 있는 장소이다.', # 11 per:place_of_residence, per, loc/org/                            poh/dat -> 172, 11, 6, 4
    '주체와 객체는 관계가 있다: 객체 인물/명칭/단체는 주체 인물을 지칭하기 위해 공식 이름 대신 부르는 대체 이름이다.', # 12 per:alternate_names, per, per/poh/org/   loc/noh -> 881, 70, 40, 9, 1
    '주체와 객체는 관계가 있다: 객체 인물은 주체 인물의 부모, 자녀, 형제자매 및 배우자를 제외한 가족 구성원이다.', # 13 per:other_family, per, per/     poh/loc/org -> 178, 7, 4, 1
    '주체와 객체는 관계가 있다: 객체 인물/단체는 주체 인물과 함께 일하는 사람들이다.', # 14 per:colleagues, per, per/org/                                  poh/dat/loc -> 512, 10, 10, 1, 1
    '주체와 객체는 관계가 있다: 객체 장소/단체/명칭은 주체 인물의 출신지 또는 국적이다.', # 15 per:origin, per, loc/org/poh/                              dat/per/noh -> 821, 267, 66, 60, 29, 2
    '주체와 객체는 관계가 있다: 객체 인물/명칭은 주체 인물의 형제자매들이다.', # 16 per:siblings, per, per/poh -> 112, 23
    '주체와 객체는 관계가 있다: 객체 인물/명칭은 주체 인물의 배우자이다.', # 17 per:spouse, per, per/poh/                                                     loc/org/dat -> 761, 26, 6, 1, 1
    '주체와 객체는 관계가 있다: 객체 날짜는 주체 단체가 설립된 날짜이다.', # 18 org:founded, org, dat -> 450
    '주체와 객체는 관계가 있다: 객체 단체/명칭/장소는 주체 단체가 소속된 정치/종교 집단이다.', # 19 org:political/religious_affilication, org, org/poh/loc   /dat -> 54, 38, 4, 1
    '주체와 객체는 관계가 있다: 객체 단체/명칭/장소는 주체 단체가 속한 단체이다.', # 20 org:member_of, org, org/poh/loc/                                            noh/dat -> 1320, 358, 173, 9, 5
    '주체와 객체는 관계가 있다: 객체 인물/명칭은 주체 인물의 부모이다.', # 21 per:parents, per, per/poh/                                                      loc/dat/noh -> 443, 64, 7, 5, 1
    '주체와 객체는 관계가 있다: 객체 날짜는 주체 단체가 해산된 날짜이다.', # 22 org:dissovled, org, dat -> 66
    '주체와 객체는 관계가 있다: 객체 단체/장소는 주체 인물이 재학 중인 학교이다.', # 23 per:schools_attend, per, org/loc -> 80, 2
    '주체와 객체는 관계가 있다: 객체 날짜/수는 주체 인물이 사망한 날짜이다.', # 24 per:date_of_death, per, dat/noh                                           per/loc/org -> 410, 2, 3, 2, 1 
    '주체와 객체는 관계가 있다: 객체 날짜/수는 주체 인물이 태어난 날짜이다.', # 25 per:date_of_birth, per, dat/noh -> 1128, 2
    '주체와 객체는 관계가 있다: 객체 장소/단체는 주체 인물이 태어난 장소이다.', # 26 per:place_of_birth, per, loc/org                                          /per/dat -> 161, 3, 1, 1
    '주체와 객체는 관계가 있다: 객체 장소는 주체 인물이 사망한 장소이다.', # 27 per:place_of_death, per, loc                                           /poh/per/org/dat -> 35, 2, 1, 1, 1
    '주체와 객체는 관계가 있다: 객체 인물은 주체 단체를 설립한 인물 또는 단체이다.', # 28 org:founded_by, org, per/                              loc -> 144, 1
    '주체와 객체는 관계가 있다: 객체 단체/명칭은 주체 인물이 믿는 종교이다.', # 29 per:religion, per, org/poh,                                                 loc -> 80, 14, 2
]
        
type_nums = {
    'PER, PER': [0, 6, 8, 10, 12, 13, 14, 16, 17, 21], 'PER, ORG': [0, 4, 6, 8, 11, 12, 14, 15, 23, 26, 29], 'PER, POH': [0, 4, 6, 8, 10, 12, 15, 16, 17, 21, 29], 'PER, DAT': [0, 24, 25], 'PER, LOC': [0, 4, 6, 11, 15, 23, 26, 27], 'PER, NOH': [0, 24, 25],
    'ORG, PER': [0, 1, 5], 'ORG, ORG': [0, 1, 2, 3, 5, 7, 20, 28], 'ORG, POH': [0, 1, 2, 3, 5, 7, 19, 20], 'ORG, DAT': [0, 18, 19, 22], 'ORG, LOC': [0, 2, 7, 19, 20], 'ORG, NOH': [0, 9],
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
      max_seqlen = padded_tensor.size(1)
      
      if max_seqlen > 512:
        padded_tensor = padded_tensor[:, :512]
      else:
        padding = torch.full((30, 512-max_seqlen), pad)
        padded_tensor = torch.cat([padded_tensor, padding], dim=1)
      
      att_mask = torch.where(padded_tensor == pad, torch.tensor(0.0), torch.tensor(1.0)) # 30, max_seqlen

      input_tensor.append(padded_tensor)
      att_mask_tensor.append(att_mask)
      
    return input_tensor, att_mask_tensor,no_predict_tensor 