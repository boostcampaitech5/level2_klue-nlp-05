import pickle as pickle
import os
import pandas as pd
import torch

type_to_num = {'PER, PER': 0, 'PER, ORG': 1, 'PER, POH': 2, 'PER, DAT': 3, 'PER, LOC': 4, 'PER, NOH': 5,
                'ORG, PER': 6, 'ORG, ORG': 7, 'ORG, POH': 8, 'ORG, DAT': 9, 'ORG, LOC': 10, 'ORG, NOH': 1}

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)
  

class RE_Restrict_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels, restrict_num):
        self.pair_dataset = pair_dataset
        self.labels = labels
        self.restric_num = restrict_num

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['restrict_num'] = torch.tensor(self.restric_num[idx])
        
        return item

    def __len__(self):
        return len(self.labels)
    

def add_discription(sentence, sub_word, obj_word, obj_type):
  
  discription = f"이 문장에서{obj_word}는{sub_word}의{obj_type}이다." # 자체에 앞 뒤 띄어쓰기와 '있음.

  sentence = sentence + ':' + discription
  
  return sentence
  
def preprocessing_dataset(dataset, discrip):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_words = []
  object_words = []
  sentences = []

  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_word = eval(sub_entity)['word']
    obj_word = eval(obj_entity)['word']
    sub_word = f' \'{sub_word}\' '
    obj_word = f' \'{obj_word}\' '

    subject_words.append(sub_word)
    object_words.append(obj_word)
    
    if discrip:
      sentence = add_discription(sentence, sub_word, obj_word, eval(obj_entity)['type'])
      sentences.append(sentence)
    else:
      sentences.append(sentence)
    
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'subject_entity':subject_words, 'object_entity':object_words, 'label':dataset['label'],})
  return out_dataset

def punct_preprocessing_dataset(dataset, discrip):
  sentences = []
  restrict_num = []
  
  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "
    
    if sub_type == 'LOC':
        restrict_num.append(-1)
    else:
        restrict_num.append(type_to_num[sub_type + ', ' + obj_type])
    
    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + f'@ §{sub_type}§' + sub_word + '@'
                  + sentence[sub_idx[1]+1:obj_idx[0]] + f'# ^{obj_type}^' + obj_word
                  + '#' + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + f'# ^{obj_type}^' + obj_word + '#'
                  + sentence[obj_idx[1]+1:sub_idx[0]] + f'@ §{sub_type}§' + sub_word
                  + '@' + sentence[sub_idx[1]+1:])
      # ex) 〈Something〉는 @ § PER § 조지 해리슨 @이 쓰고 # ^ ORG ^ 비틀즈 #가 1969년 앨범 《Abbey Road》에 담은 노래다
    
    if discrip:
      sentence = add_discription(sentence, sub_word, obj_word, f" \'{obj_type}\' ")
      
    sentences.append(sentence)
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'restrict_num': restrict_num})
  
  return out_dataset


def load_data(dataset_dir, model_type, discrip):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)

  if model_type == 'entity_punct':
    dataset = punct_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'base':
    dataset = preprocessing_dataset(pd_dataset, discrip)

  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

def punct_tokenized_dataset(dataset, tokenizer):
  tokenized_sentences = tokenizer(
    list(dataset['sentence']),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
    add_special_tokens=True,
    )
  return tokenized_sentences, dataset['restrict_num']