import pickle as pickle
import os
import pandas as pd
import torch

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
  
class RE_special_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels, entity_type):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.entity_type = entity_type

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    item['subject_type'] = self.entity_type['subject_type'].iloc[idx]
    item['object_type'] = self.entity_type['object_type'].iloc[idx]
    
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_words = []
  object_words = []

  for sub_entity, obj_entity in zip(dataset['subject_entity'], dataset['object_entity']):
    sub_word = eval(sub_entity)['word']
    obj_word = eval(obj_entity)['word']

    subject_words.append(sub_word)
    object_words.append(obj_word)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'], 'subject_entity':subject_words, 'object_entity':object_words, 'label':dataset['label'],})
  return out_dataset

def special_preprocessing_dataset(dataset):
  sentences = []
  subject_type = []
  object_type = []

  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)

    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    subject_type.append(sub_type)
    object_type.append(obj_type)

    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_start, sub_end = f'[S:{sub_type}] ', f' [/S:{sub_type}]'
    obj_start, obj_end = f'[O:{obj_type}] ', f' [/O:{obj_type}]'

    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + sub_start + sub_entity['word'] + sub_end 
                  + sentence[sub_idx[1]+1:obj_idx[0]] + obj_start + obj_entity['word']
                  + obj_end + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + obj_start + obj_entity['word'] + obj_end 
                  + sentence[obj_idx[1]+1:sub_idx[0]] + sub_start + sub_entity['word']
                  + sub_end + sentence[sub_idx[1]+1:])
      
    sentences.append(sentence)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'subject_type':subject_type, 'object_type':object_type})
  return out_dataset 

def cls_special_preprocessing_dataset(dataset) :
  '''목표 : input의 sentence의 형태를 바꿔주어야한다.'''
  sentences = []
  subject_type = []
  object_type = []

  for subj_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    subj_entity = eval(subj_entity)
    obj_entity = eval(obj_entity)

    subj_type, obj_type = subj_entity['type'], obj_entity['type']
    subject_type.append(subj_type)
    object_type.append(obj_type)
    subj_idx , obj_idx = subj_entity['start_idx'], obj_entity['start_idx']

    if subj_idx < obj_idx:
      sentence = (sentence[:subj_idx] + '[SUBJ]' + subj_type + sentence[subj_idx:obj_idx] + '[OBJ]'
                  + obj_type + sentence[obj_idx:])
    else:
      sentence = (sentence[:obj_idx] + '[OBJ]' + obj_type + sentence[obj_idx:subj_idx] + '[SUBJ]'
                  + subj_type + sentence[subj_idx:])
    sentences.append(sentence)

  output_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'subject_type':subject_type, 'object_type':object_type})
  return output_dataset

def punct_preprocessing_dataset(dataset):
  sentences = []
  
  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    
    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + f'@ § {sub_type} § ' + sub_entity['word'] + ' @'
                  + sentence[sub_idx[1]+1:obj_idx[0]] + f'# ^ {obj_type} ^ ' + obj_entity['word']
                  + ' #' + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + f'@ § {obj_type} § ' + obj_entity['word'] + ' @'
                  + sentence[obj_idx[1]+1:sub_idx[0]] + f'# ^ {sub_type} ^ ' + sub_entity['word']
                  + ' #' + sentence[sub_idx[1]+1:])
      # ex) 〈Something〉는 @ § PER § 조지 해리슨 @이 쓰고 # ^ ORG ^ 비틀즈 #가 1969년 앨범 《Abbey Road》에 담은 노래다
      
    sentences.append(sentence)
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'],})
  
  return out_dataset

def load_data(dataset_dir, model_type):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)

  if model_type == 'entity_special':
    dataset = special_preprocessing_dataset(pd_dataset)
  elif model_type == 'entity_punct':
    dataset = punct_preprocessing_dataset(pd_dataset)
  elif model_type == 'cls_entity_special':
    dataset = cls_special_preprocessing_dataset(pd_dataset)
  else:
    dataset = preprocessing_dataset(pd_dataset)

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

def special_tokenized_dataset(dataset, tokenizer):
  tokenized_sentences = tokenizer(
    list(dataset['sentence']),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
    add_special_tokens=True,
    )
  return tokenized_sentences, dataset[['subject_type', 'object_type']]

def punct_tokenized_dataset(dataset, tokenizer):
  tokenized_sentences = tokenizer(
    list(dataset['sentence']),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
    add_special_tokens=True,
    )
  return tokenized_sentences