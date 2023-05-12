import pickle as pickle
import os
import pandas as pd
import torch

sub_type_dict = {'PER': '인물', 'ORG': '단체', 'LOC': '장소'}
obj_type_dict = {'PER': '인물', 'ORG': '단체', 'POH': '명칭', 'DAT': '날짜', 'LOC': '장소', 'NOH': '수'}
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

def special_preprocessing_dataset(dataset, discrip):
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
    sub_start, sub_end = f'[S:{sub_type}]', f'[/S:{sub_type}]'
    obj_start, obj_end = f'[O:{obj_type}]', f'[/O:{obj_type}]'
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "

    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + sub_start + sub_word + sub_end 
                  + sentence[sub_idx[1]+1:obj_idx[0]] + obj_start + obj_word
                  + obj_end + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + obj_start + obj_word + obj_end 
                  + sentence[obj_idx[1]+1:sub_idx[0]] + sub_start + sub_word
                  + sub_end + sentence[sub_idx[1]+1:])
    if discrip:
      sentence = add_discription(sentence, sub_word, obj_word, f" \'{obj_type}\' ")
      
    sentences.append(sentence)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'subject_type':subject_type, 'object_type':object_type})
  return out_dataset 

def punct_preprocessing_dataset(dataset, discrip):
  sentences = []
  
  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "
    
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
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'],})
  
  return out_dataset

def new_punct_preprocessing_dataset(dataset, discrip):
  sentences = []
  
  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "
    
    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + f'@{sub_word}(주체 {sub_type_dict[sub_type]}) @' + sentence[sub_idx[1]+1:obj_idx[0]]
                  + f'#{obj_word}(객체 {obj_type_dict[obj_type]}) #' + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + f'#{obj_word}(객체 {obj_type_dict[obj_type]}) #' + sentence[obj_idx[1]+1:sub_idx[0]]
                  + f'@{sub_word}(주체 {sub_type_dict[sub_type]}) @' + sentence[sub_idx[1]+1:])
    # "영화 '기생충'이 제92회 아카데미 시상식에서 4관왕의 영광을 안은 가운데, 한국계 # '캐나다' (객체 장소) # 배우 @ '산드라 오' (주체 인물) @가 '기생충' 수상에 보인 반응이 화제다."
  
    if discrip:
      sentence = add_discription(sentence, sub_word, obj_word, f" \'{obj_type}\' ")
      
    sentences.append(sentence)
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'],})
  
  return out_dataset  

def load_data(dataset_dir, model_type, discrip):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)

  if model_type == 'entity_special':
    dataset = special_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'entity_punct':
    dataset = punct_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'new_entity_punct':
    dataset = new_punct_preprocessing_dataset(pd_dataset, discrip)
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