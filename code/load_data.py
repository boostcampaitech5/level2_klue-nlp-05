import pickle as pickle
import os
import pandas as pd
import torch
# import hanja

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
  
class RE_source_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels, source):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.source = source

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    item['source'] = self.source['source'].iloc[idx]
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

class RE_sequential_doublebert_Dataset(torch.utils.data.Dataset):
  def __init__(self, pair_dataset, labels, entity_type, entity_words):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.entity_type = entity_type
    self.entity_words = entity_words

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    item['subject_type'] = self.entity_type['subject_type'].iloc[idx]
    item['object_type'] = self.entity_type['object_type'].iloc[idx]
    item['subject_words'] = self.entity_words['subject_words'].iloc[idx]
    item['object_words'] = self.entity_words['object_words'].iloc[idx]

    return item

  def __len__(self):
    return len(self.labels)
  
def sequentialdoublebert_preprocessing_dataset(dataset, model_type):
  sentences = []
  subject_type = []
  object_type = []
  subject_words = []
  object_words = []

  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)

    sub_word = f"' {sub_entity['word']} '"
    obj_word = f"' {sub_entity['word']} '"

    subject_words.append(sub_word)
    object_words.append(obj_word)

    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    subject_type.append(sub_type)
    object_type.append(obj_type)
      
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    if model_type == 'entity_special':
      sub_start, sub_end = f'[S:{sub_type}] ', f' [/S:{sub_type}]'
      obj_start, obj_end = f'[O:{obj_type}] ', f' [/O:{obj_type}]'

      if sub_idx[0] < obj_idx[0]:
        sentence = (sentence[:sub_idx[0]] + " " + sub_start + " " + sub_entity['word'] + " " + sub_end + " "
                  + sentence[sub_idx[1]+1:obj_idx[0]] + " " + obj_start + " " + obj_entity['word'] + " "
                  + obj_end + " " + sentence[obj_idx[1]+1:])
      else:
        sentence = (sentence[:obj_idx[0]] + " " + obj_start + " " + obj_entity['word'] + " " + obj_end + " "
                  + sentence[obj_idx[1]+1:sub_idx[0]] + " " + sub_start + " " + sub_entity['word'] + " "
                  + sub_end + " " + sentence[sub_idx[1]+1:])
    elif model_type == 'entity_punct':
      if sub_idx[0] < obj_idx[0]:
        sentence = (sentence[:sub_idx[0]] + f'@ § {sub_type} § ' + sub_entity['word'] + ' @'
                  + sentence[sub_idx[1]+1:obj_idx[0]] + f'# ^ {obj_type} ^ ' + obj_entity['word']
                  + ' #' + sentence[obj_idx[1]+1:])
      else:
        sentence = (sentence[:obj_idx[0]] + f'@ § {obj_type} § ' + obj_entity['word'] + ' @'
                  + sentence[obj_idx[1]+1:sub_idx[0]] + f'# ^ {sub_type} ^ ' + sub_entity['word']
                  + ' #' + sentence[sub_idx[1]+1:])
        
    sentences.append(sentence)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'subject_type':subject_type, 'object_type':object_type, 'subject_words':subject_words, 'object_words':object_words, 'label':dataset['label']})

  return out_dataset

def add_discription(sentence, sub_word, obj_word, obj_type):
  discription = f"이 문장에서{obj_word}는{sub_word}의{obj_type}이다." # 자체에 앞 뒤 띄어쓰기와 '있음.
  sentence = sentence + ':' + discription
  
  return sentence
  
def preprocessing_dataset(dataset, discrip):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  sentences = []
  subject_words = []
  object_words = []
  sentences = []

  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_word = eval(sub_entity)['word']
    obj_word = eval(obj_entity)['word']
    sub_word = f' \'{sub_word}\' '
    obj_word = f' \'{obj_word}\' '

    # 한자 -> 한글
    # sub_word = hanja.translate(sub_word, 'substitution')
    # obj_word = hanja.translate(obj_word, 'substitution')
    # sentence = hanja.translate(sentence, 'substitution')
    
    subject_words.append(sub_word)
    object_words.append(obj_word)
    
    if discrip:
      sentence = add_discription(sentence, sub_word, obj_word, eval(obj_entity)['type'])
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
    
    #sub_start, sub_end, obj_start, obj_end = '[SS]', '[SE]', '[OS]', '[OE]'
    

    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + sub_start + sub_word + sub_end 
                  + sentence[sub_idx[1]+1:obj_idx[0]] + obj_start + obj_word
                  + obj_end + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + obj_start + obj_word + obj_end 
                  + sentence[obj_idx[1]+1:sub_idx[0]] + sub_start + sub_word
                  + sub_end + sentence[sub_idx[1]+1:])
    
    # 한자 -> 한글
    # sentence = hanja.translate(sentence, 'substitution')
    
    if discrip:
      sentence = add_discription(sentence, sub_word, obj_word, f" \'{obj_type}\' ")
      
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

def punct_preprocessing_dataset(dataset, discrip):
  sentences = []
  sources = []
  
  for sub_entity, obj_entity, sentence, source in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence'], dataset['source']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "
    
    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + f'@ §{sub_type}§' + sub_word + '@'
                  + sentence[sub_idx[1]+1:obj_idx[0]] + f'# ^{obj_type}^' + obj_word
                  + '#' + sentence[obj_idx[1]+1:] + f' [{source}]')
    else:
      sentence = (sentence[:obj_idx[0]] + f'# ^{obj_type}^' + obj_word + '#'
                  + sentence[obj_idx[1]+1:sub_idx[0]] + f'@ §{sub_type}§' + sub_word
                  + '@' + sentence[sub_idx[1]+1:] + f' [{source}]')
      # ex) 〈Something〉는 @ § PER § 조지 해리슨 @이 쓰고 # ^ ORG ^ 비틀즈 #가 1969년 앨범 《Abbey Road》에 담은 노래다
    
    # 한자 -> 한글
    # sentence = hanja.translate(sentence, 'substitution')
    
    if discrip:
      sentence = add_discription(sentence, sub_word, obj_word, f" \'{obj_type}\' ")
      
    sentences.append(sentence)
    sources.append(f'[{source}]')
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'source':sources})
  
  return out_dataset

def new_punct_preprocessing_dataset(dataset, discrip):
  sentences = []
  sources = []
  
  for sub_entity, obj_entity, sentence, source in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence'], dataset['source']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "
    
    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + f'@ § {sub_type_dict[sub_type]} §{sub_word}@' + sentence[sub_idx[1]+1:obj_idx[0]]
                  + f'# ^ {obj_type_dict[obj_type]} ^{obj_word}#' + sentence[obj_idx[1]+1:] + f' [{source}]')
    else:
      sentence = (sentence[:obj_idx[0]] + f'@ ^ {obj_type_dict[obj_type]} ^{obj_word}@' + sentence[obj_idx[1]+1:sub_idx[0]]
                  + f'# § {sub_type_dict[sub_type]} §{sub_word}#' + sentence[sub_idx[1]+1:] + f' [{source}]')
    # "영화 '기생충'이 제92회 아카데미 시상식에서 4관왕의 영광을 안은 가운데, 한국계 # ^ 장소 ^ '캐나다' # 배우 @ § 인물 § '산드라 오' @가 '기생충' 수상에 보인 반응이 화제다."
    
    if discrip:
      sentence = add_discription(sentence, sub_word, obj_word, f" \'{obj_type}\' ")
      
    sentences.append(sentence)
    sources.append(f'[{source}]')
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'source':sources})
  
  return out_dataset  

def load_data(dataset_dir, model_type, discrip, do_sequentialdoublebert=0):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  
  if do_sequentialdoublebert:
    dataset = sequentialdoublebert_preprocessing_dataset(pd_dataset, model_type)
  elif model_type == 'entity_special':
    dataset = special_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'entity_punct':
    dataset = punct_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'new_entity_punct':
    dataset = new_punct_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'cls_entity_special' or model_type == "sangmin_entity_special":
    dataset = cls_special_preprocessing_dataset(pd_dataset)
  else:
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
    max_length=512,
    add_special_tokens=True,
    )
  return tokenized_sentences, dataset[['source']]

def sequentialdoublebert_tokenized_dataset(dataset, tokenizer, model_type):
  if model_type == 'base':
    concat_entity = []
    for e01, e02 in zip(dataset['subject_words'], dataset['object_words']):
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
  else:
    tokenized_sentences = tokenizer(
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
    )

  return tokenized_sentences, dataset[['subject_type', 'object_type']], dataset[['subject_words', 'object_words']]