import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
import yaml
from load_data import *
from custom.CustomModel import *
from custom.CustomDataCollator import *
from module.seed_everything import seed_everything
from module.add_token import add_token, add_token_ver2

def inference(model, tokenized_sent, device, model_type, do_sequentialdoublebert=0):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """ 
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      if do_sequentialdoublebert:
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device),
            subject_type=data['subject_type'],
            object_type=data['object_type'],
            subject_words=data['subject_words'],
            object_words=data['object_words'],
            )
      elif model_type == 'entity_special' or model_type == "cls_entity_special" or model_type == "sangmin_entity_special" or model_type == "new_entity_special":
          outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device),
            subject_type=data['subject_type'],
            object_type=data['object_type'],
            )
      else:
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    if do_sequentialdoublebert:
      logits = outputs
    elif model_type == 'base':
      logits = outputs[0]
    else:
      logits = outputs['logits']
      
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)

  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('/opt/ml/dataset/dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer, model_type, discrip, do_sequentialdoublebert=0):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  if do_sequentialdoublebert:
    test_dataset = load_data(dataset_dir, model_type, discrip, do_sequentialdoublebert)
    test_label = list(map(int,test_dataset['label'].values))
    tokenized_test, entity_type, entity_words = sequentialdoublebert_tokenized_dataset(test_dataset, tokenizer, CFG['MODEL_TYPE'])

    return test_dataset['id'], tokenized_test, test_label, entity_type, entity_words
  else:
    if model_type == 'base':
      test_dataset = load_data(dataset_dir, model_type, discrip)
      test_label = list(map(int,test_dataset['label'].values))
      # tokenizing dataset
      tokenized_test = tokenized_dataset(test_dataset, tokenizer)
      return test_dataset['id'], tokenized_test, test_label
    
    elif model_type == 'entity_special':
      test_dataset = load_data(dataset_dir, model_type, discrip)
      test_label = list(map(int,test_dataset['label'].values))
      # tokenizing dataset
      tokenized_test, entity_type = special_tokenized_dataset(test_dataset, tokenizer)
      return test_dataset['id'], tokenized_test, test_label, entity_type
    
    elif model_type == 'entity_punct' or model_type == 'new_entity_punct':
      test_dataset = load_data(dataset_dir, model_type, discrip)
      test_label = list(map(int,test_dataset['label'].values))
      # tokenizing dataset
      tokenized_test = punct_tokenized_dataset(test_dataset, tokenizer)
      return test_dataset['id'], tokenized_test, test_label

    elif model_type == "cls_entity_special" or model_type == "new_entity_special":
      test_dataset = load_data(dataset_dir, model_type, discrip)
      test_label = list(map(int, test_dataset['label'].values))
      # tokenizing dataset
      tokenized_test, entity_type = special_tokenized_dataset(test_dataset, tokenizer)
      return test_dataset['id'], tokenized_test, test_label, entity_type


def main(cnt = None):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  with open('/opt/ml/module/config.yaml') as f:
    CFG = yaml.safe_load(f)
    
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = CFG['MODEL_NAME']
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  tokenizer = add_token(tokenizer, CFG['MODEL_TYPE'])
  MODEL_NAME = CFG['MODEL_SAVE_DIR']
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  test_dataset_dir = CFG['TEST_PATH']

  if CFG['DO_SEQUENTIALBERTMODEL']:
    model = SequentialDoubleBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer,
                                model_type=CFG['MODEL_TYPE'], device=device)
    state_dict = torch.load(f'{MODEL_NAME}/pytorch_model.bin')
    model.load_state_dict(state_dict)

    test_id, test_dataset, test_label, entity_type, entity_words = load_test_dataset(test_dataset_dir, tokenizer, CFG['MODEL_TYPE'], CFG['DISCRIP'], do_sequentialdoublebert=CFG['DO_SEQUENTIALBERTMODEL'])
    RE_test_dataset = RE_sequential_doublebert_Dataset(test_dataset, test_label, entity_type, entity_words)
  else:
    if CFG['MODEL_TYPE'] == 'base':
      model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
      test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, CFG['MODEL_TYPE'], CFG['DISCRIP'])
      Re_test_dataset = RE_Dataset(test_dataset ,test_label)
    
    elif CFG['MODEL_TYPE'] == 'entity_special':       
      model = SepecialEntityBERT(Tokenizer_NAME, model_config, tokenizer) # custom model 에는 내부에 from_pretrained 함수가 없다.
      state_dict = torch.load(f'{MODEL_NAME}/pytorch_model.bin')
      model.load_state_dict(state_dict)
    
      test_id, test_dataset, test_label, entity_type = load_test_dataset(test_dataset_dir, tokenizer, CFG['MODEL_TYPE'], CFG['DISCRIP'])
      Re_test_dataset = RE_special_Dataset(test_dataset ,test_label, entity_type)
  
    elif CFG['MODEL_TYPE'] == 'entity_punct' or CFG['MODEL_TYPE'] == 'new_entity_punct':
      model = SepecialPunctBERT(Tokenizer_NAME, model_config, tokenizer)
      state_dict = torch.load(f'{MODEL_NAME}/pytorch_model.bin')
      model.load_state_dict(state_dict)
    
      test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, CFG['MODEL_TYPE'], CFG['DISCRIP'])
      Re_test_dataset = RE_Dataset(test_dataset ,test_label)
    
    elif CFG["MODEL_TYPE"] == 'cls_entity_special' :
      tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
      tokenizer = add_token_ver2(tokenizer)
      model = CLS_SpecialEntityBERT(Tokenizer_NAME, model_config, tokenizer)

      state_dict = torch.load(f"{MODEL_NAME}/pytorch_model.bin")
      model.load_state_dict(state_dict)

      test_id, test_dataset, test_label, entity_type = load_test_dataset(test_dataset_dir, tokenizer, CFG['MODEL_TYPE'], None)
      Re_test_dataset = RE_special_Dataset(test_dataset ,test_label, entity_type)

    elif CFG["MODEL_TYPE"] == 'sangmin_entity_special' :
      tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
      tokenizer = add_token_ver2(tokenizer)
      model = CLS_SpecialEntityBERT(Tokenizer_NAME, model_config, tokenizer)

      state_dict = torch.load(f"{MODEL_NAME}/pytorch_model.bin")
      model.load_state_dict(state_dict)

      test_id, test_dataset, test_label, entity_type = load_test_dataset(test_dataset_dir, tokenizer, CFG['MODEL_TYPE'], None)
      Re_test_dataset = RE_special_Dataset(test_dataset ,test_label, entity_type)

    elif CFG["MODEL_TYPE"] == "new_entity_special" :
      tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
      tokenizer = add_token_ver2(tokenizer)
      model = New_SpecialEntityBERT(Tokenizer_NAME, model_config, tokenizer)

      state_dict = torch.load(f"{MODEL_NAME}/pytorch_model.bin")
      model.load_state_dict(state_dict)

      test_id, test_dataset, test_label, entity_type = load_test_dataset(test_dataset_dir, tokenizer, CFG['MODEL_TYPE'], None)
      Re_test_dataset = RE_special_Dataset(test_dataset ,test_label, entity_type)


  model.to(device)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device, CFG['MODEL_TYPE'], do_sequentialdoublebert=CFG['DO_SEQUENTIALBERTMODEL']) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  if(CFG['FOLD']):
    path_name = '/opt/ml/prediction/fold_csv'
    file_name = f'/fold{cnt}.csv'
    output.to_csv(path_name + file_name, index=False)
  else:  
    output.to_csv('/opt/ml/prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')

if __name__ == '__main__':
  seed_everything()
  main()