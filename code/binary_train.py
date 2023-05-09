import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle as pickle
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from binary_load_data import *
import wandb
import yaml
import json

from custom.CustomModel import *
from custom.CustomDataCollator import *

from module.seed_everything import seed_everything
from module.binary_train_val_split import train_val_split
from module.add_token import add_token

def remain_klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['org:top_members/employees', 'org:members',
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
    # no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    # label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def binary_klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'relation']
    # no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    # label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0
  
def remain_klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(29)[labels]

    score = np.zeros((29,))
    for c in range(29):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0
  
def binary_klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(2)[labels]

    score = np.zeros((2,))
    for c in range(2):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def remain_compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  f1 = remain_klue_re_micro_f1(preds, labels)
  auprc = remain_klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds)

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }
  
def binary_compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  f1 = binary_klue_re_micro_f1(preds, labels)
  auprc = binary_klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds)

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('/opt/ml/dataset/dict_label_to_num_relation.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  with open('/opt/ml/module/config.yaml') as f:
    CFG = yaml.safe_load(f)

  MODEL_NAME = CFG['MODEL_NAME']
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer = add_token(tokenizer, CFG['MODEL_TYPE'])

  if CFG['RATIO'] == 0.0:
    binary_train_dataset = load_data(CFG['TRAIN_PATH'], CFG['MODEL_TYPE'], binary=True)
    binary_dev_dataset = load_data(CFG['DEV_PATH'], CFG['MODEL_TYPE'], binary=True)
    remain_train_dataset = load_data('/opt/ml/relation_train.csv', CFG['MODEL_TYPE'], binary=False)
    remain_dev_dataset = load_data('/opt/ml/relation_train.csv', CFG['MODEL_TYPE'], binary=False)
  else:
    train_val_split(CFG['RATIO'], binary=True)
    binary_train_dataset = load_data(CFG['SPLIT_TRAIN_PATH'], CFG['MODEL_TYPE'], binary=True)
    binary_dev_dataset = load_data(CFG['SPLIT_DEV_PATH'], CFG['MODEL_TYPE'], binary=True)
    train_val_split(CFG['RATIO'], binary=False)
    remain_train_dataset = load_data('/opt/ml/dataset/save_split_dataset/train_relation.csv', CFG['MODEL_TYPE'], binary=False)
    remain_dev_dataset = load_data('/opt/ml/dataset/save_split_dataset/dev_relation.csv', CFG['MODEL_TYPE'], binary=False)

  binary_train_label= list(binary_train_dataset['label'].values)
  binary_dev_label = list(binary_dev_dataset['label'].values)
  remain_train_label = label_to_num(remain_train_dataset['label'].values)
  remain_dev_label = label_to_num(remain_dev_dataset['label'].values)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  binary_model_config = AutoConfig.from_pretrained(MODEL_NAME)
  binary_model_config.num_labels = 2
  remain_model_config = AutoConfig.from_pretrained(MODEL_NAME)
  remain_model_config.num_labels = 29
  
  if CFG['MODEL_TYPE'] == 'base':
    # tokenizing dataset
    binary_tokenized_train = tokenized_dataset(binary_train_dataset, tokenizer)
    binary_tokenized_dev = tokenized_dataset(binary_dev_dataset, tokenizer)
    remain_tokenized_train = tokenized_dataset(remain_train_dataset, tokenizer)
    remain_tokenized_dev = tokenized_dataset(remain_dev_dataset, tokenizer)

    # make dataset for pytorch.
    binary_train_dataset = RE_Dataset(binary_tokenized_train, binary_train_label)
    binary_dev_dataset = RE_Dataset(binary_tokenized_dev, binary_dev_label)
    remain_train_dataset = RE_Dataset(remain_tokenized_train, remain_train_label)
    remain_dev_dataset = RE_Dataset(remain_tokenized_dev, remain_dev_label)

    binary_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=binary_model_config)
    remain_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=remain_model_config)

    data_collator = DataCollatorWithPadding(tokenizer)

  elif CFG['MODEL_TYPE'] == 'entity_special':
    binary_tokenized_train, binary_entity_type_train = special_tokenized_dataset(binary_train_dataset, tokenizer)
    binary_tokenized_dev, binary_entity_type_dev = special_tokenized_dataset(binary_dev_dataset, tokenizer)
    remain_tokenized_train, remain_entity_type_train = special_tokenized_dataset(remain_train_dataset, tokenizer)
    remain_tokenized_dev, remain_entity_type_dev = special_tokenized_dataset(remain_dev_dataset, tokenizer)

    binary_train_dataset = RE_special_Dataset(binary_tokenized_train, binary_train_label, binary_entity_type_train)
    binary_dev_dataset = RE_special_Dataset(binary_tokenized_dev, binary_dev_label, binary_entity_type_dev)
    remain_train_dataset = RE_special_Dataset(remain_tokenized_train, remain_train_label, remain_entity_type_train)
    remain_dev_dataset = RE_special_Dataset(remain_tokenized_dev, remain_dev_label, remain_entity_type_dev)
    
    binary_model = SepecialEntityBERT(MODEL_NAME, config=binary_model_config, tokenizer=tokenizer)
    remain_model = SepecialEntityBERT(MODEL_NAME, config=remain_model_config, tokenizer=tokenizer)

    data_collator = CustomDataCollator(tokenizer)

  elif CFG['MODEL_TYPE'] == 'entity_punct':
    binary_tokenized_train = punct_tokenized_dataset(binary_train_dataset, tokenizer)
    binary_tokenized_dev = punct_tokenized_dataset(binary_dev_dataset, tokenizer)
    remain_tokenized_train = punct_tokenized_dataset(remain_train_dataset, tokenizer)
    remain_tokenized_dev = punct_tokenized_dataset(remain_dev_dataset, tokenizer)

    binary_train_dataset = RE_Dataset(binary_tokenized_train, binary_train_label)
    binary_dev_dataset = RE_Dataset(binary_tokenized_dev, binary_dev_label)
    remain_train_dataset = RE_Dataset(remain_tokenized_train, remain_train_label)
    remain_dev_dataset = RE_Dataset(remain_tokenized_dev, remain_dev_label)

    binary_model = SepecialPunctBERT(MODEL_NAME, config=binary_model_config, tokenizer=tokenizer)
    remain_model = SepecialPunctBERT(MODEL_NAME, config=remain_model_config, tokenizer=tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer)

  print(binary_model.config)
  binary_model.parameters
  binary_model.to(device)

  print(remain_model.config)
  remain_model.parameters
  remain_model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  if CFG['SWEEP_AVAILABLE']:
    wandb.init()
    cfg = wandb.config
    wandb.run.name = f"{cfg.lr}, {cfg.epochs}"

    training_args = TrainingArguments(
      output_dir=CFG['OUTPUT_DIR'],          # output directory
      save_total_limit=CFG['TOTAL_SAVE_MODEL'],              # number of total save model.
      save_steps=CFG['SAVING_STEP'],                 # model saving step.
      num_train_epochs=cfg.epochs,              # total number of training epochs
      learning_rate=cfg.lr,               # learning_rate
      per_device_train_batch_size=cfg.batch_size,  # batch size per device during training
      per_device_eval_batch_size=cfg.batch_size,   # batch size for evaluation
      warmup_steps=CFG['WARMUP_STEP'],                # number of warmup steps for learning rate scheduler
      weight_decay=CFG['WEIGHT_DECAY'],               # strength of weight decay
      logging_dir=CFG['LOGGING_DIR'],            # directory for storing logs
      logging_steps=CFG['LOGGING_STEP'],              # log saving step.
      evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      eval_steps=CFG['EVAL_STEP'],            # evaluation step.
      load_best_model_at_end=True,
      report_to="wandb",
      metric_for_best_model='micro f1 score'
    )
  else:
    wandb.init(project=CFG['WANDB_PROJECT'], name=CFG['WANDB_NAME'])
    training_args = TrainingArguments(
      output_dir=CFG['OUTPUT_DIR'],          # output directory
      save_total_limit=CFG['TOTAL_SAVE_MODEL'],              # number of total save model.
      save_steps=CFG['SAVING_STEP'],                 # model saving step.
      num_train_epochs=CFG['MAX_EPOCH'],              # total number of training epochs
      learning_rate=CFG['LR'],               # learning_rate
      per_device_train_batch_size=CFG['BATCH_SIZE'],  # batch size per device during training
      per_device_eval_batch_size=CFG['BATCH_SIZE'],   # batch size for evaluation
      warmup_steps=CFG['WARMUP_STEP'],                # number of warmup steps for learning rate scheduler
      weight_decay=CFG['WEIGHT_DECAY'],               # strength of weight decay
      logging_dir=CFG['LOGGING_DIR'],            # directory for storing logs
      logging_steps=CFG['LOGGING_STEP'],              # log saving step.
      evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      eval_steps=CFG['EVAL_STEP'],           # evaluation step.
      load_best_model_at_end=True,
      report_to="wandb", 
      metric_for_best_model='micro f1 score'
    )
    
  trainer_binary = Trainer(
    model=binary_model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=binary_train_dataset,         # training dataset
    eval_dataset=binary_dev_dataset,             # evaluation dataset
    compute_metrics=binary_compute_metrics,        # define metrics function
    data_collator=data_collator,
  )
  
  trainer_remain = Trainer(
    model=remain_model,
    args=training_args,
    train_dataset=remain_train_dataset,
    eval_dataset=remain_dev_dataset,
    compute_metrics=remain_compute_metrics,
    data_collator=data_collator
  )

  # train model
  trainer_binary.train()
  
  model_to_save = binary_model.state_dict()
  torch.save(model_to_save, './best_model/pytorch_binary_model.bin')
  config_dict = binary_model.config.to_dict()
  with open('./best_model/binary_config.json', 'w') as f:
    json.dump(config_dict, f)

  trainer_remain.train()
  
  model_to_save = remain_model.state_dict()
  torch.save(model_to_save, './best_model/pytorch_remain_model.bin')
  config_dict = remain_model.config.to_dict()
  with open('./best_model/remain_config.json', 'w') as f:
    json.dump(config_dict, f)
    
def main():
  train()

if __name__ == '__main__':
  # Seed 고정
  seed_everything()

  with open('/opt/ml/module/config.yaml') as f:
    CFG = yaml.safe_load(f)

  if CFG['SWEEP_AVAILABLE']:
    sweep_configuration = CFG['SWEEP_CONFIGURATION']
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=CFG['WANDB_PROJECT'])
    wandb.agent(sweep_id=sweep_id, function=main, count=CFG['SWEEP_COUNT'])

  main()