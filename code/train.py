import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle as pickle
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from load_data import *
import wandb
import yaml

from custom.CustomModel import *
from custom.CustomDataCollator import *
from custom.CustomTrainer import *

from module.seed_everything import seed_everything
from module.train_val_split import train_val_split
from module.add_token import add_token, add_token_ver2

label_list = ['no_relation', 'org:top_members/employees', 'org:members', 'org:product', 'per:title', 'org:alternate_names',
      'per:employee_of', 'org:place_of_headquarters', 'per:product',
      'org:number_of_employees/members', 'per:children',
      'per:place_of_residence', 'per:alternate_names',
      'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
      'per:spouse', 'org:founded', 'org:political/religious_affiliation',
      'org:member_of', 'per:parents', 'org:dissolved',
      'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
      'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
      'per:religion']
  
def viz(labels, preds, probs):
  """vizualize ROC, PR graph and confusion matrix"""
  wandb.log(
    {
    "auprc": wandb.plot.roc_curve(labels, probs, labels=label_list),
    "precision_recall": wandb.plot.pr_curve(labels, probs, labels=label_list),  
    "Confusion Matrix" : wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=label_list),
    })

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions
  
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds)
  
  viz(labels, preds, probs)
    
  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('/opt/ml/dataset/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  with open('/opt/ml/module/config.yaml') as f:
    CFG = yaml.safe_load(f)

  MODEL_NAME = CFG['MODEL_NAME']
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  if CFG['MODEL_TYPE'] !='cls_entity_special' and CFG['MODEL_TYPE'] != 'base':
    tokenizer = add_token(tokenizer, CFG['MODEL_TYPE'])

  if CFG['RATIO'] == 0.0:
    train_val_split(0.1)
    train_dataset = load_data(CFG['TRAIN_PATH'], CFG['MODEL_TYPE'], CFG['DISCRIP'], CFG['DO_SEQUENTIALBERTMODEL'])
    dev_dataset = load_data(CFG['SPLIT_DEV_PATH'], CFG['MODEL_TYPE'], CFG['DISCRIP'], CFG['DO_SEQUENTIALBERTMODEL'])
  else:
    train_val_split(CFG['RATIO'])
    train_dataset = load_data(CFG['SPLIT_TRAIN_PATH'], CFG['MODEL_TYPE'], CFG['DISCRIP'], CFG['DO_SEQUENTIALBERTMODEL'])
    dev_dataset = load_data(CFG['SPLIT_DEV_PATH'], CFG['MODEL_TYPE'], CFG['DISCRIP'], CFG['DO_SEQUENTIALBERTMODEL'])

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30
  
  if CFG['DO_SEQUENTIALBERTMODEL']:
    tokenized_train, entity_type_train, entity_words_train = sequentialdoublebert_tokenized_dataset(train_dataset, tokenizer, CFG['MODEL_TYPE'])
    tokenized_dev, entity_type_dev, entity_words_dev = sequentialdoublebert_tokenized_dataset(dev_dataset, tokenizer, CFG['MODEL_TYPE'])

    RE_train_dataset = RE_sequential_doublebert_Dataset(tokenized_train, train_label, entity_type_train, entity_words_train)
    RE_dev_dataset = RE_sequential_doublebert_Dataset(tokenized_dev, dev_label, entity_type_dev, entity_words_dev)

    model = SequentialDoubleBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer,
                                 model_type=CFG['MODEL_TYPE'], device=device)
    
    data_collator = SequentialDoubleBertDataCollator(tokenizer)
  else:
    if CFG['MODEL_TYPE'] == 'base':
      # tokenizing dataset
      tokenized_train = tokenized_dataset(train_dataset, tokenizer)
      tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

      # make dataset for pytorch.
      RE_train_dataset = RE_Dataset(tokenized_train, train_label)
      RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

      model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

      data_collator = DataCollatorWithPadding(tokenizer)

    elif CFG['MODEL_TYPE'] == 'entity_special':
      tokenized_train, entity_type_train = special_tokenized_dataset(train_dataset, tokenizer)
      tokenized_dev, entity_type_dev = special_tokenized_dataset(dev_dataset, tokenizer)

      RE_train_dataset = RE_special_Dataset(tokenized_train, train_label, entity_type_train)
      RE_dev_dataset = RE_special_Dataset(tokenized_dev, dev_label, entity_type_dev)

      model = SepecialEntityBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

      data_collator = CustomDataCollator(tokenizer)

    elif CFG['MODEL_TYPE'] == 'entity_punct' or CFG['MODEL_TYPE'] == 'new_entity_punct':
      tokenized_train = punct_tokenized_dataset(train_dataset, tokenizer)
      tokenized_dev = punct_tokenized_dataset(dev_dataset, tokenizer)

      RE_train_dataset = RE_Dataset(tokenized_train, train_label)
      RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

      model = SepecialPunctBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

      data_collator = DataCollatorWithPadding(tokenizer)

    elif CFG['MODEL_TYPE'] =='cls_entity_special' :
      '''해야할 것 : tokenizer에 specail token 추가하기. soure를 special 토큰으로 넣기'''
      tokenizer = add_token_ver2(tokenizer)
      tokenized_train, entity_type_train = special_tokenized_dataset(train_dataset, tokenizer)
      tokenized_dev, entity_type_dev = special_tokenized_dataset(dev_dataset, tokenizer)

      RE_train_dataset = RE_special_Dataset(tokenized_train, train_label, entity_type_train)
      RE_dev_dataset = RE_special_Dataset(tokenized_dev, dev_label, entity_type_dev)

      model =CLS_SpecialEntityBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

    elif CFG["MODEL_TYPE"] =='sangmin_entity_special' :
      tokenizer = add_token_ver2(tokenizer)
      tokenized_train, entity_type_train = special_tokenized_dataset(train_dataset, tokenizer)
      tokenized_dev, entity_type_dev = special_tokenized_dataset(dev_dataset, tokenizer)

      RE_train_dataset = RE_special_Dataset(tokenized_train, train_label, entity_type_train)
      RE_dev_dataset = RE_special_Dataset(tokenized_dev, dev_label, entity_type_dev)

      model =sangmin_SpecialEntityBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

  print(model.config)
  model.parameters
  model.to(device)

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
      logging_steps=CFG['LOGGING_STEP'],
      logging_strategy=CFG['STRATEGY'],# log saving step.
      save_strategy=CFG['STRATEGY'],
      evaluation_strategy=CFG['STRATEGY'], # evaluation strategy to adopt during training
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
      logging_steps=CFG['LOGGING_STEP'],
      logging_strategy=CFG['STRATEGY'],# log saving step.
      save_strategy=CFG['STRATEGY'],
      evaluation_strategy=CFG['STRATEGY'], # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      eval_steps=CFG['EVAL_STEP'],           # evaluation step.
      load_best_model_at_end=True,
      report_to="wandb", 
      metric_for_best_model='micro f1 score'
    )
  # trainer = Trainer(
  trainer = CustomTrainer(
    loss_fn=CFG['LOSS_FN'],
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,        # define metrics function
    # data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
  )

  # train model
  trainer.train()
  model.save_pretrained(CFG['MODEL_SAVE_DIR'])
  wandb.finish()
  
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