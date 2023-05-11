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
from module.add_token import add_token
from module.attention_heatmap import attention_heatmap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained("./best_model")
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

dev_dataset = load_data("/opt/ml/dataset/save_split_dataset/dev.csv", 'base')
tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

model.to(device)

attention_heatmap(model, tokenizer, dev_dataset, device, 'base')


