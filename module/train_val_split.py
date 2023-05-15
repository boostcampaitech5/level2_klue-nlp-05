import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import os
import yaml

def train_val_split(ratio: float):
    
    with open('/opt/ml/module/config.yaml') as f:
        CFG = yaml.safe_load(f)

    train = pd.read_csv(CFG['TRAIN_PATH'])

    lbl_dict = {}
    for lbl in train['label']:
        if lbl in lbl_dict:
            lbl_dict[lbl] += 1
        else:
            lbl_dict[lbl] = 1

    train_dataset = []
    validation_dataset = []

    for item in tqdm(lbl_dict.items(), desc='train_validation_split', total=len(lbl_dict)):
        sub_dataset = train[train['label'] == item[0]]

        train_data, validation_data = train_test_split(sub_dataset, test_size=ratio, random_state=14)

        train_dataset.append(train_data)
        validation_dataset.append(validation_data)

    train = pd.concat(train_dataset, ignore_index=True)
    validation = pd.concat(validation_dataset, ignore_index=True)

    train = train.sample(frac=1, random_state=14).reset_index(drop=True)
    validation = validation.sample(frac=1, random_state=14).reset_index(drop=True)

    if os.path.exists('/opt/ml/dataset/save_split_dataset') == False:
        os.mkdir('/opt/ml/dataset/save_split_dataset')

    train.to_csv('/opt/ml/dataset/save_split_dataset/train.csv')
    validation.to_csv('/opt/ml/dataset/save_split_dataset/dev.csv')