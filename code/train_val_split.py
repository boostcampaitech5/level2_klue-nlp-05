import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from seed_everything import seed_everything

# train 경로 지정 (수정 가능)


def train_val_split(ratio: float = 0.0):
    seed_everything()
    train = pd.read_csv("/opt/ml/dataset/train/train.csv")

    # dataset에서 validation의 비율 (수정 가능)
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

        train_data, validation_data = train_test_split(
            sub_dataset, test_size=ratio)

        train_dataset.append(train_data)
        validation_dataset.append(validation_data)

    train = pd.concat(train_dataset, ignore_index=True)
    validation = pd.concat(validation_dataset, ignore_index=True)

    train = train.sample(frac=1).reset_index(drop=True)
    validation = validation.sample(frac=1).reset_index(drop=True)

    train.to_csv('/opt/ml/code/save_split_dataset/train.csv')
    validation.to_csv('/opt/ml/code/save_split_dataset/dev.csv')
