# level2_Relation-Extraction_nlp-05

<br>

## 🐴Members

|<img src='https://avatars.githubusercontent.com/u/102334596?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/86002769?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/88221233?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/107304584?v=' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/60664644?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/126854237?v=4' height=100 width=100px></img>
| --- | --- | --- | --- | --- | --- |
| [변성훈](https://github.com/DNA-B) | [서보성](https://github.com/Seoboseong) | [이도현](https://github.com/aiclaudev) | [이상민](https://github.com/SangMini2) | [이승우](https://github.com/OLAOOT) | [이예원](https://github.com/aeongaewon) |

<br>

## 📎RE (Relation Extraction)

> 부스트 캠프 AI-Tech 5기 NLP 트랙 Level2 1차 경진대회 프로젝트입니다. 관계 추출(Relation Extraction)은 문장이나 텍스트에서 두 개체(대상) 사이의 관계를 식별하고 분류하는 작업입니다. RE 작업은 정보 검색 및 추출, 지식 그래프 구축 등 다양한 응용 분야에서 중요하게 활용됩니다. 프로젝트의 목표는 문장과 문장의 두 개체가 주어졌을 때, 이 두 개체 사이의 관계를 자동으로 추출하도록 하는 것입니다.
> 

<br>

### Data (Private)

- 총 데이터 개수: 40,235 문장 쌍
    - Train(학습) 데이터 개수: 32,470 (81%)
    - Test(평가) 데이터 개수: 7,765 (19%)
    - Label: 0 ~ 29 사이의 정수 ([KLUE](https://arxiv.org/pdf/2105.09680.pdf) RE)

<br>

### Metric

- Micro F1, AUPRC

<br>

## ✔️Project

### Structure

```
root/
|
|-- train.py
|-- inference.py
|
|-- custom/
|   |-- CustomDataCollator.py
|   |-- CustomModel.py
|   |-- CustomTrainer.py
|
|-- module/
|   |-- add_token.py
|   |-- config.yaml
|   |-- load_data.py
|   |-- pretrain.py
|   |-- seed_everything.py
|   |-- train_val_split.py
|
|-- utils/
|   |-- compute_metrics.py
|   |-- ensemble.py
|   |-- kfold.py
|   |-- label_to_num.py
|   |-- num_to_label.py

```


<br>

### Preprocessing

- Data Augmentation
    - Duplicated data remove
    - Token masking for an abnormal label (per:place_of_residence)
- Entity Marker
    - Special Token
        - Normal Special Token Ver1, Ver2(No CLS)
        - Korean Special Token
        - Special Token with CLS
    - Punctuation
        - Normal punctuation
        - Korean Punctuation
- Description
    - Description version 1
    - Description version 2

<br>

### Modeling

- Focal loss
- Label Smoothing
- Add modules to Pre-Trained Model
- Use two types of BERT Models
    - working independently
    - working Sequentially
- Entity Type Restriction

<br>

### Ensemble

- KFold → StratifiedKFold
- Soft Voting Ensemble

<br>

💡 __*자세한 내용은 [Wrap-up Report](https://github.com/boostcampaitech5/level2_klue-nlp-05/blob/main/%5BNLP-05%5Dklue_wrapup_report.pdf)를 참고해주세요.*__

<br>

## 🐞Usage

```python
# TRAIN
python3 code/train.py

# INFERENCE
python3 code/inference.py
```

<br>

## 🏆Result

- Public 1위

![Public](https://github.com/boostcampaitech5/level2_klue-nlp-05/assets/60664644/3ec7b2df-3d85-4a62-abe3-661f220b0bec)

- Private 2위

![Private](https://github.com/boostcampaitech5/level2_klue-nlp-05/assets/60664644/59376906-baf6-4146-b6d3-20f54c7f7806)
