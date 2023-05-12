from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import yaml
import pandas as pd
from module.mlm_dataset import TextDatasetForMaskedLanguageModeling
from sklearn.model_selection import train_test_split

def MLM(model_name):
    with open('/opt/ml/module/config.yaml') as f:
        CFG = yaml.safe_load(f)
    
    # load config, model, tokenizer
    #config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer.vocab_size)
    print(model.get_input_embeddings())
    # prepare data
    train = pd.read_csv(CFG['TRAIN_PATH'])
    sentences = train['sentence']
    sentences = [sentence + '\n' for sentence in sentences]

    train, dev = train_test_split(sentences, test_size=CFG['RATIO'], random_state=14)
    
    # save data
    with open('/opt/ml/dataset/train/mlm_train.txt', 'w', encoding='UTF8') as f:
        f.writelines(train)
    with open('/opt/ml/dataset/train/mlm_dev.txt', 'w', encoding='UTF8') as f:
        f.writelines(dev)
    
    # load data
    train_dataset = TextDatasetForMaskedLanguageModeling(
    tokenizer=tokenizer,
    file_path='/opt/ml/dataset/train/mlm_train.txt',
    block_size=128,
    overwrite_cache=True,
    short_seq_probability=0.1,
    nsp_probability=0.0,
    )

    dev_dataset = TextDatasetForMaskedLanguageModeling(
    tokenizer=tokenizer,
    file_path='/opt/ml/dataset/train/mlm_dev.txt',
    block_size=128,
    overwrite_cache=True,
    short_seq_probability=0.1,
    nsp_probability=0.0,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    # tmp1=data_collator(train_dataset.examples)
    # tmp2=data_collator(dev_dataset.examples)
    # print(tmp2)

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
        report_to="wandb", 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    trainer.train()
    model.save_pretrained(CFG['MODEL_SAVE_DIR'])

def main():
    MLM('klue/roberta-small')

if __name__ == '__main__':
    main()