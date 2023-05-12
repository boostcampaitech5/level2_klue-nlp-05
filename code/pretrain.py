import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import yaml
import wandb
from module.domain_adaptation import MLM

if __name__=='__main__':
    with open('/opt/ml/module/config.yaml') as f:
        CFG = yaml.safe_load(f)
    wandb.init(project=CFG['WANDB_PROJECT'], name=CFG['WANDB_NAME'])
    
    if CFG['PRE_TRAIN'] == 'MLM':
        MLM(CFG['MODEL_NAME'])
