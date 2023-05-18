from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import yaml
import train
import inference

from module.seed_everything import seed_everything

class Fold:
    def __init__(self, CFG): 
        # prepare cross validation
        self.CFG = CFG
        self.n_fold = self.CFG["FOLD_N"]
        self.kf = StratifiedKFold(n_splits=self.n_fold, random_state=3, shuffle=True)
        self.train_data = pd.DataFrame(pd.read_csv(CFG["TRAIN_PATH"]))
        self.label = self.train_data['label']
        
    def fold(self):
        cnt = 1
        
        for train_index, dev_index in self.kf.split(self.train_data, self.label):
            # splitting Dataframe (dataset not included)
            train_df = self.train_data.iloc[train_index]
            train_df.to_csv(self.CFG["FOLD_TRAIN_PATH"])
            dev_df = self.train_data.iloc[dev_index]
            dev_df.to_csv(self.CFG["FOLD_DEV_PATH"])
            print("!!!! train, dev split complete !!!!")
            print("Here we go~~!")
            
            # training
            train.main()
            # inference
            inference.main(cnt)
            
            cnt += 1
            print(f'!!! no.{cnt} fold finished !!!')

        def __repr__():
            print("=========== KFOLD FINISHED ===========")


def main():
    with open("/opt/ml/module/config.yaml") as f:
        CFG = yaml.safe_load(f)
    kfold = Fold(CFG)
    kfold.fold()
    print(kfold)
    
    
if __name__ == "__main__":     
    seed_everything(3)
    main()