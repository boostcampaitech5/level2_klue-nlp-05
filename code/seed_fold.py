from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import yaml
import train
import inference

from module.seed_everything import seed_everything

class SeedFold:
    def __init__(self, CFG, seeds: list): 
        # prepare cross validation
        self.CFG = CFG
        self.seeds = seeds
        self.n_fold = self.CFG["FOLD_N"]
        self.train_data = pd.DataFrame(pd.read_csv(CFG["TRAIN_PATH"]))
        self.label = self.train_data['label']
        
    def fold(self):
        cnt = 1
        for seed in self.seeds:
            seed_everything(seed)
            kf = StratifiedKFold(n_splits=self.n_fold, random_state=seed, shuffle=True)
            for train_index, dev_index in kf.split(self.train_data, self.label):

                # splitting Dataframe (dataset not included)
                train_df = self.train_data.iloc[train_index]
                train_df.to_csv(self.CFG["SEED_FOLD_TRAIN_PATH"])
                dev_df = self.train_data.iloc[dev_index]
                dev_df.to_csv(self.CFG["SEED_FOLD_DEV_PATH"])
                print("!!!! train, dev split complite !!!!")
                
                # training
                train.main()
                # inference
                inference.main(cnt)
                
                cnt += 1
                print(f'!!! no.{cnt} fold finished !!!')

    def __repr__():
                print("=========== SEED_FOLD FINISHED ===========")


def main():
    with open("/opt/ml/module/config.yaml") as f:
        CFG = yaml.safe_load(f)
    seeds = [77, 72]
    seed_fold = SeedFold(CFG, seeds)
    seed_fold.fold()
    print(seed_fold)
    
    
if __name__ == "__main__":     
    main()