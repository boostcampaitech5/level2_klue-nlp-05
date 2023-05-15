from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

import train
import inference

from module.seed_everything import seed_everything

class Fold:
    def __init__(self, n_fold): 
        # prepare cross validation
        self.n_fold = n_fold
        self.kf = KFold(n_splits=self.n_fold, random_state=14, shuffle=True)
        self.train_data = pd.DataFrame(pd.read_csv("/opt/ml/dataset/train/remove_duplicated_train.csv"))
        
    def fold(self):
        cnt = 1
        
        for train_index, dev_index in self.kf.split(self.train_data):
            # splitting Dataframe (dataset not included)
            train_df = self.train_data.iloc[train_index]
            train_df.to_csv("/opt/ml/dataset/fold_dataset/fold_train.csv")
            dev_df = self.train_data.iloc[dev_index]
            dev_df.to_csv("/opt/ml/dataset/fold_dataset/fold_dev.csv")

            # training
            train.main()
            # inference
            inference.main(cnt)
            
            cnt += 1
            print(f'!!! no.{cnt} fold finished !!!')

        def __repr__():
            print("=========== KFOLD FINISHED ===========")

def main():
    kfold = Fold(n_fold=5)
    kfold.fold()
    print(kfold)
    
    
if __name__ == "__main__":     
    seed_everything()
    main()