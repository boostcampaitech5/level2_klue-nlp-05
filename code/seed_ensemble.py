import pandas as pd
import numpy as np
import yaml
import train
import inference

from module.seed_everything import seed_everything

class SeedEnsemble:
    def __init__(self, CFG, seeds: list): 
        # prepare cross validation
        self.CFG = CFG
        self.n_seed = self.CFG["SEED_N"]
        self.seeds = seeds
        self.train_data = pd.DataFrame(pd.read_csv(CFG["TRAIN_PATH"]))
        self.label = self.train_data['label']
        
    def seed_train(self):
        cnt = 1
        self.train_data.to_csv(self.CFG["SEED_TRAIN_PATH"])
        self.train_data.to_csv(self.CFG["SEED_DEV_PATH"])
        for seed in self.seeds:
            seed_everything(seed)
            # training
            train.main()
            # inference
            inference.main(cnt)
            
            cnt += 1
            print(f'!!! no.{cnt} seed finished !!!')

        def __repr__():
            print("=========== SEED ENSEMBLE FINISHED ===========")


def main():
    seeds = [14, 42, 38, 45, 42] # seed 설정해야합니다!!!
    with open("/opt/ml/module/config.yaml") as f:
        CFG = yaml.safe_load(f)
    seed_ensemble = SeedEnsemble(CFG, seeds)
    seed_ensemble.seed_train()
    
if __name__ == "__main__":     
    main()