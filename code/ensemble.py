import yaml
import pandas as pd
import numpy as np
from inference import num_to_label

with open('/opt/ml/module/config.yaml') as f:
    CFG = yaml.safe_load(f)

def str_to_lst(probs): # output.csv의 probs는 str형태
    '''
    input : "[0.123, 0.245, ... 0.567]" : str
    output : [0.123, 0.245, ... 0.567] : list(float)
    '''
    result = probs[1:len(probs)-1].split(',')
    result = list(map(float, result))
    return result

def elementwise_sum(lst1, lst2):
    return list(np.array(lst1) + np.array(lst2))

def elementwise_divide(lst, n):
    return list(np.array(lst)/n)

def main():
    if CFG['FOLD']:
        path = '/opt/ml/prediction/fold_csv'
        fold_dfs = []
        for fold_n in range(CFG["FOLD_N"]):
            fold_dfs.append(pd.read_csv(path+f"/fold{fold_n+1}.csv"))
        probs = [[0.0]*30 for _ in range(len(fold_dfs[0]))]

        for fold_n in range(len(fold_dfs)):
            for id in range(len(fold_dfs[0])):
                prob = str_to_lst(fold_dfs[fold_n].loc[id,"probs"])
                probs[id] = elementwise_sum(probs[id], prob)
        
        # normalize
        output_prob = [elementwise_divide(prob, CFG["FOLD_N"]) for prob in probs]

        # output
        test_id = fold_dfs[0]["id"]
        pred_answer = num_to_label([np.argmax(np.array(prob)) for prob in output_prob])
        output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
        
        # save
        output.to_csv(path+"/ensemble_output.csv", index=False)

    elif CFG['SEED']:
        path = '/opt/ml/prediction/seed_csv'
        seed_dfs = []
        for seed_n in range(CFG["SEED_N"]):
            seed_dfs.append(pd.read_csv(path+f"/seed{seed_n+1}.csv"))
        probs = [[0.0]*30 for _ in range(len(seed_dfs[0]))]

        for seed_n in range(len(seed_dfs)):
            for id in range(len(seed_dfs[0])):
                prob = str_to_lst(seed_dfs[seed_n].loc[id,"probs"])
                probs[id] = elementwise_sum(probs[id], prob)
        
        # normalize
        output_prob = [elementwise_divide(prob, CFG["SEED_N"]) for prob in probs]

        # output
        test_id = seed_dfs[0]["id"]
        pred_answer = num_to_label([np.argmax(np.array(prob)) for prob in output_prob])
        output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
        
        # save
        output.to_csv(path+"/ensemble_output.csv", index=False)

if __name__ == "__main__":
    main()