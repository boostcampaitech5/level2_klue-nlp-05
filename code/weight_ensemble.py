import pandas as pd
import numpy as np
from inference import num_to_label
from ensemble import str_to_lst, elementwise_sum, elementwise_divide
import os

def elementwise_multiplication(lst, n):
    return list(np.array(lst)*n)

def main(path, weights):
    csvs = os.listdir(path)
    csvs.sort()
    dfs = []
    for csv in csvs:
        dfs.append(pd.read_csv(path+f"/{csv}.csv"))
    probs = [[0.0]*30 for _ in range(len(dfs[0]))]

    for i in range(len(dfs)):
        for id in range(len(dfs[0])):
            prob = str_to_lst(dfs[i].loc[id,"probs"])
            probs[id] = elementwise_sum(probs[id], elementwise_multiplication(prob, weights[i]))
    
    # normalize
    output_prob = [elementwise_divide(prob, sum(weights)) for prob in probs]

    # output
    test_id = dfs[0]["id"]
    pred_answer = num_to_label([np.argmax(np.array(prob)) for prob in output_prob])
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    
    # save
    output.to_csv(path+"/weighted_ensemble_output.csv", index=False)

if __name__ == "__main__":
    path = "/opt/ml/prediction/weight_ensemble_csv" # 해당 경로에 csv 다 넣어주세요
    weights = [] # csv 순으로 weights 꼭 지정해주세요!
    main(path, weights)