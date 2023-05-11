import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def attention_heatmap(model, tokenizer, dev_dataset, device, model_type):
    if model_type == 'base':
        text = dev_dataset.loc[0, 'sentence'] # 고코묘 천황(1633년 4월 20일 ~ 1654년 10월 30일)은 제110대 일본 천황으로 재위 기간은 1643년에서 1654년까지이다.
        sub = dev_dataset.loc[0, 'subject_entity']
        obj = dev_dataset.loc[0, 'object_entity']
        encoded = tokenizer(sub + '[SEP]' + obj,
                            text, return_tensors='pt', add_special_tokens=True,)
    else:
        text = dev_dataset.loc[0, 'sentence']
        encoded = tokenizer(text, return_tensors='pt', add_special_tokens=True)
    
    with torch.no_grad():
        output = model(encoded['input_ids'].to(device), encoded['attention_mask'].to(device), output_attentions=True)
        
    if model_type == 'base':    
        attention = torch.cat(output.attentions).to('cpu') # num_layer, num_heads, sequence_length, sequence_length
    else:
        attention = output['attentions'].to('cpu')
        
    attention = attention.permute(2, 1, 0, 3) # sequence_length, num_heads, num_layer, sequence_length
    
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    #layers = len(attention[0][0])
    #h eads = len(attention[0])
    #seqlen = len(attention)
    
    # avg_attention = attention.mean(dim=1).mean(dim=0)
    
    #!wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
    #!mv malgun.ttf /usr/share/fonts/truetype/
    #import matplotlib.font_manager as fm 
    #fm._rebuild()
    # font_path = '/opt/ml/malgun.ttf'
    # font = font_manager.FontProperties(fname=font_path).get_name()
    # rc('font', family=font)
    # plt.rcParams['axes.unicode_minus'] = False
    
    '''
    fig, ax = plt.subplots(figsize=(15, 15))
    
    sns.heatmap(avg_attention, vmin=0, vmax=1, xticklabels=tokens, ax=ax)
    ax.set_title('attention~~~~')
    
    plt.savefig(f'/opt/ml/attention_heatmap/attention_heatmap.png')
    plt.close()
    
    '''
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    os.mkdir(f'/opt/ml/attention_heatmap/{current_time}')
    
    for i in range(len(tokens)):
        attentions_pos = attention[i]
        avg_attention = attentions_pos.mean(dim=0)
        
        fig, ax = plt.subplots(figsize=(15, 5))
        
        sns.heatmap(avg_attention, vmin=0, vmax=1, xticklabels=tokens, ax=ax)
        ax.set_title(f'token - {tokens[i]}')
        ax.set_ylabel('layers')
        
        plt.savefig(f'/opt/ml/attention_heatmap/{current_time}/attention_heatmap_{i}_{tokens[i]}.png')
        plt.close()

    
    
    
