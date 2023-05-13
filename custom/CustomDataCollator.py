from transformers import DataCollatorWithPadding
import torch
import copy

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # DataCollatorWithPadding 에 들어가기 전에 subject_type과 object_type를 제외시킨다.
        features_2 = copy.deepcopy(features)
        for i in range(len(features_2)):
            del features_2[i]['subject_type']
            del features_2[i]['object_type']
            
        batch = super().__call__(features_2)
        
        # model의 forward 에는 들어가야 하므로, 다시 복원.
        batch['subject_type'] = [feature['subject_type'] for feature in features]
        batch['object_type'] = [feature['object_type'] for feature in features]
        
        return batch

class SequentialDoubleBertDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        features_2 = copy.deepcopy(features)
        for i in range(len(features_2)):
            del features_2[i]['subject_type']
            del features_2[i]['object_type']
            del features_2[i]['subject_words']
            del features_2[i]['object_words']
            
        batch = super().__call__(features_2)
        
        batch['subject_type'] = [feature['subject_type'] for feature in features]
        batch['object_type'] = [feature['object_type'] for feature in features]
        batch['subject_words'] = [feature['subject_words'] for feature in features]
        batch['object_words'] = [feature['object_words'] for feature in features]
        
        return batch