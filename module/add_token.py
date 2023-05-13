def add_token(tokenizer, model_type):
    if model_type == 'entity_special':
        num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens":['[S:PER]', '[/S:PER]', '[O:PER]', '[/O:PER]',
            '[S:ORG]', '[/S:ORG]', '[O:ORG]', '[/O:ORG]', '[O:POH]', '[/O:POH]',
            '[O:DAT]', '[/O:DAT]', '[S:LOC]', '[/S:LOC]', '[O:LOC]', '[/O:LOC]', '[O:NOH]', '[/O:NOH]']})
        
        '''
        num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens":['[SS]', '[SE]', '[OS]', '[OE]']})
        '''
   
    elif model_type == 'entity_punct':
        tokenizer.add_tokens(['§'])
    
    return tokenizer