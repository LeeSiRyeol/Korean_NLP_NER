import pandas as pd
from torch.utils import data
import torch

def Data_Load(path):
    load_data = pd.read_csv(path)
    
    load_text = pd.DataFrame(load_data['text'])
    load_class = pd.DataFrame(load_data['class'])
    
    return load_text, load_class

class DATA(data.Dataset):
    def __init__(self, txt, label, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.txt = txt
        self.label = label
        self.seq_len = seq_len

    def __len__(self):
        return len(self.txt)

    def __getitem__(self,idx):
        text = self.txt['text'][idx]
        
        label = self.label['class'][idx]
        
        token = self.tokenizer(text)
        
        user_mask = []
        token_type_ids = []
        type_ids = 0
        label_list = []
        
        num = 0
        label_num = 0
        add_num = 0
        for i in token:
            if i[0] == '#':
                user_mask.append(0)
                label_list.append(0)
            else:
                user_mask.append(1)
                
                if (label != []) and len(label)>label_num:
                    if num == label[label_num][0] and num == label[label_num][1]:
                        label_list.append(1)
                        label_num = label_num+1
                        
                    elif num == label[label_num][0] and num != label[label_num][1]:
                        label_list.append(1)
                        add_num = 2
                    
                    elif num != label[label_num][0] and num == label[label_num][1]:
                        label_list.append(add_num)
                        add_num = 0
                        label_num = label_num+1
                    
                    else:
                        label_list.append(add_num)
                else:
                    label_list.append(add_num)
                    
                num = num+1
                
                
                
            token_type_ids.append(type_ids)
            
            if i == '[SEP]':
                type_ids = type_ids + 1
        
        input_ids = self.tokenizer.convert_tokens_to_ids(token)    
        attention_mask = [1] * len(input_ids)
        
                
        return input_ids, attention_mask, token_type_ids, label_list, user_mask
        
    
    def pad(self,batch):
        temp = lambda x: [sample[x] for sample in batch]
        # seq_len = [len(sample[0]) for sample in batch]
        
        input_ids = temp(0)
        attention_mask = temp(1)
        token_type_ids = temp(2)
        label_list = temp(3)
        user_mask = temp(4)
        max_len = self.seq_len#np.array(seq_len).max()
        
        padding = lambda x, value, seqlen: torch.tensor([sample + [value] * (seqlen - len(sample)) for sample in x], dtype=torch.int64)
        
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, max_len)
        attention_mask = padding(attention_mask, 0, max_len)
        token_type_ids = padding(token_type_ids, self.tokenizer.pad_token_type_id, max_len)
        user_mask = padding(user_mask,0,max_len)
        label_list = padding(label_list, 0, max_len)
        
        return input_ids, attention_mask, token_type_ids, label_list, user_mask
    
    
class DATA_kor(data.Dataset):
    def __init__(self, txt, label, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.txt = txt
        self.label = label
        self.seq_len = seq_len

    def __len__(self):
        return len(self.txt)

    def __getitem__(self,idx):
        text = self.txt['text'][idx]
        
        label = self.label['class'][idx]
        
        token = self.tokenizer.tokenize(text)
        
        user_mask = []
        token_type_ids = []
        type_ids = 0
        label_list = []
        
        num = 0
        label_num = 0
        add_num = 0
        for i in token:
            if i[0] == '#':
                user_mask.append(0)
                label_list.append(0)
            else:
                user_mask.append(1)
                
                if (label != []) and len(label)>label_num:
                    if num == label[label_num][0] and num == label[label_num][1]:
                        label_list.append(1)
                        label_num = label_num+1
                        
                    elif num == label[label_num][0] and num != label[label_num][1]:
                        label_list.append(1)
                        add_num = 2
                    
                    elif num != label[label_num][0] and num == label[label_num][1]:
                        label_list.append(add_num)
                        add_num = 0
                        label_num = label_num+1
                    
                    else:
                        label_list.append(add_num)
                else:
                    label_list.append(add_num)
                    
                num = num+1
                
                
                
            token_type_ids.append(type_ids)
            
            if i == '[SEP]':
                type_ids = type_ids + 1
        
        input_ids = self.tokenizer.convert_tokens_to_ids(token)    
        attention_mask = [1] * len(input_ids)
        
                
        return input_ids, attention_mask, token_type_ids, label_list, user_mask
        
    
    def pad(self,batch):
        temp = lambda x: [sample[x] for sample in batch]
        # seq_len = [len(sample[0]) for sample in batch]
        
        input_ids = temp(0)
        attention_mask = temp(1)
        token_type_ids = temp(2)
        label_list = temp(3)
        user_mask = temp(4)
        max_len = self.seq_len#np.array(seq_len).max()
        
        padding = lambda x, value, seqlen: torch.tensor([sample + [value] * (seqlen - len(sample)) for sample in x], dtype=torch.int64)
        
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, max_len)
        attention_mask = padding(attention_mask, 0, max_len)
        token_type_ids = padding(token_type_ids, self.tokenizer.pad_token_type_id, max_len)
        user_mask = padding(user_mask,0,max_len)
        label_list = padding(label_list, 0, max_len)
        
        return input_ids, attention_mask, token_type_ids, label_list, user_mask 
