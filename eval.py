import numpy as np
from tqdm import tqdm
import argparse
import os
import random

from seqeval.metrics import f1_score, classification_report

import torch
from torch import nn
from torch.utils import data
import torch.optim as optim

from module.dataload import Data_Load
from module.NLP_model import NER

class Processing(nn.Module):
    def __init__(self,train_batch, val_batch, test_batch, label_list, param_list):
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = test_batch
        self.label_list = label_list
        self.param_list = param_list
        self.f1_list = list()
        self.loss_list = list()
        self.F1_LIST = list()
        

    def __enter__(self):
        print("Train Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Train Exit")
        
    def Test(self,model):
        true_labels = []
        predicted_labels = []
        logits_list = []
        with torch.no_grad():
            print('---------------Test Data Prediction Start---------------')
            model.eval()
            for i, batch in enumerate(self.test_batch):
                input_ids, attention_mask, token_type_ids, labels, user_mask= batch
                _, _, y_hat = model(input_ids,attention_mask, labels)
                
                for batch_idx in range(len(y_hat)):
                    _y_hat = y_hat[batch_idx].cpu().numpy()
                    _labels = labels[batch_idx].cpu().numpy()
                    _user_mask = user_mask[batch_idx].cpu().numpy()
                    
                    _y_hat = _y_hat[_user_mask==1]
                    _labels = _labels[_user_mask==1]
                    
                    true_labels.append(np.take(self.label_list,_labels).tolist())
                    predicted_labels.append(np.take(self.label_list,_y_hat).tolist())

        f1 = f1_score(true_labels, predicted_labels)
        self.F1_LIST.append(f1)
        self.REPORT = classification_report(true_labels, predicted_labels)

        self.PREDICTED_LABELS = predicted_labels
        self.TRUE_LABELS = true_labels
        self.LOGIT_LIST = logits_list
        print('---------------Test Data Prediction Finish---------------')
            
    def train_result(self):
        return self.f1_list, self.report, self.loss_list, self.model_file, self.predicted_labels, self.true_labels, self.logits_list
    
    def test_result(self):
        return self.F1_LIST, self.REPORT, self.PREDICTED_LABELS, self.TRUE_LABELS, self.LOGIT_LIST

def save_model(trained_model):
    torch.save(trained_model, './trained_model.txt')
    


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--epoch", type=int , default=10)
    parser.add_argument("--NLP_model", type=str, default='KLUE-RoBERTa')
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--random_seed", type=int, default=102)
    parser.add_argument("--data_path", type=str, default='./data')  
    args = parser.parse_args()
    
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    pct_start = args.pct_start
    epoch = args.epoch
    NLP_model = args.NLP_model
    seq_len = args.seq_len
    data_path = args.data_path
    random_seed = args.random_seed
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if NLP_model == 'KLUE-RoBERTa':
        from module.dataload import DATA
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base",TOKENIZERS_PARALLELISM=True)
        
    elif NLP_model == 'KLUE-BERT':
        from module.dataload import DATA
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base",TOKENIZERS_PARALLELISM=True)
        
    elif NLP_model == 'KoBERT':
        from module.dataload import DATA
        from kobert_tokenizer import KoBERTTokenizer
        tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1",TOKENIZERS_PARALLELISM=True)
        
    elif NLP_model == 'KoBigBird':
        from module.dataload import DATA
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base",TOKENIZERS_PARALLELISM=True)
        seq_len = 1024
        
    elif NLP_model == 'KorBERT':
        from module.dataload import DATA_kor as DATA
        from module.kor_tensorflow.src_tokenizer import tokenization
        from module.kor_tensorflow.src_tokenizer.tokenization import BertTokenizer
        tokenizer_path = os.path.join('./module/kor_tensorflow', 'vocab.korean.rawtext.list')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    label_list = ['O','B-', 'I-']

    test_txt, test_class = Data_Load(data_path+'/example_data.csv') #  +'/test.csv
    
    test_data = DATA(test_txt, test_class, tokenizer, seq_len)
    
    test_batch = data.DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=test_data.pad)
    
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    print('Parameter Informations*******\n lr : ',lr,'\n weight_decay : ',weight_decay,'\n pct_start : ',pct_start)

    with NER(head='Dense', backbone=NLP_model, device=device, label_num=len(label_list)).cuda() as model:                
        model_path = "./trained_model.txt"
        model.load_state_dict(torch.load(model_path))
        
        with Processing(test_batch, label_list) as test:            
            test.Test(model)
            F1_LIST, REPORT, PREDICTED_LABELS, TRUE_LABELS, LOGIT_LIST = train.test_result() 

    
