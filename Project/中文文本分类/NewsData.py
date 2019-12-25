from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig
from transformers import BertPreTrainedModel
import torch
import torch.nn as nn
from transformers import BertModel
import time
import argparse

class NewsData(torch.utils.data.Dataset):
    def __init__(self,root,is_train = 1):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.data_num = 7346
        self.x_list = []
        self.y_list = []
        self.posi = []
        
        with open(root + '/Train_DataSet.csv',encoding='UTF-8') as f:
            for i in range(self.data_num+1):
                line = f.readline()[:-1] + '这是一个中性的数据'

                
                data_one_str = line.split(',')[len(line.split(','))-2]
                data_two_str = line.split(',')[len(line.split(','))-1]
                

               
                if len(data_one_str) < 6:
                    z = len(data_one_str)
                    data_one_str = data_one_str + '，' + data_two_str[0:min(200,len(data_two_str))]
                else:
                    data_one_str = data_one_str
                if i==0:
                    continue
                

                word_l = self.tokenizer.encode(data_one_str, add_special_tokens=False)
                if len(word_l)<100:
                    while(len(word_l)!=100):
                        word_l.append(0)
                else:
                    word_l = word_l[0:100]
                
                word_l.append(102)
                l = word_l
                word_l = [101]
                word_l.extend(l)
                
                
                self.x_list.append(torch.tensor(word_l))

                self.posi.append(torch.tensor([i for i in range(102)]))
                
        with open(root + '/Train_DataSet_Label.csv',encoding='UTF-8') as f:
            for i in range(self.data_num+1):
                #print(i)
                label_one = f.readline()[-2]
                if i==0:
                    continue
                label_one = int(label_one)
                self.y_list.append(torch.tensor(label_one))
                
        
        
        if is_train == 1:
            self.x_list = self.x_list[0:6000]
            self.y_list = self.y_list[0:6000]
            self.posi = self.posi[0:6000]
        else:
            self.x_list = self.x_list[6000:]
            self.y_list = self.y_list[6000:]
            self.posi = self.posi[6000:]
        
        self.len = len(self.x_list)
        
        
        
        
    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index],self.posi[index]
             
    def __len__(self):
        return self.len
