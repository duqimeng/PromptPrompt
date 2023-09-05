# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import sys
import numpy as np
import random
from ContrastLoss import contrastive_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
import os
import time

class embedd(nn.Module):                    
    def __init__(self,embedding_dim):
        super(embedd, self).__init__()     
        self.embedding = nn.Embedding(81,embedding_dim, padding_idx=0)

    def forward(self,x):
        x = torch.tensor(x).to(torch.int64)
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)
        return x




def sort_key(s):
    if s:
        try:
            c = s.split('_')[1].split('.')[0]
        except:
            c = -1
        return int(c)
def strsort(alist):
    alist.sort(key=sort_key,reverse=True)
    return alist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#seq = input('请输入要判断的序列：')
#print(seq)
def wirite_encode(ckpt_path):#train_path,test_path,encode_train,encode_test)
    print(train_path,test_path)
    encode_train = 'encode_data/'+name+'/'+name + '_train.txt'
    encode_test = 'encode_data/'+name+'/'+name + '_test.txt'
    path_ = 'encode_data/'+name
    if not os.path.exists(path_):
        os.makedirs(path_)
    #print(encode_train)
    #print(encode_test)
    #exit()
    #ckpt = ckpt_path + '/model_400.pth'
    aa = {'A':0,'T':1,'C':2,'G':3,'N':4,'M':5,'Y':6,'W':7}
    with open(encode_train,'w',encoding = 'utf-8') as f:
        #ckpt_path = glob.glob('ckpt' + '/*')
        
            
        
        for line in tqdm(open(train_path,'r')):
            seq,label = line.strip().split('\t')
            
            sequence = torch.Tensor([aa[i] for i in seq]).to(device)
            import glob

            model = torch.load(ckpt_path)
            model.eval()
            with torch.no_grad():
                sequence = sequence.unsqueeze(0)
                outputs = model(sequence)
                outputs = outputs.tolist()
                f.write(seq + '\t'+ str(label) +'\t'+ str(outputs[0])   + '\n')
        #print(outputs.shape)
    
    
    with open(encode_test,'w',encoding = 'utf-8') as f:
        for line in tqdm(open(test_path,'r')):
            seq,label = line.strip().split('\t')
            #aa = {'A':0,'T':1,'C':2,'G':3,'N':4,'M':5}
            sequence = torch.Tensor([aa[i] for i in seq]).to(device)
            model = torch.load(ckpt_path)
            model.eval()
            with torch.no_grad():
                sequence = sequence.unsqueeze(0)
                outputs = model(sequence)
                outputs = outputs.tolist()
                f.write(seq + '\t'+ str(label) +'\t'+ str(outputs[0])+ '\n')
                
                
if __name__ == "__main__":
    for i in glob.glob('../one' + '/*'):
        name = i.split('/')[-1]
        print('===================={}==============='.format(name))
        data_path = glob.glob(i + '/1_qidongzi/Data/*')
        all_data_path = [file for file in data_path if 'all_data' in file][0]
        train_path = [file for file in data_path if 'train' in file][0]
        test_path = [file for file in data_path if 'test' in file][0]
        #print(train_path)
        #exit()
    
        ckpt_list = glob.glob('ckpt/' + name + '/*')
        ckpt_path = [file for file in ckpt_list if '400' in file][0]
        wirite_encode(ckpt_path)
        time.sleep(10)
        #print(ckpt_path)
        #exit()
         


























