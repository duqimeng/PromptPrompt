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
import glob
import os
from tqdm import tqdm
def to_encode(x):
    #print(len(x))
    new_lst =[]
    data_dict = {'A':0,'T':1,'C':2,'G':3,'N':4,'M':5,'Y':6,'W':7}
    for i in x:
        data = [data_dict[j] for j in list(i)]
        new_lst.append(data)
    return new_lst
        


def creat_data(x):
    ret_xx = []
    for j in x:
        a = random.randint(0,80)
        j = list(j)
        try:
            d_l = ['A','T','C','G']
            d_l.remove(j[a])
        except:
            d_l = ['A','T','C','G','N','M','Y','W']
            d_l.remove(j[a])
            
        j[a] = random.choice(d_l)
        jj = "".join(j)
        ret_xx.append(jj)
    x = to_encode(x)
    ret_xx = to_encode(ret_xx)
    return ret_xx
        
        

class embedd(nn.Module):                    
    def __init__(self,embedding_dim):
        super(embedd, self).__init__()      
        # self.layer1 = nn.Linear(2592,1024)
        self.embedding = nn.Embedding(81,embedding_dim, padding_idx=0)
        # self.layer2 = nn.Linear(1024,512)
        # self.relu = nn.ReLU()
        # self.layer3 = nn.Linear(512,num_classes)
        # self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self,x):
        x = torch.tensor(x).to(torch.int64)
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)
        return x



class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define hyperparameters

def load_data(path):
    all_x=[]
    all_y=[]
    train_dict={}
    #print(path)
    #exit()
    for line in open(path,"r",encoding="utf-8"):  #all_data  train_data_1
        x,y = line.split('\t')
        all_x.append(x) 
        all_y.append(y) 
    all_xx = to_encode(all_x)
    #print(all_xx)
    #exit()
    all_streng_xx = creat_data(all_x)
    all_yy = []
    for i in all_y:
        all_yy.append(int(i.strip()))
    all_yy = all_yy
    return all_xx,all_streng_xx,all_y
    
  
  
''' 
test_x=[]
test_y=[]
for line in open("data/test_data_1","r",encoding="utf-8"):
    x,y = line.split('\t')
    test_x.append(x)
    test_y.append([int(y.strip())])
test_y = np.array(test_y)
test_x = np.array(to_encode(test_x))
'''
# test_dataset = MyDataset(test_data, test_label)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize model and move to GPU if available



# Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()

#optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 定义损失函数
# 开始训练
def train_model(model,num_epochs,all_xx_dataloader,all_streng_xx_dataloader,learning_rate,batch_size,ckpt_path):
    sw1 = SummaryWriter(log_dir='log1/name' +'log1')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        print(epoch,'---------')
        running_loss = 0.0
        all_data_train = []
        for (inputs, labels),(inputs_str, labels) in zip(all_xx_dataloader,all_streng_xx_dataloader):
            optimizer.zero_grad()
            # 前向传递和计算损失
            inputs = inputs.float().to(device)
            inputs_str = inputs_str.float().to(device)
            outputs = model(inputs)
            
            outputs_str = model(inputs_str)
            loss = contrastive_loss(outputs, outputs_str)
            # 反向传递和更新模型参数
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            #running_loss = loss.item()
            # 每 1000 次迭代输出一次训练状态 
            # if (+1) %len(all_xx_dataloader)  == 0:
            
            # print('Epoch [%d/%d], Loss: %.4f'% (epoch+1, num_epochs, running_loss / 1000))
        print('Epoch [%d/%d], Loss: %.4f'% (epoch+1, num_epochs, running_loss))
        sw1.add_scalar("train_loss", running_loss, epoch)
        running_loss = 0.0
                
        # with open('bianma_train.txt','w',encoding = 'utf-8')as f:
            # for x,y in zip(raw_data_lst_train,all_data_train):
                # x = num2na(x)
                # result = x+ '\t' + y + '\n'
                # f.write(result)
        if (epoch+1) %50 == 0:
            torch.save(model, "./{}/model_{}.pth".format(ckpt_path,epoch+1))
    print("模型已保存")
    wirite_encode(ckpt_path)
    
    

    # 评估模型
        # model.eval()
        # with torch.no_grad():
            # correct = 0
            # total = 0
            # for inputs, labels in train_dataloader:
                # outputs = model(inputs)
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
            # print('Accuracy of the model on the %d train samples: %d %% ++++++++++' % (len(train_dataset), 100 * correct / total))

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
    ckpt = ckpt_path + '/model_400.pth'
    aa = {'A':0,'T':1,'C':2,'G':3,'N':4,'M':5,'Y':6,'W':7}
    with open(encode_train,'w',encoding = 'utf-8') as f:
        #ckpt_path = glob.glob('ckpt' + '/*')
        for line in tqdm(open(train_path,'r')):
            seq,label = line.strip().split('\t')
            
            sequence = torch.Tensor([aa[i] for i in seq]).to(device)
            import glob

            model = torch.load(ckpt)
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
            model = torch.load(ckpt)
            model.eval()
            with torch.no_grad():
                sequence = sequence.unsqueeze(0)
                outputs = model(sequence)
                outputs = outputs.tolist()
                f.write(seq + '\t'+ str(label) +'\t'+ str(outputs[0])+ '\n')


if __name__ == '__main__':
    #data_path = "data/all_data.txt"
    for i in glob.glob('../one' + '/*'):
        name = i.split('/')[-1]
        if name == '26695':
        
            print('===================={}==============='.format(name))
            data_path = glob.glob(i + '/1_qidongzi/Data/*')
            all_data_path = [file for file in data_path if 'all_data' in file][0]
            train_path = [file for file in data_path if 'train' in file][0]
            test_path = [file for file in data_path if 'test' in file][0]
            #print(train_path)
            #exit()
    
            ckpt_path = 'ckpt/' + name 
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            num_epochs = 500
            learning_rate = 0.0001
            batch_size =128
            
            #ckpt_path = 'ckpt/'
            all_xx,all_streng_xx,all_yy = load_data(all_data_path)
            
            all_xx = torch.Tensor(all_xx)
            all_streng_xx = torch.Tensor(all_streng_xx)
            all_xx_dataset = MyDataset(all_xx, all_yy)
            all_xx_dataloader = DataLoader(all_xx_dataset, batch_size=batch_size, shuffle=False)
            
            all_streng_xx_dataset = MyDataset(all_streng_xx, all_yy)
            all_streng_xx_dataloader = DataLoader(all_streng_xx_dataset, batch_size=batch_size, shuffle=False)
            model = embedd(32).to(device)
            if sys.argv[1]=="train": 
                train_model(model,num_epochs,all_xx_dataloader,all_streng_xx_dataloader,learning_rate,batch_size,ckpt_path)
        
        
    #if sys.argv[1]=="test":
    
    






