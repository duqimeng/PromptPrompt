# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import sys
import glob
import os
import time

class MLP_(nn.Module):                    
    def __init__(self, num_classes):
        super(MLP_, self).__init__()      
        self.layer1 = nn.Linear(81,16)
        self.Tanh = nn.Tanh()
        # self.layer2 = nn.Linear(1024,512)
        # self.relu = nn.ReLU()
        self.layer3 = nn.Linear(16,num_classes)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self,x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.Tanh(x)
        # x = self.layer2(x)
        # x = self.Tanh(x)
        x = self.layer3(x)
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





def load_train_data(path):
    train_data = []
    train_label =[]
    #train2bianma = {}
    raw_data_lst_train= []
    data_dict = {'A':0,'T':1,'C':2,'G':3,'N':4,'M':5,'Y':6,'W':7}
    with open(path, 'r') as f:
        for line in f:
            raw_data,label = line.strip().split('\t')
            data = [data_dict[i] for i in raw_data]
            train_label.append(int(label))
            train_data.append(data)
            raw_data_lst_train.append(raw_data)
    return train_data,train_label,raw_data_lst_train





def load_test_data(path):
    test_data = []
    test_label =[]
    #train2bianma = {}
    raw_data_lst_test= []
    data_dict = {'A':0,'T':1,'C':2,'G':3,'N':4,'M':5,'Y':6,'W':7}
    with open(path, 'r') as f:
        for line in f:
            raw_data,label = line.strip().split('\t')
            data = [data_dict[i] for i in raw_data]
            test_label.append(int(label))
            test_data.append(data)
            raw_data_lst_test.append(raw_data)
    return test_data,test_label,raw_data_lst_test



def dataset(train_data,train_label,test_data,test_label,batch_size):  
    
    train_data = torch.Tensor(train_data)
    train_label = torch.Tensor(train_label)
    test_data = torch.Tensor(test_data)
    test_label = torch.Tensor(test_label)
    
    # Move data and labels to GPU if available
    
    train_data = train_data.to(device)
    train_label = train_label.to(device)
    
    test_data = test_data.to(device)
    test_label = test_label.to(device)
    # Create dataloader
    train_dataset = MyDataset(train_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    
    test_dataset = MyDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader,test_dataloader
    
#print(len(test_dataloader))
#exit()
# Initialize model and move to GPU if available
# Define loss function and optimizer
#optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 定义损失函数
# 开始训练
def train_model(model,num_epochs,train_dataloader,test_dataloader,raw_data_lst_test,learning_rate,batch_size,ckpt_path,train_data,test_data):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_data_train = []
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # 前向传递和计算损失
            #print(inputs)
            #exit()
            outputs = model(inputs.float())

            loss = criterion(outputs, labels.long())
            # 反向传递和更新模型参数
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 每 1000 次迭代输出一次训练状态 
            if (i+1) %len(train_dataloader)  == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch+1, num_epochs, i+1, len(train_data)//batch_size, running_loss / len(train_dataloader)))
                running_loss = 0.0
        if (epoch+1) %50 == 0:
            torch.save(model, "./{}model_{}.pth".format(ckpt_path,epoch+1))

    # 评估模型
        model.eval()
        with torch.no_grad():
            correct_ = 0                                                                                                       
            total_ = 0                                                                                                         
            all_data_train = []                                                                                                
            all_predict_train = []                                                                                                  
            for inputs, labels in train_dataloader:                                                                            
                outputs = model(inputs)                                                                                       
                _, predicted = torch.max(outputs.data, 1)                                                                    
                #print(len(predicted))                                                                                        
                #exit()
                                                                                              
                all_predict_train.extend(predicted.tolist())                                                                        
                total_ += labels.size(0)                                                                                       
                                                                                                                              
                correct_ += (predicted == labels).sum().item()
                         
                         
                         
                         
                         
                         
                         
                         
                         
                                                                       
            ree = 'Accuracy of the model on the %d train samples: %d %%-------------' % (len(train_data), 100 * correct_ / total_)
            print(ree)
        with torch.no_grad():
            correct = 0
            total = 0
            correct_ = 0
            total_ = 0
            
            all_data_test = []
            all_predict = []
            for inputs, labels in test_dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                #print(len(predicted))
                #exit()
                all_predict.extend(predicted.tolist())
                total += labels.size(0)
                
                correct += (predicted == labels).sum().item()
                
                
                
                
     
            for inputs, labels in train_dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                #print(len(predicted))
                #exit()
                #all_predict_train.extend(predicted.tolist())
                total_ += labels.size(0)
                correct_ += (predicted == labels).sum().item()
                
                 
            ree_test = 'Accuracy of the model on the %d test samples: %d %%-------------' % (len(test_data), 100 * correct / total)
            ree_train = 'Accuracy of the model on the %d train samples: %d %%-------------' % (len(train_data), 100 * correct_ / total_)
            #print(ree)  
            if epoch == num_epochs - 1:
                with open('result/' + 'result.txt','a',encoding = 'utf-8')as fff:
                    fff.write(name + '\t' + ree_train +'\t'+ree_test+'\n')
                
                
                
                
                
                
                
                
            ree = 'Accuracy of the model on the %d test samples: %d %%-------------' % (len(test_data), 100 * correct / total)
            print(ree)  
            if epoch == num_epochs - 1:
                with open('result/' + 'result.txt','a',encoding = 'utf-8')as fff:
                    fff.write(name + '\t' + ree +'\n')
                
            #print(len(all_predict))
            #exit()
            
            with open('predict/' + name + '_mpl_predicted.txt','w',encoding = 'utf-8')as ff:
               if epoch == num_epochs - 1:
                   for k,v in zip(raw_data_lst_test,all_predict):
                       #print(len(k))
                       #exit()
                       # k = num2na_(k)
                       ff.write(str(k)+ '\t' + str(v) + '\n')

if __name__ == '__main__':
#if sys.argv[1]=="train":
    for i in glob.glob('../one' + '/*'):
        name = i.split('/')[-1]
        print('===================={}==============='.format(name))
        data_path = glob.glob(i + '/1_qidongzi/Data/*')
        
        #print(data_path)
        #exit()
        #ckpt_path = 'ckpt/'
        
        ckpt_path = 'ckpt/' + name +'/' 
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            
        #print(ckpt_path)
        #exit()
        
        #all_data_path = [file for file in data_path if 'all_data' in file][0]
        train_path = [file for file in data_path if 'train' in file][0]
        test_path = [file for file in data_path if 'test' in file][0]
        #train_path = '../bianma_train.txt'
        #test_path = '../bianma_test.txt'
        #print(train_path)
        #print(test_path)
        #exit()
        #print(train_path)
        #print(test_path)
        #exit()
        learning_rate = 0.0001
        batch_size =128
        
        num_epochs = 500 
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        train_data,train_label,raw_data_lst_train = load_train_data(train_path)
        test_data,test_label,raw_data_lst_test = load_test_data(test_path)
        #print(len(test_data))
        #exit()
        train_dataloader,test_dataloader = dataset(train_data,train_label,test_data,test_label,batch_size)
        
        
        

        model = MLP_(2).to(device)
        train_model(model,num_epochs,train_dataloader,test_dataloader,raw_data_lst_test,learning_rate,batch_size,ckpt_path,train_data,test_data)
        time.sleep(20)

                           
    
'''  
if sys.argv[1]=="test":
    seq = input('请输入要判断的序列：')
    print(seq)
    # seq = 'ACACTATTATTGCACTAATTCGCCCTTTGCAATCTATCAATGAGTAGTATAAATACGCTCAGTTACCTTCATTCAATCTAT'
    aa = {'A':0,'T':1,'C':2,'G':3}
    sequence = torch.Tensor([aa[i] for i in seq]).to(device)
    import glob
    ckpt_path = glob.glob('ckpt' + '/*')
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
        
    
    model = torch.load(strsort(ckpt_path)[0])
    sequence = sequence.unsqueeze(0)
    outputs = model(sequence)
    _,predicted = torch.max(outputs.data, 1)
    if predicted.tolist()[0] == 0:
        print('此序列为非启动子')
    if predicted.tolist()[0] == 1:
        print('此序列为启动子')

'''


