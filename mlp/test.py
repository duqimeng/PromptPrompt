import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import sys



    
if __name__ == "__main__":
    # make_data_mian()
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