import glob
from functools import reduce
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# import config as C

def qidongzi(path):    #将完整序列拆分为一个一个的特征利用加号连接起来
    data_0_lst = []
    all_0 = []
    for line in open(path,'r',encoding = 'utf-8'):
        da = line.strip().split('\t')[0]
        data_0_lst.append(da)
    data2fenci = {}
    for da in data_0_lst:
        bi_lst = []
        tri_lst = []
        fo_lst = []
        five_lst = []
        six_lst = []
        for bi in ["".join(da[i:i+2]) for i in range(len(da)-1)]:
            bi_lst.append(bi)
        
        for tri in ["".join(da[i:i+3]) for i in range(len(da)-2)]:
            tri_lst.append(tri)
        for fo in ["".join(da[i:i+4]) for i in range(len(da)-3)]:
            fo_lst.append(fo)
        for five in ["".join(da[i:i+5]) for i in range(len(da)-4)]:
            five_lst.append(five)
        for six in ["".join(da[i:i+6]) for i in range(len(da)-5)]:
            six_lst.append(six)

        fenci = bi_lst + tri_lst + fo_lst + five_lst +six_lst
        all_0.append(fenci)
    return all_0



def load_feat(path):                 #导入box数据，建立一个box为keys,特征为values的字典
    box2fea = {}
    for line in open(path,'r'):
        # print(line)
        # exit()
        box,daibiao,fea = line.strip().split('\t')
        box2fea[box] = fea
    return box2fea

def result(data_qi,box_lst):         #遍历数据，得到一个完整序列为keys，序列中所含的特征为values的字典
    data_ls = []
    data2num_qi = {}
    for one in data_qi:             #遍历数据 (遍历启动子数据还是非启动子数据)
        aa,fea_lst_qi,data_fea_qi = [],[],[]
        fe2 = [item for item in one if len(item) <= 2]
        for a in one[:len(fe2)]:
            last = one[len(fe2)-1:len(fe2)]
            letter = a[0]
            la = last[0][1]
            aa.append(letter)
        aa.extend(la)
        lst = ''.join(aa)
        data_ls.append(lst)
        
        for i in box_lst:          #遍历特征
            if i in one:
                data_fea_qi.append(i)
                fea_lst_qi.append(i)
        data2num_qi[lst] = fea_lst_qi
    return data2num_qi
        
        
        
def write_result(data_dict,write_txt = 'result_1.txt',num = 5): #将结果写入，num代表阈值（一条序列中出现的次数（超过阈值则认为为启动子））
    qq = []
    with open(write_txt,'w')as f:
        select_fea = []
        all_select_fea = []
        for k,v in data_dict.items():
            if len(v) >= num:
                new_data = str(k) + '\t' + str(v) + '\n'
                select_fea.append(v)
                f.write(new_data)
                qq.append(v)
        for ii in select_fea:
            all_select_fea.extend(ii)
        result = dict(Counter(all_select_fea))
        result = dict(sorted(result.items(),key = lambda x:x[1],reverse = True))
    return len(qq)




def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):   #绘制混淆矩阵
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":
#*******************************路径配置*********************************************************
    qi_datapath = 'promoter'                            #启动子数据对应的文件路径
    nonqi_datapath = 'non-promoter'                     #非启动子数据对应的文件路径
#*******************************数据切分*********************************************************
    data_qi = qidongzi(qi_datapath)                          #启动子数据切分之后的列表,一个列表是一条序列[序列1，序列2.....],其中序列1里面是一条序列切分成2，3，，4，5，6的短特征
    data_nonqi = qidongzi(nonqi_datapath)
#*******************************加载box数据*********************************************************
    box_fea = [i.split('\\')[1].split('_')[0]for i in glob.glob('select_pic_50' + '/*')]
                
    box_lst = box_fea
    # print(len(box_lst))
    # exit()
#*******************************设置阈值，写入结果*********************************************************
    n = 4                                                # 阈值 有多少以及上才判定为启动子
    qi_dict = result(data_qi,box_lst)
    cc = write_result(qi_dict,write_txt = '结果/result_11.txt',num = n)
    
    nonqi_dict = result(data_nonqi,box_lst)
    aa = write_result(nonqi_dict,write_txt = '结果/result_01.txt',num = n)
    
    print('启动子数据共有：%s条'%len(qi_dict))
    print('非启动子数据共有：%s条'%len(nonqi_dict))
    
    
    print('预测启动子，真实为启动子的个数：',cc)  # 11 z11
    print('预测启动子，真实为非启动子的个数：',aa)  #01 c01
    
#*******************************计算TP、TN、FP、FN,绘制混淆矩阵*********************************************************
    TP = int(cc)
    FP = int(aa)
    FN = len(qi_dict) - TP
    TN = len(nonqi_dict) - FP
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision_pos = TP/(TP+FP)
    precision_neg = TN/(TN+FN)
    recall_pos = TP/(TP+FN)
    recall_neg = TN/(TN+FP)
    mcc = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    # 定义分类标签
    classes = ['Non-promoter', 'Promoter']  #promoter
    # 生成混淆矩阵
    cm = np.array([[TN, FP], [FN, TP]])
    
    plot_confusion_matrix(cm, classes, title='Confusion Matrix')
    # 将图像保存为文件
    print(f'precision_pos:{precision_pos:.4f}\t recall_pos:{recall_pos:.4f}\t precision_neg:{precision_neg:.4f}\trecall_neg:{recall_neg:.4f}\taccuracy:{accuracy:.4f}\t')
    plt.savefig(f'plt_confusion_matrix/{precision_pos:.4f}_{recall_pos:.4f}_{precision_neg:.4f}_{recall_neg:.4f}_{accuracy:.4f}_{mcc:.4f}.png', dpi=1200)
    
    
    #保存格式：正样本的精确度，正样本的召回率，负样本的精确度，负样本的召回率，整体准确率，mcc值
    
    