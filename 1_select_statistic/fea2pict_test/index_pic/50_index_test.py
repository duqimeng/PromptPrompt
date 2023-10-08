import glob 
from functools import reduce
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable



def collect_name(path_50):
    data_lst = glob.glob(path_50+'/*')
    sta_lst = []
    for i in data_lst:
        sta = i.split('\\')[-1].split('_')[0]
        sta_lst.append(sta)
    return sta_lst




def qidongzi(path):    #将完整序列拆分为一个一个的特征利用加号连接起来
    data_0_lst = []
    all_0 = []
    for line in open(path,'r',encoding = 'utf-8'):
        da = line.strip().split('\t')[0]
        data_0_lst.append(da)
    data2fenci = {}
    data2fenci_test = {}
    
    for da in data_0_lst:
        bi_lst = []
        tri_lst = []
        fo_lst = []
        five_lst = []
        six_lst = []
        six_lst_ = []
        da_ = da[45:55]
        
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
            
        for six in ["".join(da_[i:i+6]) for i in range(len(da_)-5)]:
            six_lst_.append(six)

        fenci = six_lst_
        
        fenci_1 = bi_lst + tri_lst + fo_lst + five_lst +six_lst
        all_0.append(fenci)
        data2fenci[da] = fenci
        data2fenci_test[da] = fenci_1
    return data2fenci, data2fenci_test


def collect_data():                  #读取筛选出的的特征，建立一个列表
    data_lst = glob.glob('result_1'+ '/*')
    feature_lst = []
    for one in data_lst:
        if one.find('tf')!= -1:
            for line in open(one,'r'):
                feature = line.strip().split('\t')[0]
                feature_lst.append(feature)
    return feature_lst

              
def make_data(data_lst):              #制造特征数据，目前没有用到
    all_ = []
    for i in range(2,7):
        result = reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [data_lst] * i)
        all_.extend(result)
    print('共制造出TATTA衍生出来的特征数共：',len(all_))
    return all_
        

def load_feat(path):                 #导入box数据，建立一个box为keys,特征为values的字典
    box2fea = {}
    for line in open(path,'r'):
        box,fea = line.strip().split('\t')
        box2fea[box] = fea
    return box2fea

def result(data_qi,box_lst):         #遍历数据，得到一个完整序列为keys，序列中所含的特征为values的字典
    data_qi_fea = data_qi.values()
    data_qi = data_qi.keys()
    bb,data_ls, = [],[]
    data2num_qi = {}
    data2num_nonqi = {}
    for x,one in zip(data_qi,data_qi_fea):             #遍历数据 (遍历启动子数据还是非启动子数据)
        aa,fea_lst_qi,data_fea_qi = [],[],[]
        for i in box_lst:          #遍历特征
            if i in one:
                fea_lst_qi.append(i)
        data2num_qi[x] = fea_lst_qi

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




if __name__ == "__main__":
    path_50 =  'select_pic_50'
    sta_lst = collect_name(path_50)
    qi_datapath = 'B100.txt'                        #启动子数据对应的文件路径
    nonqi_datapath = 'non_qiB100.txt'                     #非启动子数据对应的文件路径
    box_lst = sta_lst
    
    data_qi,data_qi_test = qidongzi(qi_datapath)
    data_nonqi,data_nonqi_test = qidongzi(nonqi_datapath)
    # print(data_qi_test)
    # exit()
    n = 1  # 阈值
    qi_dict = result(data_qi,box_lst)

    cc = write_result(qi_dict,write_txt = '结果/result_11.txt',num = n)

    nonqi_dict = result(data_nonqi,box_lst)
    aa = write_result(nonqi_dict,write_txt = '结果/result_01.txt',num = n)
    print('预测启动子，真实为启动子的个数：',cc)
    print('预测启动子，真实为非启动子的个数：',aa)
  
    z00 = len(nonqi_dict) - aa
    c10 = len(qi_dict) - cc
    
    plt_pic(float(aa),cc,z00,c10)
    
    
    # zq0 = 3382 -  aa # 
    # cu1 = 3382 -  cc #
    # plt_hunxiao(zq0,aa,cu1,cc,n)
    