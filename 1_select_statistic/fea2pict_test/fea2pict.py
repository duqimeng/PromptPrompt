import glob
from collections import Counter
from tqdm import tqdm
import re
import numpy as np
import config as C


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def collect_data():                  #读取筛选出的的特征，建立一个列表
    data_lst = glob.glob('../1_select_statistic/result_1'+ '/*')
    feature_lst = []
    for one in data_lst:
        if one.find('tf')!= -1:
            for line in open(one,'r'):
                feature = line.strip().split('\t')[0]
                feature_lst.append(feature)
        if one.find('other')!= -1:
            for line in open(one,'r'):
                feature_ = line.strip().split('\t')[0]
                feature_lst.append(feature_)
    return feature_lst
    

def load_qidata(path,pic_topath,sty = 'nonqi'):
    feature_list = collect_data()
    feature_lst = [''.join(i.split('+'))for i in feature_list]
    new_lst = get_max_len_string(feature_lst)
    for feature in tqdm(new_lst[0]):
        a = []
        fea2index = {}
        for line in open(path,'r'):
            data,fea = line.strip().split('\t')
            # print(data)
            try:
                addr = [substr.start() for substr in re.finditer(feature,data)]
                index = data.index(feature)
                a.extend(addr)
                b.append(index)
            except:
                pass
        print('特征：',feature,'在启动子序列中共出现：',len(a),'次')
        result = dict(Counter(a))
        paixu_data = dict(sorted(result.items(),key=lambda x:x[0])) 
        x_axis = list(paixu_data.keys())
        y_axis = list(paixu_data.values())
        plt.clf()
        
        fig, ax = plt.subplots()
        ax.set_xlim(xmin = 0, xmax = 81)
        plt.bar(x_axis,y_axis,width=0.8, bottom=0,color = 'blue')
        
        plt.xlabel('特征所处位置')
        plt.ylabel('不同位置对应的个数')
        plt.savefig(pic_topath + str(feature)+'_'+ str(len(a)) +'_'+str(sty)+".png",dpi=600)
    return fea2index 



def get_max_len_string(input_list):
    input_list.sort(key=lambda x: len(x), reverse=True)     #降序排序，因为最长的字符串是不可能是其他字符串的子串
    out, filter_list = [], []
    for s in input_list:
        mask_list = [s in o for o in out]
        if not any(mask_list):                              #any函数全部为false才返回false
            out.append(s)
        else:
            # 记录是因为那个元素的存在导致被过滤
            been_contained_index = mask_list.index(True)
            large_word = out[been_contained_index]
            filter_list.append(s)
    return out, filter_list


if __name__ == "__main__":
    qi_path = C.Promoter_path
    nonqi_path = C.non_Promoter_path
    
    
    pic_tononqipath = C.fea_0
    pic_toqipath = C.fea_1
    load_qidata(qi_path,pic_toqipath,sty = 'qi')
    # load_qidata(nonqi_path,pic_tononqipath,sty = 'nonqi')