import glob
from collections import Counter
import os


def collect_data(path1):                   #读取result_1文件夹中的数据，所有特征全部存放在feature_lst列表中
    feature_lst = []
    for one in glob.glob(path1+ '/*'):
        if one.find('tf')!= -1:
            for line in open(one,'r'):
                feature = line.strip().split('\t')[0]
                feature_lst.append(feature)
    return feature_lst
        
        
    
def find(fea2nu):                          #将字典中相同values存放在一个列表中，组成新的字典，并将结果写进box_result.txt
    _dict = {}
    for key, value in fea2nu.items():
        if not value:
            continue
        if value not in _dict.keys():
            _dict[value] = []
        _dict[value].append(key)
    with open('box_result.txt','w')as f:
        for key, value in _dict.items():
            if len(value) >= 1:
                _str = ",".join([str(x) for x in value])
                new_data = key +'\t'+value[-1]+ '\t' + _str
                f.write(new_data + '\n')       
                
                
def word_sort_m1(words):                   #对字母列表进行排序
    # 先转换成列表
    tmp = [i for i in words if i != ' ']
    res = sorted(tmp)
    return res
    
def Clean_empty(path):                     #清除result文件夹下的空白文件
    for (dirpath,dirnames,filenames) in os.walk(path):
        for filename in filenames:
            file_folder=dirpath+'/'+filename
            if os.path.isdir(file_folder): 
                if not os.Listdir(file_folder): 
                    print(file_folder)
            elif os.path.isfile(file_folder): 
                if os.path.getsize(file_folder) == 0: 
                    print(file_folder)
                    os.remove(file_folder)  
    print(path, 'clean over!')    
    
    

if __name__ == "__main__":
    path1 = 'result_1'
    Clean_empty(path1)   #清除result文件夹下的空白文件
    feature_lst = collect_data(path1)
    path_lst = glob.glob(path1+'/*')
    only_qi = []
    for i in path_lst:
        if i.find('other')!= -1:
            for lin in open(i):
                only_qi.append(lin.strip())

    feature_lst = feature_lst + only_qi
    print('总共有特征：%s个'%(len(feature_lst)))
    da_l = []
    fea2nu = {}
    for i in feature_lst:
        data = ''.join(i.split('+'))
        dan = ''.join(word_sort_m1("".join(set(data))))
        fea2nu[data] = dan
        da_l.append(dan)
    a = Counter(da_l)
    find(fea2nu)
        
    #此代码主要是读取resullt_1文件夹下的数据，生成box_result.txt文件

    