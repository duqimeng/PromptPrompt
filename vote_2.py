#-*- coding: UTF-8 -*- 
import glob
import os
from collections import Counter


def read_data(path):
    data2label = {}
    for line in open(path,'r',encoding = 'utf-8'):
        data,label = line.strip().split('\t')
        data2label[data] = label
        # all_data.append(data)
    return data2label

def flip_label(label):
    if label == '0':
        return '1'
    elif label == '1':
        return '0'
    else:
        return label
        
        
def read_data_cubt(path):
    data2label = {}
    for line in open(path,'r',encoding = 'utf-8'):
        data,label,label_ = line.strip().split('\t')
        #print(label)
        #label = flip_label(label)
        #print(label)
        data2label[data] = label
        # all_data.append(data)
    return data2label

def read_data_cubt_flip(path):
    data2label = {}
    for line in open(path,'r',encoding = 'utf-8'):
        data,label,label_ = line.strip().split('\t')
        #print(label)
        label = flip_label(label)
        #print(label)
        data2label[data] = label
        # all_data.append(data)
    return data2label
    
    
    
def hebingzidian(dic_a,dic_b):
    result_dic = {}
    for k,v in dic_a.items():
        for m,n in dic_b.items():
            if k == m:
                result_dic[k] = []
                result_dic[k].append(dic_a[k])
                result_dic[k].append(dic_b[k])
                dic_a[k] = result_dic[k]
                dic_b[k] = result_dic[k]
            else:
                result_dic[k] = dic_a[k]
                result_dic[m] = dic_b[m]
    return result_dic
 

if __name__== '__main__':
    file_path = 'vote_result/result.txt'
    if os.path.exists(file_path):
        os.remove(file_path)
    file_path_ = 'vote_result/result_2.txt'
    if os.path.exists(file_path_):
        os.remove(file_path_)
    
        file_path_1 = 'vote_result/test.txt'
    if os.path.exists(file_path_1):
        os.remove(file_path_1)
    
    #flip_label_lst = ['mg1655','b100','j2315','usda110','26695','rm1221','sbr5','168','13032']
    result_ = '物种\t全部预测一致：\t  其中启动子：\t非启动子： \t 预测对\t  其中启动子：\t非启动子： \t   预测错\t  其中启动子：\t非启动子： \t 准确率\t  测试集总数\t 启动子个数：\t 非启动子个数：\t没进行预测的个数'
    with open(file_path, "a") as file: file.write(result_ + '\n')
    
    
    result__ = '物种\t全部预测一致：\t  其中启动子：\t非启动子： \t 预测对\t  其中启动子：\t非启动子： \t   预测错\t  其中启动子：\t非启动子： \t 准确率\t  测试集总数\t 启动子个数：\t 非启动子个数：\t没进行预测的个数'
    with open(file_path_, "a") as file: file.write(result__ + '\n')  
    
    rr = '物种\t所有\t启动子个数\t非启动子个数\t预测对的\t预测对启动子的\t预测对非启动子的\t预测错的\t预错启动子\t预测错非启动子\t整体准确率'
    with open(file_path_1, "a") as file: file.write(rr + '\n') 
    for i in glob.glob('one' + '/*'):
        name = i.split('/')[-1]
        
        print('===================={}==============='.format(name))
        
        raw_path = glob.glob(i + '/*')[0] + '/Data/test_data_1.txt'
        duibi_path= [file for file in glob.glob('duibi/duibi_mlp/predict/' + '/*') if name in file][0]
        
        mlp_path = [file for file in glob.glob('mpl/predict/' + '/*') if name in file][0]
        cubt_path = glob.glob(i + '/*')[0] + '/6CUBT_LR_model/LR_model/cubt_predicts_result'
        
        
        textcnn_path= [file for file in glob.glob('textcnn/textcnn/predict/' + '/*') if name in file][0]
        #print(textcnn_path)
        #exit()
        all_data = []
        #duibi2label = read_data(duibi_path)
        #mlp2label = read_data(mlp_path)
        #
        #cubt2label = read_data_cubt(cubt_path)
        #print(textcnn_path)
        #if name in flip_label_lst:
        #    cubt2label = read_data_cubt_flip(cubt_path)
        #if name not in flip_label_lst:
        #   
        cubt2label = read_data_cubt(cubt_path)
        textcnn2label = read_data(textcnn_path)
        duibi2label = read_data(duibi_path)
        #cubt2label = read_data_cubt(cubt_path)
        
        mlp2label = read_data(mlp_path)
        
        
         
        test2label = read_data(raw_path)
        #raw_list = list(test2label.values())
        #from collections import Counter
        #counts = Counter(raw_list)
        #print(counts)
        #xit()
        #exit()
        '''
        print(len(test2label))
        print(len(textcnn2label))
        exit()
        new_dict = {}
        for key in test2label.keys():
            new_dict[key] = [cubt2label[key], duibi2label[key], mlp2label[key], test2label[key]]
        
        '''
        
        try:
            new_dict = {}
            for key in test2label.keys():
                new_dict[key] = [cubt2label[key], duibi2label[key], mlp2label[key], test2label[key]]
        except:
            print(len(cubt2label))
            print(len(mlp2label))
            print(len(duibi2label))
            #print(len(textcnn2label))
            
        
        dd = 0
        cc = 0
        for k,v in new_dict.items():
            v = [int(i) for i in v]
            cc+=1
            if v[0] == v[-1]:
                dd+=1
        #print(dd)
        #print(cc)
        #exit()
        #print(test2label)
        count_dict = {value: len([key for key, val in test2label.items() if val == value]) for value in set(test2label.values())}
        #print(count_dict)
        #exit()
        all_0 = count_dict['0']
        all_1 = count_dict['1']
        print('非启动子数量：',all_0)
        print('启动子数量：',all_1)
        #exit()
        '''
        print(new_dict)
        exit()
        try:
            cubt_duibi = hebingzidian(cubt2label,duibi2label)
            mlp_test = hebingzidian(mlp2label,test2label)
            new_dict = {key: cubt_duibi[key] + mlp_test[key] for key in cubt_duibi}
        except:
            print(len(cubt2label))
            print(len(mlp2label))
            print(len(duibi2label))
        '''   
            
        zheng_000 = 0
        zheng_111 = 0
        zheng_001 = 0
        zheng_010 = 0
        zheng_011 = 0
        zheng_101 = 0
        zheng_110 = 0
        zheng_100 = 0
        
        cuo_000 = 0
        cuo_111 = 0
        cuo_001 = 0
        cuo_010 = 0
        cuo_011 = 0
        cuo_101 = 0
        cuo_110 = 0
        cuo_100 = 0
        aa = 0
        bb = 0
        fu_dui,fu_cuo,zheng_dui,zheng_cuo =0,0,0,0
        for k,v in new_dict.items():
            v = [int(i) for i in v]
            if sum(v[:3]) == 0 and v[-1] == 0:
                zheng_000+=1
            if sum(v[:3]) == 0 and v[-1] == 1:
                cuo_000+=1
        
            if sum(v[:3]) == 3 and v[-1] == 1:
                zheng_111+=1
            if sum(v[:3]) == 3 and v[-1] == 0:
                cuo_111+=1
    
    
            if sum(v[:2]) == 0 and v[2] ==1 and v[-1] == 1:
                zheng_001+=1
            if sum(v[:2]) == 0 and v[2] ==1 and v[-1] == 0:
                cuo_001+=1
        
            if v[0] == 0 and v[1] == 1 and v[2] == 0 and v[-1] == 1:
                zheng_010+=1
            if v[0] == 0 and v[1] == 1 and v[2] == 0 and v[-1] == 0:
                cuo_010+=1
        
            if v[0] == 0 and v[1] == 1 and v[2] == 1 and v[-1] == 1:
                zheng_011+=1
            if v[0] == 0 and v[1] == 1 and v[2] == 1 and v[-1] == 0:
                cuo_011+=1
        
            if v[0] == 1 and v[1] == 0 and v[2] == 1 and v[-1] == 1:
                zheng_101+=1
            if v[0] == 1 and v[1] == 0 and v[2] == 1 and v[-1] == 0:
                cuo_101+=1
        
            if v[0] == 1 and v[1] == 1 and v[2] == 0 and v[-1] == 1:
                zheng_110+=1
            if v[0] == 1 and v[1] == 1 and v[2] == 0 and v[-1] == 0:
                cuo_110+=1
                
            if v[0] == 1 and v[1] == 0 and v[2] == 0 and v[-1] == 1:
                zheng_100+=1
                
            if v[0] == 1 and v[1] == 0 and v[2] == 0 and v[-1] == 0:
                cuo_100+=1
            
            
            if sum(v[:3]) != 0 and sum(v[:3]) != 3:
                if sum(v[:3]) == 1 and v[-1] ==0:
                    fu_dui+=1
                if sum(v[:3]) == 1 and v[-1] ==1:
                    fu_cuo+=1
                    
                if sum(v[:3]) == 2 and v[-1] ==1:
                    zheng_dui+=1
                if sum(v[:3]) == 2 and v[-1] ==0:
                    zheng_cuo+=1
                    
                    
                aa+=1
                if v[2] == v[-1]:
                    #print(v[:])
                    bb+=1
                
                #print(v[:3])
        #print(aa)
        #print(bb)
        
        print('***********************************************')
        suoyou = fu_dui+fu_cuo+zheng_dui+zheng_cuo
        
        
        print('没进行预测个数',suoyou )
        qi = fu_cuo +zheng_dui
        
        feiqi = fu_dui + zheng_cuo
        
        '物种\t所有\t预测对的\t预测对启动子的\t预测对非启动子的\t预测错的\t预错启动子\t预测错非启动子\t整体准确率'
        cuuu = fu_cuo + zheng_cuo
        
        lvv = round(((fu_dui + zheng_dui)/suoyou),4)
        print('{}\t{}\t{}'.format(name,suoyou,lvv))
        result_11 = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{},\t{}'.format(name,suoyou,qi,feiqi,(fu_dui + zheng_dui),zheng_dui,fu_dui,cuuu,zheng_cuo,fu_cuo,lvv)
        
        
        
        print(aa)
        print('---------------------------------------------')
        
        lv = round((bb/aa),4)
        #print('{}\t{}\t{}'.format(name,aa,lv))
        
        
        #print(bb/aa)
        #exit()
        result_1 = '{}\t{}\t{}'.format(name,aa,lv)
        
        
        with open(file_path_1, "a") as file: file.write(result_11 + '\n')
            
        all = zheng_000 + zheng_111 + cuo_000 + cuo_111
        
        all_yuce_0 = zheng_000 + cuo_000
        all_yuce_1 = zheng_111 + cuo_111
        
        yuce_0 = zheng_000
        yuce_1 = zheng_111
        #print(yuce_0,yuce_1,all)
        #exit()
        yuce_0_ = cuo_000
        yuce_1_ = cuo_111
        
        recall_0 = round((zheng_000/all_0),4)
        recall_1 = round((zheng_111 /all_1),4)
        
        pre_0 = round(zheng_000/(zheng_000 + cuo_000),4)
        pre_1 = round(zheng_111 /(zheng_111+cuo_111),4)
        
        
        #print('')
        #print(recall_0,recall_1,pre_0,pre_1)
        #exit()
        all_dui = zheng_000 + zheng_111
        all_cuo = cuo_000+cuo_111
        acc = round((all_dui/all),4)
        rest = len(test2label) - all
        
        result = '{}\t{}\t  {}\t{} \t {}\t  {}\t{} \t {}\t  {}\t{} \t{}\t{}\t{}\t{}\t{}'.format(name,all,all_yuce_1,all_yuce_0,all_dui,yuce_1,yuce_0,all_cuo,yuce_1_,yuce_0_,acc,len(test2label),all_1,all_0,rest)
        with open(file_path, "a") as file: file.write(result + '\n')
        
        
        cubt_duibi_0 = 0
        cubt_duibi_1 = 0
        cubt_duibi_cuo_0 = 0
        cubt_duibi_cuo_1 = 0
        for k,v in new_dict.items():
            v = [int(i) for i in v]
            
            if v[1] == 1 and v[2] == 1  and v[-1] == 1:
                cubt_duibi_1+=1
                
            if v[1] == 0 and v[2] == 0  and v[-1] == 0:
                cubt_duibi_0+=1
                
            if v[1] == 1 and v[2] == 1  and v[-1] == 0:
                cubt_duibi_cuo_0+=1
            if v[1] == 0 and v[2] == 0  and v[-1] == 1:
                cubt_duibi_cuo_1+=1
                
                
        all_ = cubt_duibi_0 + cubt_duibi_1 + cubt_duibi_cuo_0 + cubt_duibi_cuo_1
        all_dui_ = cubt_duibi_0 + cubt_duibi_1
        all_cuo_ = cubt_duibi_cuo_0 + cubt_duibi_cuo_1
        
        yuce_00 = cubt_duibi_0 + cubt_duibi_cuo_0
        yuce_11 = cubt_duibi_1 + cubt_duibi_cuo_1
        
        acc_ = round((all_dui_ / all_),4)
        
        rest_ = len(test2label) - all_
        #print(rest_)
        #exit()
        #'物种\t全部预测一致：\t  其中启动子：\t非启动子： \t 预测对\t  其中启动子：\t非启动子： \t   预测错\t  其中启动子：\t非启动子： \t 准确率\t  测试集总数\t 启动子个数：\t 非启动子个数：\t没进行预测的个数'
        
        result_ = '{}\t{}\t  {}\t{} \t {}\t  {}\t{} \t {}\t  {}\t{} \t{}\t{}\t{}\t{}\t{}'.format(name,all_,yuce_11,yuce_00,all_dui_,cubt_duibi_1,cubt_duibi_0,all_cuo_,cubt_duibi_cuo_1,cubt_duibi_cuo_0,acc_,len(test2label),all_1,all_0,rest_)
        
        with open(file_path_, "a") as file_: file_.write(result_ + '\n')

        all_ = 0
        all = 0
        all_dui = 0
        all_cuo = 0
        acc = 0
        rest = 0
        acc_ = 0
        rest_ = 0
        cubt_duibi = 0
        cubt_duibi_cuo = 0

        
        #zheng_111
        '''
        print('000===' ,zheng_000)
        print('000===',cuo_000)
        print('===============================')
        print(zheng_111)
        print(cuo_111)
        print('=======')
        print(zheng_001)
        print(cuo_001)
        print('=======')
        print(zheng_010)
        print(cuo_010)
        print('=======')
        print(zheng_011)
        print(cuo_011)
        print('=======')
        print(zheng_101)
        print(cuo_101)
        print('=======')
        print(zheng_110)
        print(cuo_110)
        print('=======')
        print(zheng_100)
        print(cuo_100)
        print('-------------')
        '''

