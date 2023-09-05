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


def read_data_cubt(path):
    data2label = {}
    for line in open(path,'r',encoding = 'utf-8'):
        data,label,label_ = line.strip().split('\t')
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
    file_path = 'result.txt'
    if os.path.exists(file_path):
        os.remove(file_path)
    file_path_ = 'result_2.txt'
    if os.path.exists(file_path_):
        os.remove(file_path_)
        
        
    file_path__ = 'hight.txt'
    if os.path.exists(file_path__):
        os.remove(file_path__)  
    file_path___ = 'low.txt'
    if os.path.exists(file_path___):
        os.remove(file_path___)  
        
        
    for i in glob.glob('one' + '/*'):
        name = i.split('/')[-1]
        print('===================={}==============='.format(name))
        
        raw_path = glob.glob(i + '/*')[0] + '/Data/test_data_1.txt'
        duibi_path= [file for file in glob.glob('duibi/duibi_mlp/predict/' + '/*') if name in file][0]
        mlp_path = [file for file in glob.glob('mpl/predict/' + '/*') if name in file][0]
        cubt_path = glob.glob(i + '/*')[0] + '/6CUBT_LR_model/LR_model/cubt_predicts_result'

        all_data = []
        
        #print(mlp_path)
        #exit()
        #duibi_path = [file for file in path_lst if 'duibi' in file][0]
        #cubt_path = [file for file in path_lst if 'cubt' in file][0]
        #mlp_path = [file for file in path_lst if 'mlp' in file][0]
        #raw_path = [file for file in path_lst if 'test' in file][0]
    
        
        cubt2label = read_data(duibi_path)
        mlp2label = read_data(mlp_path)
        duibi2label = read_data_cubt(cubt_path)
        
        
        test2label = read_data(raw_path)

        try:
            cubt_duibi = hebingzidian(cubt2label,duibi2label)
            mlp_test = hebingzidian(mlp2label,test2label)
            new_dict = {key: cubt_duibi[key] + mlp_test[key] for key in cubt_duibi}
        except:
            print(len(cubt2label))
            print(len(mlp2label))
            print(len(duibi2label))
            
            
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
        #print(new_dict)
        #exit()
        if name == 'coli':
            for k,v in new_dict.items():
                v = [int(i) for i in v]
                if sum(v[:3]) == 0 or sum(v[:3]) == 3:
                    with open(file_path__, "a") as file: file.write(k+ '\t'+ str(v[-1]) + '\n')
                    
                if sum(v[:3]) == 2 or sum(v[:3]) == 1:
                    with open(file_path___, "a") as file: file.write(k+ '\t'+ str(v[-1]) + '\n')
            
        for k,v in new_dict.items():
            v = [int(i) for i in v]
            #print(k)
            #exit()

                
            
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
                
        all = zheng_000 + zheng_111 + cuo_000 + cuo_111
        all_dui = zheng_000 + zheng_111
        all_cuo = cuo_000+cuo_111
        acc = round((all_dui/all),4)
        rest = len(test2label) - all
        result = '物种{}\t全部预测一致：{}\t预测对{}\t预测错{}\t准确率{}\t  测试集总数{}\t没进行预测的个数{}'.format(name,all,all_dui,all_cuo,acc,len(test2label),rest)
        with open(file_path, "a") as file: file.write(result + '\n')
        
        
        cubt_duibi = 0
        cubt_duibi_cuo = 0
        for k,v in new_dict.items():

            v = [int(i) for i in v]
            if v[0] ==v[2] == v[-1]:  #v[0] == v[1] ==v[-1]:
                cubt_duibi+=1
                
            if v[0] ==v[2]!=v[-1]:  #v[0] == v[1] !=v[-1]:
                cubt_duibi_cuo+=1
         
        all_ = cubt_duibi + cubt_duibi_cuo
        acc_ = round((cubt_duibi / (cubt_duibi + cubt_duibi_cuo)),4)
        rest_ = len(test2label) - all_
        #print(rest_)
        #exit()
        result_ = '物种{}\t全部预测一致：{}\t预测对{}\t预测错{}\t准确率{}\t  测试集总数{}\t没进行预测的个数{}'.format(name,cubt_duibi + cubt_duibi_cuo,cubt_duibi,cubt_duibi_cuo,acc_,len(test2label),rest_)
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
        #print(zheng_000)
        #print(cubt_duibi)
        #print(cubt_duibi_cuo)
        #print('总数：',str(len(test2label)))
        #exit()
        
        
        
        
        
        
        
        
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
        
        #pre = (zheng_000 + zheng_111)/(zheng_000 + zheng_111+cuo_000+cuo_111)
        #rest = len(test2label) - (zheng_000 + zheng_111 + cuo_000 + cuo_111)
        #
        #print(pre)
        #print('总数：',str(len(test2label)))
        #print('丢弃数：',rest)
        print('*******************************')
        
        
        

        
        
            
            
            
            

            
    # all_dui  = 0
    # for k,v in new_dict.items():  #列表的第一个cubt结果，第二个对比结果，第三个mlp结果，第四个真实标签
        # v = [int(i) for i in v]
        
        # if v[3] == 0 and sum(v[:3]) <=1 or v[3] == 1 and sum(v[:3])>=2:
        # if sum(v[:3]) <=1 or sum(v[:3])>=2:
            # all_dui +=1
            
    # print('投票准确率：' + str(all_dui/len(new_dict)) + '\t个数为' + str(all_dui))

    # a =0
    # for k,v in new_dict.items():
        # v = [int(i) for i in v]
        
        # if sum(v[:3]) == 3 and v[-1] == 0 or sum(v[:3]) == 0 and v[-1] == 1:
            # a += 1
    # print(a)
