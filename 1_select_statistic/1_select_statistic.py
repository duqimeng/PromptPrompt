import glob
import os
import shutil
import config as C
#label_0代表启动子
#label_1代表非启动子


def load_one_data(one_path,threshold = 3):
    txt_name = one_path.split('\\')[-1]
    a = []
    txt_name = txt_name + '_result.txt'
    with open('result_1/' + txt_name,'w')as f1:
        for line in open(one_path,'r',encoding = 'utf-8'):
            static,num  = line.strip().split('\t')
            try:
                la_0,la_1 = num.split(',')
                label_0 = float(la_0.split(':')[1])
                label_1 = float(la_1.split(':')[1].split('}')[0])
                if label_0/label_1 >=threshold:
                    new_data = static + '\t' + 'label_0<=>' + str(label_0)+'\t'+ 'label_1<=>' + str(label_1) + '\n'
                    f1.write(new_data)
            except:
                if num.find('label_0') != -1:
                    a.append(static)
                    # new_ = '该特征只在启动子中出现:' + '\t'+static + '\t'+ num
                    # print('该特征只在启动子中出现:',static,num)
                    # f2.write(new_+'\n')
                # if num.find('label_1') != -1:
                    # new_ = '该特征只在非启动子中出现:' + '\t'+static+ '\t' + num
                    # print('该特征只在非启动子中出现',static,num)
                    # f2.write(new_+'\n')
    with open('result_1/other_result.txt','a',encoding = 'utf-8')as f2:  #注意写入方式是累加，换数据集，更新数据需要删除源文件other_result.txt
        if a !=[]:
            for i in a:
                f2.write(i+'\n')

def del_file(filepath):  #查看文件夹内是否有文件，删除此文件夹下所有数据
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            

if __name__ == "__main__":
    data_path = 'all_stastic'
    del_file('result_1')
    for one in glob.glob(data_path+ '/*'):
        load_one_data(one,threshold = C.threshold)      #threshold:阈值，筛选特征使用，阈值越大，筛选条件越严格，得到的有效特征数量越小
    
    
    
    
    #此代码根据给的数据:all_stastic文件夹下的数据，得到筛选的特征，存放在result_1文件夹中
