from collections import Counter
import glob
from functools import reduce
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
#label_0代表启动子
#label_1代表非启动子




    

def word_sort_m1(words):
    # 先转换成列表
    tmp = [i for i in words if i != ' ']
    res = sorted(tmp)
    return res




def one_seq(sequence):
    bi_lst = []
    tri_lst = []
    fo_lst = []
    five_lst = []
    six_lst = []
    for bi in ["".join(sequence[i:i+2]) for i in range(len(sequence)-1)]:
        bi_lst.append(bi)
    
    for tri in ["".join(sequence[i:i+3]) for i in range(len(sequence)-2)]:
        tri_lst.append(tri)
    for fo in ["".join(sequence[i:i+4]) for i in range(len(sequence)-3)]:
        fo_lst.append(fo)
    for five in ["".join(sequence[i:i+5]) for i in range(len(sequence)-4)]:
        five_lst.append(five)
    for six in ["".join(sequence[i:i+6]) for i in range(len(sequence)-5)]:
        six_lst.append(six)

    fenci = bi_lst + tri_lst + fo_lst + five_lst +six_lst
    return fenci


def load_feat(path):                 #导入box数据，建立一个box为keys,特征为values的字典
    box2fea = {}
    for line in open(path,'r'):
        # print(line)
        # exit()
        box,daibiao,fea = line.strip().split('\t')
        box2fea[box] = fea
    return box2fea

        
def result_(data_qi,box_lst):         #遍历数据，得到一个完整序列为keys，序列中所含的特征为values的字典
    data_ls = []
    data2num_qi = {}
    aa,fea_lst_qi = [],[]
    for j in data_qi:
        if len(j) <=2:
            aa.append(j)
    data = ''.join(i[0] for i in aa) + aa[-1][1]
    for i in box_lst:          #遍历特征
        if i in data_qi:
            fea_lst_qi.append(i)
    data2num_qi[data] = fea_lst_qi
    return data2num_qi

def write_result_(data_dict,write_txt = 'result_1.txt',num = 5): #将结果写入，num代表阈值（一条序列中出现的次数（超过阈值则认为为启动子））
    qq = []
    with open(write_txt,'a')as f:
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
    if len(qq) != 0:
        result = 'Promoter'
    else:
        result = 'Non-promoter'
    return result,v






def find_fea(seq,fea_lst):
    new_seq = "-" * len(seq)  # 初始化新序列，全部使用-代替
    for feature in fea_lst:
        index = seq.find(feature)  # 找到特征在序列中的位置
        if index != -1:  # 如果特征存在于序列中
            new_seq = new_seq[:index] + feature + new_seq[index+len(feature):]  # 将特征替换进新序列中
    return new_seq



def run_main_function():
    # print(selected_species)
    # print(selected_Box)
    sequence = sequence_entry.get()
    Mutation_Loci = int(threshold_entry.get())
    print('阈值:',Mutation_Loci)
    # Mutation = mutation_entry.get()
    # selected_species = 'Haloferax volcanii DS2'  #物种
    # selected_Box = 'AT'   #选择的box
    # sequence = 'GATGAGAAGACTGATTTTACGGGCTCAAAAGACTGGCACACTTCTTGCATTTATAATGGTGAACCCTAAATAGAAGGAGGC'
    # Mutation_Loci = 2
    file_lst = glob.glob('box/*')
    species_files_dict = {}

    for species in species_lst:
        for file in file_lst:
            if species[-3:] in file:
                species_files_dict[species] = file
                break
    file = species_files_dict[selected_species]
    if selected_Box != 'location':
        
        box_path = file + '/analyze/1_select_statistic/box_result.txt'
        box2fea = load_feat(box_path)
    
        box_lst = box2fea[selected_Box].split(',')
        print(box_lst)
        print('=================')
        # print(file + '/analyze/fea2pict_test/index_pic/select_pic_50/*')
    if selected_Box == 'location':
        box_path =  glob.glob(file + '/analyze/fea2pict_test/index_pic/select_pic_50/*')[0]
        # box_lst = []
        for i in open(box_path,'r',encoding = 'utf-8'):
            # box_lst.append(i.)
            
            # print(i)
            # exit()
            # box_lst = [i.split('\\')[-1].split('_')[0] for i in box_path]
            box_lst = i.strip().split('\t')
            
        print(box_lst)
        
    cut_data = one_seq(sequence)

    data2feature = result_(cut_data,box_lst)
    
    result,fea = write_result_(data2feature,write_txt = 'result_result/result.txt',num = Mutation_Loci)
    new_seq = find_fea(sequence,fea)
    print(len(fea))
    part_length = int(len(fea)/3)
    
    part1 = fea[:part_length]
    part2 = fea[part_length:2*part_length]
    part3 = fea[2*part_length:]
    
    fea = str(part1) +'\n' + str(part2) + '\n' + str(part3)
    
    result_text_0.set(f"Forecast results：{result}")
    
    if result == 'Promoter':
        result_text_1.set(f"{fea}")
    else:
        result_text_1.set(f"None")

    result_text_2.set(f"{new_seq}")

def start_thread():
    threading.Thread(target=run_main_function).start()

def update_progressbar(value):
    progress_var.set(value)
    progress_bar.update()


def on_species_select(event):
    global selected_species
    selected_species = species_combobox.get()
    print("Selected species:", selected_species)
    
    a = selected_species
    return selected_species
    # 在这里执行处理所选物种的操作
    
def on_Box_select(event):
    global selected_Box
    selected_Box = box_combobox.get()
    print("Selected Box:", selected_Box)
    return selected_Box
    
    
if __name__ == "__main__":
    species_lst = ['Haloferax volcanii DS2',
                    'Escherichia coli str K-12 substr.MG1655',
                    'Shigella flexneri 5a str. M90T',
                    'Xanthomonas campestris pv. campestrie B100',
                    'Burkholderia cenocepacia J2315',
                    'Bradyrhizobium japonicum USDA 110',
                    'Agrobacterium tumefaciens str C58',
                    'Sinorhizobium meliloti 1021',
                    'Helicobacter pylori strain 26695',
                    'Campylobacter jejuni RM1221',
                    'Paenibacillus riograndensis SBR5',
                    'Bacillus subtilis subsp. subtilis str. 168',
                    'Staphylococcus epidermidis ATCC 12228',
                    'Staphylococcus aureus subsp. aureus MW2',
                    'Corynebacterium diphtheriae NCTC 13129',
                    'Corynebacterium glutamicum ATCC 13032',
                    'coil']
                    
    Box_lst = ['AT',
                'ACT',
                'ACGT',
                'ACT',
                'location']
    
    root = tk.Tk()
    selected_species = ''
    selected_Box = ''
    root.title("Promoter prediction tool")

    # 物种选择下拉框
    species_label = ttk.Label(root, text="Select species:")
    species_label.grid(column=0, row=0, padx=10, pady=10)  # 使用 sticky=tk.E 将标签靠右对齐
    species_combobox = ttk.Combobox(root,values=species_lst, width=40)#, width=40
    species_combobox.grid(column=1, row=0, padx=20, pady=20)
    species_combobox.bind("<<ComboboxSelected>>", on_species_select)
 
    
    
    # BOX选择下拉框
    box_label = ttk.Label(root, text="Selection BOX:")
    box_label.grid(column=2, row=0, padx=5, pady=5)
    box_combobox = ttk.Combobox(root, values=Box_lst)
    box_combobox.grid(column=3, row=0, padx=5, pady=5)
    box_combobox.bind("<<ComboboxSelected>>", on_Box_select)
    
    # 阈值输入框
    threshold_label = ttk.Label(root, text="Threshold:")
    threshold_label.grid(column=0, row=1, padx=10, pady=10)
    threshold_entry = ttk.Entry(root, width=40)
    threshold_entry.grid(column=1, row=1, padx=10, pady=10)

    # 输入序列输入框
    sequence_label = ttk.Label(root, text="Base sequence to be predicted:")
    sequence_label.grid(column=0, row=2, padx=10, pady=10)
    sequence_entry = ttk.Entry(root, width=50)
    sequence_entry.grid(column=1, row=2, padx=10, pady=10)

    calculate_button = ttk.Button(root, text="predict", command=start_thread)
    calculate_button.grid(column=0, row=3, columnspan=2, pady=20)
    
    # 预测出的结果
    result_text_0 = tk.StringVar()
    result_label_0 = ttk.Label(root, textvariable=result_text_0)
    result_label_0.grid(column=0, row=4, columnspan=2, pady=10)

    result_label_1 = ttk.Label(root, text="Identified features:")
    result_label_1.grid(column=0, row=5, columnspan=2, pady=10)
    result_text_1 = tk.StringVar()
    # result_text_1.set("结果1")
    result_label_1_value = ttk.Label(root, textvariable=result_text_1)
    result_label_1_value.grid(column=1, row=6, columnspan=2, pady=10, sticky='nsew')

    result_label_2 = ttk.Label(root, text="Visualize sequence features:")
    result_label_2.grid(column=0, row=7, columnspan=2, pady=10)
    result_text_2 = tk.StringVar()
    # result_text_2.set("结果2")
    result_label_2_value = ttk.Label(root, textvariable=result_text_2)
    result_label_2_value.grid(column=1, row=8, columnspan=2, pady=10, sticky='nsew')

    root.mainloop()




#TGTAATAAAAAGAAAAGATTACGTGCCTGAATCTTCTCTTTATCAGCAGTAAACTAGTGGGTATTCATCCCCCTACCTCTT