#-*-coding:utf-8-*-
import glob
import os,sys
import time
import pdb
import pickle
import logging
import re
import random
import json
import argparse

def givelabel2count_uni_bi_tri2cnt(filename,min_cha,min_uni,min_bi,min_tri,min_fo,min_six,min_five):
    print "now give label2count"
    label2count={}
    label2count2ratial={}

    t_char2cnt={}
    t_uni2cnt={}
    t_bi2cnt={}
    t_tri2cnt={}
    t_fo2cnt={}
    t_five2cnt = {}
    t_six2cnt ={}
    total=0
    total_char=0
    total_uni=0
    total_bi=0
    total_tri=0
    total_fo=0
    total_five=0
    total_six=0  
    for j,line in enumerate(open("../../Data/"+filename,"r")):
        #print(filename)
        #exit()
        if j%1000==0:print "processing the %s th query"%j
        #if j>1000: break
        try:
            query,label=line.strip().split("\t")
        except:
            continue
        label2count[label]=label2count.get(label,0)+1
        total +=1
       # print label2count
        
        for char in query.decode("utf-8"):
            #print(char)
            #exit()
            if char in [" ".decode("utf-8"),"  ".decode("utf-8")]:continue
            t_char2cnt[char]=t_char2cnt.get(char,0)+1
            total_char +=1
        #print(query)
        tokens=query#.split(" ")
        #print(tokens)
        for token in tokens:
            if token in [" ","  ","   "]: continue
            t_uni2cnt[token]=t_uni2cnt.get(token,0)+1
            total_uni +=1
        #print(len(tokens))
        #print(["+".join(tokens[i:i+2]) for i in range(len(tokens)-2)])
        #exit()
        for bi in ["+".join(tokens[i:i+2]) for i in range(len(tokens)-2)]:
            if bi.find("|")!=-1:continue
            t_bi2cnt[bi]=t_bi2cnt.get(bi,0)+1
            total_bi +=1
        for tri in ["+".join(tokens[i:i+3]) for i in range(len(tokens)-3)]:
            if tri.find("|")!=-1:continue
            t_tri2cnt[tri]=t_tri2cnt.get(tri,0)+1
            total_tri +=1
        for fo in ["+".join(tokens[i:i+4]) for i in range(len(tokens)-4)]:
            if fo.find("|")!=-1:continue
            t_fo2cnt[fo]=t_fo2cnt.get(fo,0)+1
            total_fo +=1

        for five in ["+".join(tokens[i:i+5]) for i in range(len(tokens)-5)]:
            if five.find("|")!=-1:continue
            t_five2cnt[five]=t_five2cnt.get(five,0)+1
            total_five +=1
            
        for six in ["+".join(tokens[i:i+6]) for i in range(len(tokens)-6)]:
            if six.find("|")!=-1:continue
            t_six2cnt[six]=t_six2cnt.get(six,0)+1
            total_six +=1
            
    print "now saving label2count"
    acc=0
    with open("result/"+filename+"_label2cnt","w") as f:
        for label,count in sorted(label2count.items(),key=lambda x:x[1],reverse=True):
            acc +=count
            label2count2ratial[label]=[count,float(acc)/total]
            f.write("%s\t%s\t%s\n"%(float(acc)/total,count,label))
    
    acc_char=0
    with open("result/"+filename+"_char2cnt","w") as f:
        for char,count in sorted(t_char2cnt.items(),key=lambda x:x[1],reverse=True):
            if count < min_char:break
            acc_char +=count
            f.write("%s\t%s\t%s\n"%(float(acc_char)/total_char,count,char.encode("utf-8")))
    
    acc_uni=0
    with open("result/"+filename+"_uni2cnt","w") as f:
        for label,count in sorted(t_uni2cnt.items(),key=lambda x:x[1],reverse=True):
            if count <min_uni:break
            acc_uni +=count
            f.write("%s\t%s\t%s\n"%(float(acc_uni)/total_uni,count,label))
    
    acc_bi=0
    with open("result/"+filename+"_bi2cnt","w") as f:
        for label,count in sorted(t_bi2cnt.items(),key=lambda x:x[1],reverse=True):
            if count<min_bi:break
            acc_bi +=count
            f.write("%s\t%s\t%s\n"%(float(acc_bi)/total_bi,count,label))
    
    acc_tri=0
    with open("result/"+filename+"_tri2cnt","w")as f:
        for label,count in sorted(t_tri2cnt.items(),key=lambda x:x[1],reverse=True):
            #print(t_tri2cnt)
            #print(total_tri)
            #exit()
            if count <min_tri:break
            acc_tri +=count
            f.write("%s\t%s\t%s\n"%(float(acc_tri)/total_tri,count,label))
    
    acc_fo=0
    with open("result/"+filename+"_fo2cnt","w")as f:
        for label,count in sorted(t_fo2cnt.items(),key=lambda x:x[1],reverse=True):
            if count <min_fo:break
            acc_fo +=count
            f.write("%s\t%s\t%s\n"%(float(acc_fo)/total_fo,count,label))

    acc_six=0
    with open("result/"+filename+"_six2cnt","w")as f:
        for label,count in sorted(t_six2cnt.items(),key=lambda x:x[1],reverse=True):
            if count <min_six:break
            acc_six +=count
            f.write("%s\t%s\t%s\n"%(float(acc_six)/total_six,count,label))
    
    acc_five=0
    with open("result/"+filename+"_five2cnt","w")as f:
        for label,count in sorted(t_five2cnt.items(),key=lambda x:x[1],reverse=True):
            if count <min_five:break
            acc_five +=count
            f.write("%s\t%s\t%s\n"%(float(acc_five)/total_five,count,label))
            
    return label2count2ratial


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='test_uni_parser', type=str, help='输入待分词文件')
    parser.add_argument('--min_count_char', default=5, type=int, help='字出现的最小频率')
    parser.add_argument('--min_count_uni', default=5, type=int, help='单词出现的最小频率')
    
    
    parser.add_argument('--min_count_bi', default=3, type=str, help='bi-gram出现的最小频率')
    parser.add_argument('--min_count_tri', default=3, type=str, help='tri-gram出现的最小频率')
    parser.add_argument('--min_count_fo', default=3, type=str, help='fo-gram出现的最小频率')
    parser.add_argument('--min_count_five', default=2, type=str, help='five-gram出现的最小频率')
    
    
    
    parser.add_argument('--min_count_six', default=2, type=str, help='six-gram出现的最小频率')
    args = parser.parse_args()
    file_name = args.file
    min_char=args.min_count_char
    min_uni=args.min_count_uni
    min_bi=args.min_count_bi
    min_tri=args.min_count_tri
    min_fo=args.min_count_fo


    min_five=args.min_count_five
    min_six=args.min_count_six
    t=givelabel2count_uni_bi_tri2cnt(file_name,min_char,min_uni,min_bi,min_tri,min_fo,min_five,min_six)

    data_info_path=glob.glob("result/data_info.pkl")
    if data_info_path !=[]:
        with open(data_info_path[0],"r")as f:
            data_info=pickle.load(f)
    else:
        data_info={}
    data_info["1_place_holder_char"]={}
    data_info["2_place_holder_uni"]={}
    data_info["3_place_holder_bi"]={}
    data_info["4_place_holder_tri"]={}
    data_info["5_place_holder_fo"]={}
    
    data_info["6_place_holder_five"]={}
    data_info["7_place_holder_six"]={}
    
    data_info["1_place_holder_char"]["min_count"]=min_char
    data_info["2_place_holder_uni"]["min_count"]=min_uni
    data_info["3_place_holder_bi"]["min_count"]=min_bi
    data_info["4_place_holder_tri"]["min_count"]=min_tri
    data_info["5_place_holder_fo"]["min_count"]=min_fo
    
    data_info["6_place_holder_five"]["min_count"]=min_five
    data_info["7_place_holder_six"]["min_count"]=min_six
    with open("result/data_info.pkl","w")as f:
        pickle.dump(data_info,f)
