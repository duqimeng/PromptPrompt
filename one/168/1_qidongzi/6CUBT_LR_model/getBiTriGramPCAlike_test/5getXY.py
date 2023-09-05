#!/usr/bin/python
#-*-coding:utf-8-*-
import os,sys
import time
import pdb
import pickle
import logging
import re
import glob
import random
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--limit_char', default=0, type=int, help='每个样本字的最大数量')
parser.add_argument('--limit_uni', default=0, type=int, help='每个样本词的最大数量')
parser.add_argument('--limit_bi', default=0, type=int, help='每个样本bi-gram的最大数量')
parser.add_argument('--limit_tri', default=0, type=int, help='每个样本tri-gram的最大数量')
parser.add_argument('--limit_fo', default=0, type=int, help='每个样本fo-gram的最大数量')

parser.add_argument('--limit_five', default=0, type=int, help='每个样本five-gram的最大数量')
parser.add_argument('--limit_six', default=0, type=int, help='每个样本six-gram的最大数量')

parser.add_argument('--portion', default=0.0, type=float, help='测试与训练集的比例')
parser.add_argument('--which', default="train", type=str, help='测试集合还是训练集')
parser.add_argument('--LiYu', default=False, type=bool, help='是否带原始问句')
args = parser.parse_args()

INPUT_LST=["classify_data/*","result/char2id","result/uni2id","result/bi2id","result/tri2id","result/five2id","result/six2id"]
OUT_LST=["result/xy_train","result/xy_test","basic_info.pkl"]


data_info_path=glob.glob("result/data_info.pkl")
if data_info_path!=[]:
    with open(data_info_path[0],"r")as f:
        data_info=pickle.load(f)
else:
    print "data_info_wrong"
    exit()
total_train=0
limit_char=args.limit_char if args.limit_char!=0 else data_info["char_limit"]
limit_uni=args.limit_uni if args.limit_uni!=0 else data_info["uni_limit"]
limit_bi=args.limit_bi if args.limit_bi !=0 else data_info["bi_limit"]
limit_tri=args.limit_tri if args.limit_tri!=0 else data_info["tri_limit"]
limit_fo=args.limit_fo if args.limit_fo!=0 else data_info["fo_limit"]

limit_five=args.limit_five if args.limit_five!=0 else data_info["five_limit"]
limit_six=args.limit_six if args.limit_six!=0 else data_info["six_limit"]

portion=args.portion

CHAR2ID={}
ID2CHAR={}
for line in open("result/char2id","r"):
    char,id=line.strip().split("\t")
    CHAR2ID[char]=int(id)
    ID2CHAR[int(id)]=char

UNI2ID={}
ID2UNI={}
for line in open("result/uni2id","r"):
    try:
        uni,id=line.strip().split("\t")
    except:
        continue
    UNI2ID[uni]=int(id)
    ID2UNI[int(id)]=uni

BI2ID={}
ID2BI={}
for line in open("result/bi2id","r"):
    try:
        bi,id=line.strip().split("\t")
    except:
        continue
    BI2ID[bi]=int(id)
    ID2BI[int(id)]=bi

ID2TRI={}
TRI2ID={}
for line in open("result/tri2id","r"):
    try:
        tri,id=line.strip().split("\t")
    except:
        continue
    ID2TRI[int(id)]=tri
    TRI2ID[tri]=int(id)

ID2FO={}
FO2ID={}
for line in open("result/fo2id","r"):
    try:
        foo,id=line.strip().split("\t")
    except:
        continue
    ID2FO[int(id)]=foo
    FO2ID[foo]=int(id)



ID2FO={}
FI2ID={}
for line in open("result/five2id","r"):
    try:
        five,id=line.strip().split("\t")
    except:
        continue
    ID2FO[int(id)]=five
    FI2ID[five]=int(id)



ID2FO={}
SI2ID={}
for line in open("result/six2id","r"):
    try:
        six,id=line.strip().split("\t")
    except:
        continue
    ID2FO[int(id)]=six
    SI2ID[six]=int(id)


print "all dict are loaded!"
if args.which=="train":
    nname="result/xy_train"
else:
    nname="result/xy_test"
with open(nname,"w") as fid:
    for j,filename in enumerate(glob.glob("classify_data/*")):
        for k,line in enumerate(open(filename,"r")):
            if k%1000==0:
                print k,filename
            try:
                char,uni,bi,tri,fo,five,six,tag,raw_query,label=json.loads(line.strip())
                total_train +=1
            except:
                continue

            tag=int(tag)
            char_id=[]
            uni_id=[]
            bi_id=[]
            tri_id=[]
            fo_id=[]
            five_id=[]
            six_id=[]
            #print(str(char))
            #exit()
            for ch in char.split(" "):
                if ch.encode("utf-8") in CHAR2ID:
                    char_id.append(CHAR2ID[ch.encode("utf-8")])
            char = str(char)
            aa = []
            for i in char.split(' '):
                if i!= '':
                   aa.append(i) 
            raw_data = ''.join(aa)
            #print(len(raw_data))
            #    
            #exit()
            for u in uni.split(" "):
                if u.encode("utf-8") in UNI2ID:
                    uni_id.append(UNI2ID[u.encode("utf-8")])
            
            for bi in bi.split(" "):
                if bi.encode("utf-8") in BI2ID:
                    bi_id.append(BI2ID[bi.encode("utf-8")])
            
            for tr in tri.split(" "):
                if tr.encode("utf-8") in TRI2ID:
                    tri_id.append(TRI2ID[tr.encode("utf-8")])
            #print(fo)
            for foo in fo.split(" "):
                if foo.encode("utf-8") in FO2ID:
                    fo_id.append(FO2ID[foo.encode("utf-8")])
                    #print(FO2ID)
                    #print(foo.encode("utf-8"))
                    #exit()
            
            for fi in five.split(" "):
                if fi.encode("utf-8") in FI2ID:
                    five_id.append(FI2ID[fi.encode("utf-8")])
            
            for si in six.split(" "):
                if si.encode("utf-8") in SI2ID:
                    six_id.append(SI2ID[si.encode("utf-8")])
            
            
            char_id=char_id[:limit_char] if limit_char<len(char_id) else char_id + [0]*(limit_char-len(char_id))
            #print(char_id)
            #print('============')
            #exit()
            uni_id=uni_id[:limit_uni] if limit_uni<len(uni_id) else uni_id + [0]*(limit_uni-len(uni_id))
            #print(limit_uni)
            #exit()
            bi_id=bi_id[:limit_bi] if limit_bi<len(bi_id) else bi_id + [0]*(limit_bi-len(bi_id))
            tri_id=tri_id[:limit_tri] if limit_tri<len(tri_id) else tri_id + [0]*(limit_tri-len(tri_id))
            fo_id=fo_id[:limit_fo] if limit_fo<len(fo_id) else fo_id + [0]*(limit_fo-len(fo_id))
            
            
            five_id=five_id[:limit_five] if limit_five<len(five_id) else five_id + [0]*(limit_five-len(five_id))
            six_id=six_id[:limit_six] if limit_six<len(six_id) else six_id + [0]*(limit_six-len(six_id))
            #print('============')
            #print(args.LiYu)
            #print(tag)
            #print(label)
            label = int(label)
            #exit()
            if args.LiYu: 
	              fid.write("%s\n"%(json.dumps([raw_data,char_id,uni_id,bi_id,tri_id,fo_id,five_id,six_id,tag,raw_query,label]))) 
            else:
                fid.write("%s\n"%(json.dumps([raw_data,char_id,uni_id,bi_id,tri_id,fo_id,five_id,six_id,label]))) 
print "saving xy.ids"

if args.which == "train":

    data_info["portion"]=portion
    data_info["1_place_holder_char"]["limit"]=limit_char
    data_info["2_place_holder_uni"]["limit"]=limit_uni
    data_info["3_place_holder_bi"]["limit"]=limit_bi
    data_info["4_place_holder_tri"]["limit"]=limit_tri
    data_info["5_place_holder_fo"]["limit"]=limit_fo
    
    data_info["6_place_holder_five"]["limit"]=limit_five
    data_info["7_place_holder_six"]["limit"]=limit_six
    
      
    data_info["1_place_holder_char"]["size"]=len(CHAR2ID)+1
    data_info["2_place_holder_uni"]["size"]=len(UNI2ID)+1
    data_info["3_place_holder_bi"]["size"]=len(BI2ID)+1
    data_info["4_place_holder_tri"]["size"]=len(TRI2ID)+1
    data_info["5_place_holder_fo"]["size"]=len(FO2ID)+1
    
    data_info["6_place_holder_five"]["size"]=len(FI2ID)+1
    data_info["7_place_holder_six"]["size"]=len(SI2ID)+1
    data_info["total_train"]=total_train
    with open("result/data_info.pkl","w")as f:
        pickle.dump(data_info,f)
