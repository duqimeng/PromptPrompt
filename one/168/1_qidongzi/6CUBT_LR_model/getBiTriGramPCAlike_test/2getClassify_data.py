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

def GE(filename):
    print "now classify the sample by label with number of samples lager than a given value!"
    t={}
    for j,line in enumerate(open("../../Data/"+filename,"r")):
        if j%1000==0:print "processing the %s th query"%j
        try:
            query,label=line.strip().split("\t")
        except:
            continue
        if label in t:
            t[label].append(query)
        else:
            t[label]=[query]
    print "saving the files"

    for j,labelquerys in enumerate(t.items()):
        label,querys=labelquerys
       # print label
        #print querys
        #exit()
        with open("classify_data/label_%s"%j,"w")as f:
            for k,query in enumerate(querys):
                if k%1000==0:print "processing %s th query in label %s"%(k,j)
                char=""
                for ch in query.decode("utf-8"):
                    if ch in char_list:
                        char +=" "+ch
                tokens=query#.split(" ")
                #print(tokens)
                #exit()
                uni=""
                for to in tokens:
                    if to in uni_list:
                        uni += " "+to
                bi=""
                for bii in ["+".join(tokens[i:i+2]) for i in range(len(tokens)-2)]:
                    if bii in bi_list:
                        bi +=" "+bii
                tri=""
                #print(tri_list)
                #exit()
                for tii in ["+".join(tokens[i:i+3]) for i in range(len(tokens)-3)]:
                    if tii in tri_list:
                        tri +=" "+tii
                fo=""
                for foo in ["+".join(tokens[i:i+4]) for i in range(len(tokens)-4)]:
                    if foo in fo_list:
                        fo +=" "+foo


                five=""
                for fiv in ["+".join(tokens[i:i+5]) for i in range(len(tokens)-5)]:
                    if fiv in five_list:
                        five +=" "+fiv

                six=""
                for si in ["+".join(tokens[i:i+6]) for i in range(len(tokens)-6)]:
                    if si in six_list:
                        six +=" "+si


                raw_query="".join(tokens)
                try:
                #    print label2id[label]
                #    print "==============="
                    f.write("%s\n"%json.dumps([char.encode("utf-8"),uni,bi,tri,fo,five,six,label2id[label],raw_query,label],ensure_ascii=False))
                except Exception as e:
                    print e
                    print label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='test_uni_parser', type=str, help='输入待分词文件')
    parser.add_argument('--which', default='train', type=str, help='输入待分词文件')
    args = parser.parse_args()
    file_name = args.file

    print "load label2id"


    print "now load list"
    char_list={}
    uni_list={}
    bi_list={}
    tri_list={}
    fo_list={}
    
    five_list ={}
    six_list ={}
    
    name_list=glob.glob("result/*")
    for name in name_list:
        if name.find("char2cnt")!=-1:
            char_file=name
        if name.find("uni2cnt")!=-1:
            uni_file=name
        if name.find("bi2cnt")!=-1:
            bi_file=name
        if name.find("tri2cnt")!=-1:
            tri_file=name
        if name.find("fo2cnt")!=-1:
            fo_file=name
            
        if name.find("five2cnt")!=-1:
            five_file=name
        if name.find("six2cnt")!=-1:
            six_file=name

        if name.find("label2cnt")!=-1:
            label_file=name
    if args.which =="train":
        label2id={}
        with open("result/label2id","w")as f:
            for idd,line in enumerate(open(label_file,"r")):
                acc,count,label=line.strip().split("\t")
                label2id[label]=int(idd)
                f.write("%s\t%s\n"%(label,idd))
    else:
        label2id={}
        for line in open("result/label2id","r"):
            label,idd=line.strip().split("\t")
            label2id[label]=idd
    for line in open(char_file,"r"):
        try:
            _,_,key=line.strip().split("\t")
            char_list[key.decode("utf-8")]=1
        except:
            print line.strip()
            continue
    for line in open(uni_file,"r"):
        try:
            _,_,key=line.strip().split("\t")
            uni_list[key]=1
        except:
            print line.strip()
            continue

    for line in open(bi_file,"r"):
        try:
            _,_,key=line.strip().split("\t")
            bi_list[key]=1
        except:
            print line.strip()
            continue

    for line in open(tri_file,"r"):
        try:
            _,_,key=line.strip().split("\t")
            tri_list[key]=1
            print('==========================')
            print(tri_list)
        except:
            print line.strip()
            continue
    for line in open(fo_file,"r"):
        try:

            _,_,key=line.strip().split("\t")
            fo_list[key]=1
        except:
            print line.strip()
            continue


    for line in open(five_file,"r"):
        try:
            _,_,key=line.strip().split("\t")
            five_list[key]=1
        except:
            print line.strip()
            continue
    for line in open(six_file,"r"):
        try:

            _,_,key=line.strip().split("\t")
            six_list[key]=1
        except:
            print line.strip()
            continue



    print "list is loaded"


    GE(file_name)
    #GE(filename,1.0,t)

    if args.which =="train":
        data_info_path=glob.glob("result/data_info.pkl")
        if data_info_path !=[]:
            with open(data_info_path[0],"r")as f:
                data_info=pickle.load(f)
        else:
            data_info={}
        data_info["limit_label"]=len(label2id)
        #print(data_info)
        #exit()
        with open("result/data_info.pkl","w")as f:
            pickle.dump(data_info,f)
