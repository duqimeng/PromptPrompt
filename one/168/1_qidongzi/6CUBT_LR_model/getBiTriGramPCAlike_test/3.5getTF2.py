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
import glob
import numpy as np
def givelabel2count_uni_bi_tri2cnt(filenames,len_label):
    for filename in filenames:
        name=filename.split("/")[-1]
        X=name.split("_")[-1]
        idd=2
         
        with open(filename,"r")as fr,open("statistic_2/"+name,"w")as fw,open("result/"+X+"2id","w")as fx:
            fx.write("%s\t%s\n"%("UNK",0))
            fx.write("%s\t%s\n"%("PAD",1))
            for j,line in enumerate(fr):
                if j%1000==0:print "processing the %s th query"%j
                try:
                    fea,dic=line.strip().split("\t")
                except:
                    print line.strip()
                    continue
                dic=json.loads(dic)
                tf_lst=[]
                print(dic)
                #exit()
                for i in range(len_label):
                    for j in range(i+1,len_label):
                        tf1=dic.get("label_%s"%i,0)
                        tf2=dic.get("label_%s"%j,0)
                        #print(tf1)
                        #print(tf2)
                        #print(abs(tf1-tf2)/max(tf1,tf2))
                        #exit()
                        try:
                           # print dic["label_0"]
                            tf_lst.append(abs(tf1-tf2)/max(tf1,tf2))
                        except:
                            tf_lst.append(0)
                
                tf_distance=[]
                tf_cob=[]
                #print(tf_lst)
                #exit()
                for i in range(len(tf_lst)):
                    #print(i)
                    #exit()
                    for j in range(1,len(tf_lst)):
                        print(i,j)
                        #exit()
                        if i==j or (i,j) in tf_cob or (j,i) in tf_cob: continue
                        tf_cob.append((i,j))
                        #print(tf_lst)
                        #exit()
                        tf_distance.append(abs(tf_lst[i]-tf_lst[j]))
                #exit()
                tf_dis=[1 for x in tf_distance if x<0.07]
                
                print('----------')
                #print(tf_dis)
                #exit() 
                #print(tf_dis)
                #a =tf_lst
                #print('==============')
                #print(tf_lst)
                #print(tf_lst[0])
                #exit()
                if sum(tf_dis)<=2 and np.mean(tf_lst)>0.05 and sum(tf_lst)<2:  #0.1
                    #print(fea)
                    #exit()
                #if np.mean(tf_lst)>0.4: 
                    #print(tf_lst)
                    print('+++++++++++++++++++++')
                    fw.write("%s\t%s\n"%(fea,json.dumps(tf_lst)))
                    fx.write("%s\t%s\n"%(fea,idd))
                    idd +=1
                #if np.mean(tf_lst)>0.3:
                    #fw.write("%s\t%s\n"%(fea,json.dumps(tf_lst)))
                    #fx.write("%s\t%s\n"%(fea,idd))
                    #idd +=1

if __name__ == '__main__':
    data_info_path=glob.glob("result/data_info.pkl")
    if data_info_path !=[]:
        with open(data_info_path[0],"r")as f:
            data_info=pickle.load(f) 
    else:
        print "Error,check result/data_info.pkl if it exist"
        exit() 
    len_label=data_info["limit_label"]
    filenames=glob.glob("statistic/*")
    givelabel2count_uni_bi_tri2cnt(filenames,len_label)

   
