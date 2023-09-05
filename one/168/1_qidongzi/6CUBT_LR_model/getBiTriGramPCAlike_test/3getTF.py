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

def givelabel2count_uni_bi_tri2cnt(filenames):

    t_char2cnt_f={}
    t_uni2cnt_f={}
    t_bi2cnt_f={}
    t_tri2cnt_f={}
    t_fo2cnt_f={}
    
    t_five2cnt_f={}
    t_six2cnt_f={}

    for filename in filenames:
        name=filename.split("/")[-1]
        total_char=0
        total_uni=0
        total_bi=0
        total_tri=0
        total_fo=0
        
        total_five=0
        total_six=0
        t_char2cnt={}
        t_uni2cnt={}
        t_bi2cnt={}
        t_tri2cnt={}
        t_fo2cnt={}
        
        t_five2cnt={}
        t_six2cnt={}
        for j,line in enumerate(open(filename,"r")):
            #print(line)
            if j%1000==0:print "processing the %s th query"%j
            try:
                char,Uni,Bi,Tri,Fo,Five,Six,Labelid,Query,Label=json.loads(line.strip())        
            except:
                continue
            
            for cha in char.split(" "):
                if cha not in t_char2cnt:
                    t_char2cnt[cha]={}
                t_char2cnt[cha][name]=t_char2cnt[cha].get(name,0)+1
                total_char +=1
            
            for token in Uni.split(" "):
                if token not in t_uni2cnt:
                    t_uni2cnt[token]={}
                t_uni2cnt[token][name]=t_uni2cnt[token].get(name,0)+1
                total_uni +=1
            
            for bi in Bi.split(" "):
                if bi not in t_bi2cnt:
                    t_bi2cnt[bi]={}
                t_bi2cnt[bi][name]=t_bi2cnt[bi].get(name,0)+1
                total_bi +=1

            for tri in Tri.split(" "):
                if tri not in t_tri2cnt:
                    t_tri2cnt[tri]={}
                t_tri2cnt[tri][name]=t_tri2cnt[tri].get(name,0)+1
                total_tri +=1
            
            for fo in Fo.split(" "):
                if fo not in t_fo2cnt:
                    t_fo2cnt[fo]={}
                t_fo2cnt[fo][name]=t_fo2cnt[fo].get(name,0)+1
                total_fo +=1
        
        
            for five in Five.split(" "):
                if five not in t_five2cnt:
                    t_five2cnt[five]={}
                t_five2cnt[five][name]=t_five2cnt[five].get(name,0)+1
                total_five +=1
            
            
            for six in Six.split(" "):
                if six not in t_six2cnt:
                    t_six2cnt[six]={}
                #print()
                t_six2cnt[six][name]=t_six2cnt[six].get(name,0)+1
                total_six +=1
        
        
        
        
        
        for cha,namecount in t_char2cnt.items():
            for name,count in namecount.items():
                if cha not in t_char2cnt_f:
                    t_char2cnt_f[cha]={name:float(count)/total_char}
                t_char2cnt_f[cha][name]=float(count)/total_char
                #print(t_char2cnt_f)
                #exit()

        for uni,namecount in t_uni2cnt.items():
            for name,count in namecount.items():
                if uni not in t_uni2cnt_f:
                    t_uni2cnt_f[uni]={name:float(count)/total_uni}
                t_uni2cnt_f[uni][name]=float(count)/total_uni
        for bi,namecount in t_bi2cnt.items():
            for name,count in namecount.items():
                if bi not in t_bi2cnt_f:
                    t_bi2cnt_f[bi]={name:float(count)/total_bi}
                t_bi2cnt_f[bi][name]=float(count)/total_bi
        for tri,namecount in t_tri2cnt.items():
            for name,count in namecount.items():
                if tri not in t_tri2cnt_f:
                    t_tri2cnt_f[tri]={name:float(count)/total_tri}
                t_tri2cnt_f[tri][name]=float(count)/total_tri
        for fo,namecount in t_fo2cnt.items():
            for name,count in namecount.items():
                if fo not in t_fo2cnt_f:
                    t_fo2cnt_f[fo]={name:float(count)/total_fo}
                t_fo2cnt_f[fo][name]=float(count)/total_fo



        for five,namecount in t_five2cnt.items():
            for name,count in namecount.items():
                if five not in t_five2cnt_f:
                    t_five2cnt_f[five]={name:float(count)/total_five}
                t_five2cnt_f[five][name]=float(count)/total_five
        for six,namecount in t_six2cnt.items():
            for name,count in namecount.items():
                if six not in t_six2cnt_f:
                    t_six2cnt_f[six]={name:float(count)/total_six}
                t_six2cnt_f[six][name]=float(count)/total_six
                
                
                
                

    print "now saving label2count"

    with open("statistic/tf_char","w") as f:
        for char,dic in t_char2cnt_f.items():        
            try:
                f.write("%s\t%s\n"%(char.encode("utf-8"),json.dumps(dic)))
            except:
                f.write("%s\t%s\n"%(char,json.dumps(dic)))
    with open("statistic/tf_uni","w") as f:
        for uni,dic in t_uni2cnt_f.items():
            try:
                f.write("%s\t%s\n"%(uni,json.dumps(dic)))
            except:
                f.write("%s\t%s\n"%(uni.encode("utf-8"),json.dumps(dic)))
    
    with open("statistic/tf_bi","w") as f:
        for bi,dic in t_bi2cnt_f.items():
            try:
                f.write("%s\t%s\n"%(bi,json.dumps(dic)))
            except:
                f.write("%s\t%s\n"%(bi.encode("utf-8"),json.dumps(dic)))
    
    with open("statistic/tf_tri","w") as f:
        for tri,dic in t_tri2cnt_f.items():
            try:

                f.write("%s\t%s\n"%(tri,json.dumps(dic)))
            except:
                f.write("%s\t%s\n"%(tri.encode("utf-8"),json.dumps(dic)))
    
    with open("statistic/tf_fo","w") as f:
        for fo,dic in t_fo2cnt_f.items():
            try:
                f.write("%s\t%s\n"%(fo,json.dumps(dic)))
            except:
                f.write("%s\t%s\n"%(fo.encode("utf-8"),json.dumps(dic)))


    with open("statistic/tf_five","w") as f:
        for five,dic in t_five2cnt_f.items():
            try:

                f.write("%s\t%s\n"%(five,json.dumps(dic)))
            except:
                f.write("%s\t%s\n"%(five.encode("utf-8"),json.dumps(dic)))
    
    with open("statistic/tf_six","w") as f:
        for six,dic in t_six2cnt_f.items():
            try:
                f.write("%s\t%s\n"%(six,json.dumps(dic)))
            except:
                f.write("%s\t%s\n"%(six.encode("utf-8"),json.dumps(dic)))




if __name__ == '__main__':
    filenames=glob.glob("classify_data/*")
    givelabel2count_uni_bi_tri2cnt(filenames)
    
   
