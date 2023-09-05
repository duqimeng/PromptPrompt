#!-*-coding:utf-8-*-
import glob
import sys,os
import pickle
import json
char_limit={}
uni_limit={}
bi_limit={}
tri_limit={}
fo_limit={}
five_limit={}
six_limit={}



total=0
for j,filename in enumerate(glob.glob("classify_data/*")):
    for k,line in enumerate(open(filename,"r")):
        if k%1000==0:
            print k,filename
        try:
            char,uni,bi,tri,fo,five,six,tag,raw_query,label=json.loads(line.strip())
        except Exception as e:
            print e
            print line.strip()
            #exit()
            continue
        len_char=len(char.split(" "))
        len_uni=len(uni.split(" "))
        len_bi=len(bi.split(" "))
        len_tri=len(tri.split(" "))
        len_fo=len(fo.split(" "))
        len_five=len(five.split(" "))
        len_six=len(six.split(" "))
        total +=1
        char_limit[len_char]=char_limit.get(len_char,0)+1 
        uni_limit[len_uni]=uni_limit.get(len_uni,0)+1
        bi_limit[len_bi]=bi_limit.get(len_bi,0)+1 
        tri_limit[len_tri]=tri_limit.get(len_tri,0)+1         
        fo_limit[len_fo]=fo_limit.get(len_fo,0)+1
        
        five_limit[len_five]=five_limit.get(len_five,0)+1         
        six_limit[len_six]=six_limit.get(len_six,0)+1
char_limit_90=[]
uni_limit_90=[]
bi_limit_90=[]
tri_limit_90=[]
fo_limit_90=[]

five_limit_90=[]
six_limit_90=[]
with open("otherResult/charlimit","w") as fc,open("otherResult/unilimit","w") as fu,open("otherResult/bilimit","w") as fb,open("otherResult/trilimit","w") as ft,open("otherResult/folimit","w") as ff,open("otherResult/fivelimit","w") as fi,open("otherResult/sixlimit","w") as fs:
    acc_char=0
    acc_char=0
    acc_uni=0
    acc_bi=0
    acc_tri=0
    acc_fo=0
    
    acc_five=0
    acc_six=0
    
    for l,count in sorted(char_limit.items(),key=lambda x:x[0]):
        acc_char +=count
        
        if float(acc_char)/total>0.9 and len(char_limit_90)==0:
            char_limit_90.append(l)
        
        fc.write("%s\t%s\t%s\n"%(l,count,float(acc_char)/total))
    for l,count in sorted(uni_limit.items(),key=lambda x:x[0]):
        acc_uni +=count
        if float(acc_uni)/total>0.9 and len(uni_limit_90)==0:
            uni_limit_90.append(l)
        fu.write("%s\t%s\t%s\n"%(l,count,float(acc_uni)/total))
    for l,count in sorted(bi_limit.items(),key=lambda x:x[0]):
        acc_bi +=count
        if float(acc_bi)/total>0.9 and len(bi_limit_90)==0:
            bi_limit_90.append(l)
        fb.write("%s\t%s\t%s\n"%(l,count,float(acc_bi)/total))
   
    for l,count in sorted(tri_limit.items(),key=lambda x:x[0]):
        acc_tri +=count
        if float(acc_tri)/total>0.9 and len(tri_limit_90)==0:
            tri_limit_90.append(l)
        ft.write("%s\t%s\t%s\n"%(l,count,float(acc_tri)/total))
    for l,count in sorted(fo_limit.items(),key=lambda x:x[0]):
        acc_fo +=count
        if float(acc_fo)/total>0.9 and len(fo_limit_90)==0:
            fo_limit_90.append(l)
        ff.write("%s\t%s\t%s\n"%(l,count,float(acc_fo)/total))
        
        
        
        
    for l,count in sorted(five_limit.items(),key=lambda x:x[0]):
        acc_five +=count
        if float(acc_five)/total>0.9 and len(five_limit_90)==0:
            five_limit_90.append(l)
        fi.write("%s\t%s\t%s\n"%(l,count,float(acc_five)/total))

    for l,count in sorted(six_limit.items(),key=lambda x:x[0]):
        acc_six +=count
        if float(acc_six)/total>0.9 and len(six_limit_90)==0:
            six_limit_90.append(l)
        fs.write("%s\t%s\t%s\n"%(l,count,float(acc_six)/total))

print char_limit_90 
print uni_limit_90
print bi_limit_90
print tri_limit_90
print fo_limit_90

print five_limit_90
print six_limit_90

print "Done!"
#exit()
data_info_path=glob.glob("result/data_info.pkl")
if data_info_path !=[]:
    with open(data_info_path[0],"r")as f:
        data_info=pickle.load(f)
else:
    print "data_info_wrong"
    exit()

data_info["char_limit"]=char_limit_90[0] if char_limit_90[0] <1000 else 1000
data_info["uni_limit"]=uni_limit_90[0] if uni_limit_90[0] < 1000 else 1000
data_info["bi_limit"]=bi_limit_90[0] if bi_limit_90[0] <1000 else 1000
data_info["tri_limit"]=tri_limit_90[0] if tri_limit_90[0]<1000 else 1000
data_info["fo_limit"]=fo_limit_90[0] if fo_limit_90[0]<1000 else 1000
data_info["five_limit"]=five_limit_90[0] if five_limit_90[0]<1000 else 1000
data_info["six_limit"]=six_limit_90[0] if six_limit_90[0]<1000 else 1000


with open("result/data_info.pkl","w")as f:
    pickle.dump(data_info,f)
