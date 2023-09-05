#!/usr/bin/python
#-*-coding:utf-8-*-
import os,sys
import logging
import tensorflow as tf
import re
import pickle
from datetime import datetime
import math

sr=str(datetime.now())
sr=re.sub("[:. -]","",sr)
time=sr[4:12]
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename="log_",
                    filemode='w')

TF_CONFIG = tf.ConfigProto()
TF_CONFIG.gpu_options.allow_growth = True

####################################
#
#data config
#
####################################
with open("data/dict_info/data_info.pkl","r")as f:
    data_info=pickle.load(f)
print "data_info loaded!"
#print data_info
#exit()
logging.info(data_info)
ba=data_info["total_train"]/4
#print(data_info["total_train"])
#exit()
batch_size=ba if ba<1024 else 1024
print(batch_size,'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
DATA_CONFIG = {
        "batch_size":batch_size,
        "ckpt":"ckpt/",
        }


###################################
#data_info={"first_place_holder_char":{"limit":300,"size":10000},"second_place_holder_uni":{"limit":300,"size":20000}}
#
#
#
######################################
PLACE_HOLDER_DICT=[]
for key,value in data_info.items():
    #print key
    #print value
    if key.find("place_holder")!=-1:
        #print key
        name=int(key.split("_")[0])
        PLACE_HOLDER_DICT.append([name,value["limit"],value["size"]])
        
#exit()
print('========================')
#print(PLACE_HOLDER_DICT)
#exit()
print(PLACE_HOLDER_DICT)
PLACE_HOLDER_DICT=sorted(PLACE_HOLDER_DICT,key=lambda x:x[0],reverse=True)[:-2]
#print(PLACE_HOLDER_DICT)
#exit()
#PLACE_HOLDER_DICT=[[],[],[]]

####################################
#
#model config
####################################
MODEL_CONFIG = {
        "learning_rate":3e-4,
        "l2_regularization":0.7,
        "l1_regularization":0.1,
        "dropout":0.5,

        }

TRAIN_CONFIG = {
        "max_epoch":101,
        }
