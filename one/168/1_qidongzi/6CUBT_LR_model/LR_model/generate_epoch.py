#!/usr/bin/python
# -*- coding: utf-8 -*-
import os,sys
import time
from datetime import datetime
import re
import numpy as np
#import config_dcnn
#在关键节点打印时间
def now():
    return str(datetime.now())+"|"
def get_epoch_file(batch_size):
    os.system("sh data/generate_batch.sh %s"% batch_size)

def give_train_batch():
    batch_name=[]
    for _,_,files in os.walk("data/batch/"):
        for fi in files:
            batch_name.append(fi)
    return batch_name

if __name__=="__main__":
    get_epoch_file(200)
    testname,batch_name=give_test_batch()
    print(testname)
    for i,na in enumerate(batch_name):
        if i %10==0: print(na)
    print len(batch_name)
