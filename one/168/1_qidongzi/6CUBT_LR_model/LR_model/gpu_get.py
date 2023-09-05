#!/usr/bin/python
# coding=utf-8
import re
import os
import sys


def GPU(server):

    os.system("nvidia-smi > gpu_info.txt")

    fr = open("gpu_info.txt", 'r')
    pattern = re.compile(r'(?<=W\|).*?(?=MiB)')
    result = []

    for line in fr.readlines():
        line = line.replace(' ', '')
        target = pattern.findall(line)
        if target:
            result.extend(target)
    #print(result)
    #exit()
    print result
    if server == "67":
        gpu_list=[3,2,1,0]
    else:
        gpu_list=[0,1,2,3]
    gpu_no = 0
    flag = False
    for i in range(len(result)-1,-1,-1):
        if int(result[i]) < 3000:
            gpu_no = i
            flag = True
            break
    if flag == True:
        return  gpu_list[gpu_no]
    else:
        return -1
if __name__=="__main__":
    server=sys.argv[1]
    gpu=GPU(server)
    print gpu
