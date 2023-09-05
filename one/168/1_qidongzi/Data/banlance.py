#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os
import sys

#reload(sys)
#sys.setdefaultencoding('utf-8')
def enlarge(data_file,tag_col):
    t_label2content={}
    t_label2cnt={}
    for line in open(data_file,"r"):
        tag=line.strip().split("\t")[tag_col-1]

        if tag not in t_label2content:
            t_label2content[tag]=[line.strip()]
        else:
            t_label2content[tag].append(line.strip())
        t_label2cnt[tag]=t_label2cnt.get(tag,0)+1
    print "load label2content and label2cnt succeed!"
    for label,tag in sorted(t_label2cnt.items(),key=lambda x:x[1],reverse=True):
        print label,tag
    max_num=max([x[1] for x in t_label2cnt.items()])
    with open(data_file+"_enlarged","w")as f:
        for tag,contents in t_label2content.items():
            print "enlarge label: %s"% tag
            num_enlarge=max_num/t_label2cnt[tag]
            for content in contents:
                for _ in range(num_enlarge):
                    f.write("%s\n"%content)

    print("数据增强完成")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='1', type=str, help='数据文件名')
    parser.add_argument('--tag_col', default=2, type=int, help='标签列')
    args = parser.parse_args()
    data_file = "train_rd_parser"
    tag_col = args.tag_col
    #raw_data_path = "."
    #result_data_path = "result_data"
    #data_file = raw_data_path+'/'+data_file
    enlarge(data_file,tag_col)
    os.system("mv %s %s"%(data_file,"train_rd_parser_bak"))
    os.system("mv %s_enlarged %s"%(data_file,data_file))




