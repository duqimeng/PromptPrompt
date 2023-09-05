#!/usr/bin/python
#-*-coding:utf-8-*-
#zhouchichun
#/usr/bin/env python
# -*- coding: UTF-8 -*-
import os,sys
import numpy as np
import tensorflow as tf
import pdb
import time
import pickle
import json
import random
import glob
import tensorflow as tf
import config
sys.path.append("../")
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops
from config import *

class model(object):
    def __init__(self,sess,config):
        self.logging=config.logging
        self._sess = sess
        self._place_holder_list=config.PLACE_HOLDER_DICT
        print  "============================="
        print self._place_holder_list

        self.load_id2label()
        self._tag_colunms=len(self.id2label)
        #print(self._tag_colunms)
        #exit()
        

        self._lr = config.MODEL_CONFIG["learning_rate"]
        self._l2_regularization = config.MODEL_CONFIG["l2_regularization"]
        self._l1_regularization = config.MODEL_CONFIG["l1_regularization"]
        self._drop_out = config.MODEL_CONFIG["dropout"]
        
        #self.raw_data=tf.placeholder(tf.float32)

        
        self._checkpoint_path = config.DATA_CONFIG ["ckpt"]
        self._activation = lambda x:x*tf.nn.relu(x)
        
        self._global_step = tf.Variable(0,trainable=False)
        self._activation = lambda x:x*tf.nn.relu(x)
        self._initializer = initializers.xavier_initializer()

        self.load_place_holder()
        
        self.buildLoss()
        self.buildOptimizer2()
        self._saver = tf.train.Saver(tf.global_variables(),max_to_keep=4)
        self.initializer()
        self.loggingAll()

    def load_id2label(self):
        ID2LABEL={}
        LABEL2ID={}
 
        for line in open("data/dict_info/label2id","r"):
            label,id=line.strip().split("\t")
            ID2LABEL[int(id)]=label
            LABEL2ID[label]=int(id)
        self.id2label=ID2LABEL
    def loggingAll(self):
        for name in dir(self):
            if name.find("_") == 0 and name.find("__") == -1:
                self.logging.info("self.%s\t%s"%(name,str(getattr(self,name))))

    def readCKPT(self):
        ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
        if ckpt:
            self.logging.info("reading training record from '%s'"%ckpt.model_checkpoint_path)
            self._saver.restore(self._sess,ckpt.model_checkpoint_path)
            return True
        return False

    def savePB(self):
        if self.readCKPT():
            output_node_names = ["loss/Sigmoid"]
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                                        self._sess,
                                        self._sess.graph_def,
                                        output_node_names=output_node_names
                                        )
            with tf.gfile.FastGFile("model.pb",mode="wb") as f:
                f.write(output_graph_def.SerializeToString())
            self.logging.info("pb file is saved")
        else:
            self.logging.warn("there is nothing to be save")



    def load_place_holder(self):
        self.logging.info("build_place_holder")
        self._dropout=tf.placeholder(tf.float32) 
    
        self.bianma_lst=tf.placeholder(tf.float32,[None,2]) 
        self._tag=tf.placeholder(tf.int32,[None,self._tag_colunms],name = "tag")
        with tf.variable_scope("alpha"):
            self.alpha = tf.Variable(tf.constant(0.5,shape=[2]),name="ap")
        with tf.variable_scope("input_linear_layer"):
            self._lr_out=[]
            self._place_holders=[]
            #exit()
            #print(self._place_holder_list)
            if len (self._place_holder_list)!=0:
                name,limit,size=self._place_holder_list.pop()
                name=str(name)
                #print(name,limit,size) 
                #print('==================')
                #exit()
                self._feature1_vocab = size    #14
                self._feature1_limit = limit   #80
                self._feature1_id = tf.placeholder(tf.int32,[None,self._feature1_limit],name = "feature_%s"%name) #   ?,80
                #print(self._feature1_id)
                #exit()
                self._place_holders.append(self._feature1_id)
                self._feature1_weight = tf.get_variable(name = name+"_feature1_weight",shape = [self._feature1_vocab,self._tag_colunms],initializer = self._initializer)  #14,2
                #print(self._feature1_weight)
                #exit()
                self._feature1_a = tf.nn.embedding_lookup(self._feature1_weight,self._feature1_id)  #?,80,2
                #print(self._feature1_a)
                #exit()
                self._feature1_bias = tf.get_variable(name = "feature1_bias",shape = [self._tag_colunms],initializer = tf.constant_initializer(0.00000001)) #2,
                #print(self._feature1_bias)
                #exit()
                self._lr_out.append(tf.reduce_sum(self._feature1_a,1)+self._feature1_bias)

            if len (self._place_holder_list)!=0:
                name,limit,size=self._place_holder_list.pop()
                #print(name,limit,size)
                #exit()
                name=str(name)
                self._feature2_vocab = size
                self._feature2_limit = limit
                self._feature2_id = tf.placeholder(tf.int32,[None,self._feature2_limit],name = "feature_%s"%name)
                self._place_holders.append(self._feature2_id)
                self._feature2_weight = tf.get_variable(name = name+"_feature1_weight",shape = [self._feature2_vocab,self._tag_colunms],initializer =self._initializer)
                self._feature2_a = tf.nn.embedding_lookup(self._feature2_weight,self._feature2_id)
                self._feature2_bias = tf.get_variable(name = "feature2_bias",shape = [self._tag_colunms],initializer = tf.constant_initializer(0.00000001))
                self._lr_out.append(tf.reduce_sum(self._feature2_a,1)+self._feature2_bias)

            if len (self._place_holder_list)!=0:
                name,limit,size=self._place_holder_list.pop()
                name=str(name)
                self._feature3_vocab = size
                self._feature3_limit = limit
                self._feature3_id = tf.placeholder(tf.int32,[None,self._feature3_limit],name = "feature_%s"%name)
                self._place_holders.append(self._feature3_id)
                self._feature3_weight = tf.get_variable(name = name+"_feature3_weight",shape = [self._feature3_vocab,self._tag_colunms],initializer =self._initializer)
                self._feature3_a = tf.nn.embedding_lookup(self._feature3_weight,self._feature3_id)
                self._feature3_bias = tf.get_variable(name = "feature3_bias",shape = [self._tag_colunms],initializer = tf.constant_initializer(0.00000001))
                self._lr_out.append(tf.reduce_sum(self._feature3_a,1)+self._feature3_bias)

            if len (self._place_holder_list)!=0:
                name,limit,size=self._place_holder_list.pop()
                name=str(name)
                self._feature4_vocab = size
                self._feature4_limit = limit
                self._feature4_id = tf.placeholder(tf.int32,[None,self._feature4_limit],name = "feature_%s"%name)
                self._place_holders.append(self._feature4_id)
                self._feature4_weight = tf.get_variable(name = name+"_feature4_weight",shape = [self._feature4_vocab,self._tag_colunms],initializer = self._initializer)
                self._feature4_a = tf.nn.embedding_lookup(self._feature4_weight,self._feature4_id)
                self._feature4_bias = tf.get_variable(name = "feature4_bias",shape = [self._tag_colunms],initializer = tf.constant_initializer(0.00000001))
                self._lr_out.append(tf.reduce_sum(self._feature4_a,1)+self._feature4_bias)

            if len (self._place_holder_list)!=0:
                name,limit,size=self._place_holder_list.pop()
                name=str(name)
                self._feature5_vocab = size
                self._feature5_limit = limit
                self._feature5_id = tf.placeholder(tf.int32,[None,self._feature5_limit],name = "feature_%s"%name)
                self._place_holders.append(self._feature5_id)
                self._feature5_weight = tf.get_variable(name = name+"_feature5_weight",shape = [self._feature5_vocab,self._tag_colunms],initializer =self._initializer)
                self._feature5_a = tf.nn.embedding_lookup(self._feature5_weight,self._feature5_id)
                self._feature5_bias = tf.get_variable(name = "feature5_bias",shape = [self._tag_colunms],initializer = tf.constant_initializer(0.00000001))
                self._lr_out.append(tf.reduce_sum(self._feature5_a,1)+self._feature5_bias)

            if len (self._place_holder_list)!=0:
                name,limit,size=self._place_holder_list.pop()
                name=str(name)
                self._feature6_vocab = size
                self._feature6_limit = limit
                self._feature6_id = tf.placeholder(tf.int32,[None,self._feature6_limit],name = "feature_%s"%name)
                self._place_holders.append(self._feature6_id)
                self._feature6_weight = tf.get_variable(name = name+"_feature6_weight",shape = [self._feature6_vocab,self._tag_colunms],initializer =self._initializer)
                self._feature6_a = tf.nn.embedding_lookup(self._feature6_weight,self._feature6_id)
                self._feature6_bias = tf.get_variable(name = "feature6_bias",shape = [self._tag_colunms],initializer = tf.constant_initializer(0.00000001))
                self._lr_out.append(tf.reduce_sum(self._feature6_a,1)+self._feature6_bias)
                
                
            if len (self._place_holder_list)!=0:
                name,limit,size=self._place_holder_list.pop()
                name=str(name)
                self._feature7_vocab = size
                self._feature7_limit = limit
                self._feature7_id = tf.placeholder(tf.int32,[None,self._feature7_limit],name = "feature_%s"%name)
                self._place_holders.append(self._feature7_id)
                self._feature7_weight = tf.get_variable(name = name+"_feature6_weight",shape = [self._feature6_vocab,self._tag_colunms],initializer =self._initializer)
                self._feature7_a = tf.nn.embedding_lookup(self._feature7_weight,self._feature7_id)
                self._feature7_bias = tf.get_variable(name = "feature7_bias",shape = [self._tag_colunms],initializer = tf.constant_initializer(0.00000001))
                self._lr_out.append(tf.reduce_sum(self._feature7_a,1)+self._feature7_bias)
                

        self._batch_size = tf.shape(self._feature1_id)[0]
        
        
    #def 
        
    def buildLoss(self):
        self.logging.info("buildloss")
        #print "==================================="
        self._lr_out_final=0
        self.coe=[]
        self.max_=[]
        with tf.variable_scope("coe"):
            for j in range(len(self._lr_out)):
                #print(tf.Variable(0.01,name="coe_%s"%j),'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                self.coe.append(tf.Variable(0.01,name="coe_%s"%j))
        for k,coelinear_out in enumerate(zip(self.coe,self._lr_out)):
            #if k==0:continue
            #print(coelinear_out)
            #exit()
            coe,linear_out=coelinear_out
            #print(coe,linear_out)
            #exit()
            self._lr_out_final += coe*linear_out

        with tf.variable_scope("loss"):
            self._out = tf.nn.sigmoid(self._lr_out_final)
            print('*******************************************************************************')
            
            self._loss = tf.pow(self._out-tf.cast(self._tag,tf.float32),2) 
            self._cost = tf.reduce_sum(self._loss)   #计算张量之和
            
            #softmax
            #self._out = tf.nn.softmax(self._lr_out_final)
            #self._cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._tag,logits=self._lr_out_final))
            #print(self._cost)
            #exit()
    def toonehot(self,tag_nums):
        #self.logging.info("toonehot")
        ret=[]
        for tag_num in tag_nums:
            temp=[0.0]*self._tag_colunms
            try:
                temp[tag_num]=1
                ret.append(temp)
            except Exception as e:
                print e
                print "tag_num",tag_num
                print "self.colume",self._tag_colunms
                print tag_nums
                exit()
        return ret
    
    def buildOptimizer2(self):
        self.logging.info("buildOptimizer2")
        weights_nums_in_l1 = 0
        weights_nums_in_l2 = 0
        with tf.variable_scope("optimizer"):
            for item in dir(self):
                type_string = str(type(getattr(self,item)))
                if type_string.find("Variable") != -1 and item.find("step") == -1:
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS,getattr(self,item))
                    tf.add_to_collection("l2",getattr(self,item))
            #tf.add_to_collection(tf.GraphKeys.WEIGHTS,self._wordid2vec)
            self._l2_regularizer =  tf.contrib.layers.l2_regularizer(scale=self._l2_regularization)
            self._l2_reg_term = tf.contrib.layers.apply_regularization(self._l2_regularizer,tf.get_collection("l2"))
        self._train_op=tf.train.AdamOptimizer(self._lr).minimize(self._cost+self._l2_reg_term)


    def calpre(self,pre,rea,ktop=[1,2,3]):
        total=len(pre)
        right={}
        for p,r in zip(pre,rea):
            rr=list(r).index(max(list(r)))
            for k in ktop:
                if rr in [list(p).index(a) for a in sorted(list(p),reverse=True)[:k]]:
                    right[k] =right.get(k,0)+1
        result={}
        for key,num in right.items():
            result[key]=float(num)/total
        return result

    def printwrongsentence(self,sens,pre,rea):
        self.logging.info("printsentence")
        for idd,pr in enumerate(zip(pre,rea)):
            p,r=pr    
            pp=[int(round(ppp)) for ppp in p]
            rr=[int(round(rrr)) for rrr in r]
            if pp!=rr:
                try:
                    print "decode","".join(sens[idd].split(" ")).decode("utf-8")
                    print "label:",self.id2label[list(rr).index(1)]
                except:
                    print "encode","".join(sens[idd].split(" ")).encode("utf-8")
                    print "label:",self.id2label[list(rr).index(1)]


    def initializer(self):
        if not self.readCKPT():
            self._sess.run(tf.global_variables_initializer())


    #def load_

    def train(self,ith_epoch,filename_tests,filename_batch):
        self.logging.info("%d th epoch!"%ith_epoch)
        print("======================%d th epoch!"%ith_epoch)
        all_train_real = []
        all_train_pre = []
        for i,filename in enumerate(filename_batch):   
            print i,filename
            feed_dict,real_tags=self.load_data_train(filename)
            
            
            #self.raw_data = feed_dict['raw_data']
            #feed_dict.pop('raw_data') # 删除字典中对应的键值对，如果键不存在，返回错误print(dict4)
            #code_path = 'bianma_train.txt' #对比学习经过MLP降到二维
            #data2bianma = self.load_bianma(code_path) 
            #bianma_lst = []
            #for x in self.raw_data:
            #    y2 = data2bianma[x]
            #    bianma_lst.append(y2)
            #self.bianma_lst = tf.convert_to_tensor(bianma_lst, dtype='float32')
            
            #feed_dict
            #print('alpah num',self._sess.run(self.alpha,feed_dict = feed_dict))
            #self.predict_tags_1 = self.alpha*self._out+(1-self.alpha)*self.bianma_lst   #self._lr_out_final         self._out
            #print(feed_dict.keys())
            #exit()
            
            #self._loss = tf.pow(self.predict_tags_1-tf.cast(self._tag,tf.float32),2)
            #self._cost = tf.reduce_sum(self._loss)   #计算张量之和
            #print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
            #print(feed_dict.keys())
            #exit()
            
            global_step,cost,predict_tags,_s= self._sess.run([self._global_step,self._cost,self._out,self._train_op],feed_dict = feed_dict)  #self._cost ：loss
            #print(len(predict_tags))
            #exit()
            all_train_real.extend(real_tags)
            all_train_pre.extend(predict_tags)
            if i %4==0 and i!=0:
                if self._tag_colunms==2:
                    recall0,pre0,f10,recall1,pre1,f11,num=self.cal(predict_tags,real_tags)
                    #top1=self.calpre(predict_tags,real_tags,ktop=[1])[1]
                    #print '训练集准确率：' + str(top1)
                    #print(i,cost,recall0)
                    #self.logging.info("train: steps %d,cost %.5f,recall_0 is %.5f,pre_0 is %s f1_0 is %.5f,recall_1 is %s pre_1 is %.5f, f1_1 is %.5f"%(i,cost,recall0,pre0,f10,recall1,pre1,f11))
                    #self.logging.info("train: right is%s, in pre is %s, in real is %s"%(num[0],num[1],num[2]))
                    
                    try:
                        top1=self.calpre(predict_tags,real_tags,ktop=[1])[1]
                        #print(top1)
                        #exit()
                        #self.logging.info("train: steps %d,cost %.5f,accuracy is %.5f"%(i,cost,top1))
                        self.pre_train=self.calpre(predict_tags,real_tags,ktop=[1])[1]
                    except:
                        print self.calpre(predict_tags,real_tags,ktop=[1])
                        pass     		
                else:
                    try:
                        top1=self.calpre(predict_tags,real_tags,ktop=[1,2,3])[1]
                        top2=self.calpre(predict_tags,real_tags,ktop=[1,2,3])[2]
                        top3=self.calpre(predict_tags,real_tags,ktop=[1,2,3])[3]
                        self.pre_train=self.calpre(predict_tags,real_tags,ktop=[1,2,3])[1]
                        #print(top1)
                        #exit()
                        #self.logging.info("train: steps %d,cost %.5f,accurate is %.5f,top%s acc is %.5f,top%s acc is %.5f"%(i,cost,top1,2,top2,3,top3))
                    except:
                        print self.calpre(predict_tags,real_tags,ktop=[1,2,3])
                        print predict_tags
                        print real_tags
                    
                if i%4==0 and i!=0 :
                    for filename in filename_tests:
                        feed_dict,real_tags=self.load_data_test(filename) 
                        #print(real_tags)
                        #exit()
                        
                        self.raw_data_test = feed_dict['raw_data']
                        feed_dict.pop('raw_data') # 删除字典中对应的键值对，如果键不存在，返回错误print(dict4)
                        #code_test = 'bianma_test.txt' #对比学习经过MLP降到二维
                        #data2bianma_test = self.load_bianma(code_test) 
                        #bianma_test_lst = []
                        #for x2 in self.raw_data_test:
                        #    y1 = data2bianma_test[x2]
                        #    bianma_test_lst.append(y1)
                        #self.bianma_test_lst = tf.convert_to_tensor(bianma_test_lst, dtype='float32')
                        #predict_tags_1 = self.alpha*self._out+(1-self.alpha)*self.bianma_test_lst   #self._lr_out_final         self._out
                        predicts = self._sess.run(self._out,feed_dict = feed_dict)

                        if ith_epoch == config.TRAIN_CONFIG["max_epoch"] -1:
                            with open('cubt_predicts_result','w')as ff:
                                for p,r,raw_data in zip(predicts,real_tags,self.raw_data_test):
                                    pp=list(p).index(max(list(p)))
                                    rr=list(r).index(max(list(r)))
                                    ff.write(str(raw_data) + '\t' + str(pp) +'\t'+str(rr)+ '\n')
                                   #print(raw_data,pp)
                                    #print(pp)
                                    #exit()
                        #print(predicts)
                        #exit()
                        
                        if self._tag_colunms==2:
                    	    recall0,pre0,f10,recall1,pre1,f11,num=self.cal(predicts,real_tags)
                            self.logging.info("test: the accuracy is %s"%self.calpre(predicts,real_tags,[1])[1])
                            print '测试集准确率：' + str(self.calpre(predicts,real_tags,[1])[1])
                            self.pre_test=self.calpre(predicts,real_tags,[1])[1]
                            self.logging.info("test: steps %d,cost %.5f,recall_0 is %.5f,pre_0 is %s f1_0 is %.5f,recall_1 is %s pre_1 is %.5f, f1_1 is %.5f"%(i,cost,recall0,pre0,f10,recall1,pre1,f11))
                            self.logging.info("test: right is%s, in pre is %s, in real is %s"%(num[0],num[1],num[2]))
                        else:
                            try:
                            	top1=self.calpre(predicts,real_tags,[1,2,3])[1]
                            	top2=self.calpre(predicts,real_tags,[1,2,3])[2]
                            	top3=self.calpre(predicts,real_tags,[1,2,3])[3]
                                self.pre_test=self.calpre(predicts,real_tags,[1])[1]
                            	self.logging.info("test on %s: accurate is %.5f,top%s acc is %.5f,top%s acc is %.5f"%(filename,top1,2,top2,3,top3))      
                            except:
                                print self.calpre(predicts,real_tags,[1,2,3])
                                pass
                        self._saver.save(self._sess,self._checkpoint_path+"checkpoint",global_step = global_step)
                        #self.printwrongsentence(query,predicts,y_t)
                    #for j,coe in enumerate(self.coe):
                    #    print "the coe of feature%s is %s"%(j,self._sess.run(coe))
                    
        
        #print(len(all_train_real))
        #exit()
        recall0,pre0,f10,recall1,pre1,f11,num=self.cal(all_train_pre,all_train_real)
        top1=self.calpre(all_train_pre,all_train_real,ktop=[1])[1]
        print '训练集准确率：' + str(top1)
        #print(i,cost,recall0)
        self.logging.info("train: steps %d,cost %.5f,recall_0 is %.5f,pre_0 is %s f1_0 is %.5f,recall_1 is %s pre_1 is %.5f, f1_1 is %.5f"%(i,cost,recall0,pre0,f10,recall1,pre1,f11))
        #self.logging.info("train: right is%s, in pre is %s, in real is %s"%(num[0],num[1],num[2]))
        self.logging.info("train_acc: %s"%(top1))
    def getfinal(self):
        return self.pre_train,self.pre_test
        
        
    def test(self,filenames,give_tag=False):
        self.logging.info("train")
        if not self.readCKPT():
            print "no ckpt"
            exit()
        for filename in filenames:
            feed_dict,real_tags,raw_query,raw_label=self.load_data_test(filename)
            predicts = self._sess.run(self._out,feed_dict = feed_dict)
            if self._tag_colunms==2:
                recall0,pre0,f10,recall1,pre1,f11=self.cal(predict_tags,real_tags)
                self.logging.info("train: steps %d,cost %.5f,recall_0 is %.5f,pre_0 is %s f1_0 is %.5f,recall_1 is %s pre_1 is %.5f, f1_1 is %.5f"%(i,cost,recall0,pre0,f10,recall1,pre1,f11))

            else:
                top1=self.calpre(predicts,real_tags,[1,2,3])[1]
                top2=self.calpre(predicts,real_tags,[1,2,3])[2]
                top3=self.calpre(predicts,real_tags,[1,2,3])[3]
                self.logging.info("test on %s: accurate is %.5f,top%s acc is %.5f,top%s acc is %.5f"%(filename,top1,2,top2,3,top3))

            if give_tag:
                with open(filename+"_tag","w")as f:
                    for k,pr in enumerate(zip(predicts,real_tags,raw_query,raw_label)):
                        if k%10000==0:print "processing %s th sample"%k
                        p,r,rq,rl=pr
			indp=list(p).index(max(list(p)))
			indr=list(r).index(max(list(r)))
                        try:
                            state= "True" if indp==indr else "False"
 			    f.write("%s\t%s\t%s\t%s\n"%(rq.encode("utf-8"),self.id2label[indp],rl.encode("utf-8"),state))
                        except Exception as e:
                            print e
                            print rq,type(rq.encode("utf-8"))
                            print rl,type(rl.encode("utf-8"))
                            f.write("%s\t%s\n"%(self.id2label[indp],self.id2label[indr]))
                            #f.write("%s\t%s\n"%(rq.encode("utf-8"),rl.encode("utf-8")))
                            #print "=-============================"
                            #f.write("%s<==>%s\t%s<==>%s\n"%(rq.encode("utf-8"),self.id2label[indp],self.id2label[indr],rl.encode("utf-8"),indp==indr))
                            #exit()

                #self.printwrongsentence(raw_query,predicts,real_tags)

    def empty(self,num):
        lis=[]
        for i in range(num):
            lis.append([])
        return lis
    
    
    
    
    
    
    def load_bianma(self,file_name):
        data2bianma = {}
        for i in open(file_name,'r'):
            data,bianma,label = i.strip().split('\t')
            bianma = [float(i) for i in bianma.split(',')]
            data2bianma[data] = bianma 
        #print(len(data2bianma))
        #exit()
        return data2bianma
    
    def load_data_train(self,filename):
        fee_dic={}

        featres_list=self.empty(len(self._place_holders))
        tags=[]
        raw_data = []
        for j,line in enumerate(open(filename)):
            #print(line)
            #exit()
            features_tag=json.loads(line.strip())
            #print(features_tag[2:-1])
            #exit()
            raw_data.append(features_tag[0])
            tag=features_tag[-1]
            tags.append(tag)
            for fea,feas in zip(features_tag[3:-1],featres_list):
                feas.append(fea)
        for feaname,fea in zip(self._place_holders,featres_list):
            fee_dic[feaname]=fea
        fee_dic[self._tag]=self.toonehot(tags)
        fee_dic[self._dropout]=self._drop_out
        #print(len(raw_data))
        #print('---')
        #exit()
        #fee_dic['raw_data'] = raw_data
        #feed_dict.pop('raw_data') # 删除字典中对应的键值对，如果键不存在，返回错误print(dict4)
        #code_path = 'pre_code/mlp/bianma_train.txt' #对比学习经过MLP降到二维
        #data2bianma = self.load_bianma(code_path) 
        #bianma_lst = []
        #for x in raw_data:
         #   y2 = data2bianma[x]
         #   bianma_lst.append(y2)
        #self.bianma_lst = tf.convert_to_tensor(bianma_lst, dtype='float32')

        #print "load suceed"
       # fee_dic[self.bianma_lst] = bianma_lst
        return fee_dic,self.toonehot(tags)

    def load_data_test(self,filename):
        fee_dic={}

        featres_list=self.empty(len(self._place_holders))
        tags=[]
        raw_data = []
        for j,line in enumerate(open(filename)):
            #print(line)
            #exit()
            features_tag=json.loads(line.strip())
            #print(features_tag[2:-1])
            #exit()
            raw_data.append(features_tag[0])
            tag=features_tag[-1]
            tags.append(tag)
            for fea,feas in zip(features_tag[3:-1],featres_list):
                feas.append(fea)
        for feaname,fea in zip(self._place_holders,featres_list):
            fee_dic[feaname]=fea
        fee_dic[self._tag]=self.toonehot(tags)
        fee_dic[self._dropout]=self._drop_out
        fee_dic['raw_data'] = raw_data
        #feed_dict.pop('raw_data') # 删除字典中对应的键值对，如果键不存在，返回错误print(dict4)
        #code_path = 'pre_code/mlp/bianma_test.txt' #对比学习经过MLP降到二维
        #data2bianma = self.load_bianma(code_path) 
        #bianma_lst = []
        #for x in raw_data:
        #    y2 = data2bianma[x]
         #   bianma_lst.append(y2)
        #self.bianma_lst = tf.convert_to_tensor(bianma_lst, dtype='float32')

        #print "load suceed"
        #fee_dic[self.bianma_lst] = bianma_lst
        return fee_dic,self.toonehot(tags)




    def cal(self,pre,rea):
        total0_in_pre=1
        total0_in_real=1
        total1_in_pre=1
        total1_in_real=1
        right1=1
        right0=1
        zz = 0
        cc = 0
        for p,r in zip(pre,rea):
            rr=list(r).index(max(list(r)))
            pp=list(p).index(max(list(p)))
            #print('=========================')
            #print rr,"真实标签"
            #print pp,"预测标签"
            if rr ==pp:
                zz+=1
            if rr!=pp:
                cc+=1
            if rr==0:
                total0_in_real +=1
                if pp==0:
                    right0 +=1   
            if rr==1:
                total1_in_real +=1
                if pp==1:
                    right1 +=1
            if pp==0:
                total0_in_pre +=1
            if pp==1:
                total1_in_pre +=1
        print('============')
        print "正确数量",zz 
        print "错误个数",cc
        recall0=float(right0)/total0_in_real
        pre0=float(right0)/total0_in_pre
        f10=2*recall0*pre0/(pre0+recall0)
        recall1=float(right1)/total1_in_real
        pre1=float(right1)/total1_in_pre
        f11=2*recall1*pre1/(pre1+recall1)
        
        if total0_in_real<total1_in_real:
            total_in_real = total0_in_real-1
            total_in_pre=total0_in_pre-1
            right=right0-1 
        else:
            total_in_real = total1_in_real-1
            total_in_pre=total1_in_pre-1
            right=right1-1
        num=[right,total_in_pre,total_in_real]
        return recall0,pre0,f10,recall1,pre1,f11,num
