#!/usr/bin/python
#-*-coding:utf-8-*-
import os,sys
import config
import model.LR2 as LR
import tensorflow as tf
import glob
import pickle
import generate_epoch as GE
import gpu_get as  gpu
import random
import numpy as np
sys.path.append("../../../../")

gpu_num=gpu.GPU(0)
#print(gpu_num)
#exit()
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
flags = tf.app.flags
#top config
flags.DEFINE_string("device",       "/gpu:%s"%os.environ["CUDA_VISIBLE_DEVICES"],       "running device")
#run config
flags.DEFINE_boolean("train",        False,      "training the networks")
flags.DEFINE_boolean("saveBoard",   False,      "save tensorboard or not")
flags.DEFINE_boolean("savePB",      False,      "save pb file or not")
flags.DEFINE_boolean("test",        False,      "test the networks")
flags.DEFINE_string("testfile",    "file"    ,      "training the networks")
flags.DEFINE_string("extra_test1",    "file1"    ,      "training the networks")
flags.DEFINE_string("extra_test2",    "file2"    ,      "training the networks")
flags.DEFINE_boolean("new_ckpt",    False    ,      "rm ckpt")


#train config
FLAGS = tf.app.flags.FLAGS
if FLAGS.new_ckpt:
    os.system("rm ckpt/*")
with tf.device(FLAGS.device):
    with tf.Graph().as_default():
        with tf.Session(config=config.TF_CONFIG) as sess:
            model = LR.model(sess,config)
            config.logging.info("in model now")
            if FLAGS.saveBoard:
                tf.summary.FileWriter("./tensorboard",sess.graph)
            if FLAGS.savePB:
                model.savePB()
            if FLAGS.train:
                test_names=[]
                if FLAGS.extra_test1!="file1":
                    print(FLAGS.extra_test1)
                    test_names.append(FLAGS.extra_test1)
                if FLAGS.extra_test2!="file2":
                    print(FLAGS.extra_test2)
                    test_names.append(FLAGS.extra_test2)
                config.logging.info("train")
                for i in range(config.TRAIN_CONFIG["max_epoch"]):
                    GE.get_epoch_file(batch_size=config.DATA_CONFIG["batch_size"])
                    #exit()
                    batch_name=glob.glob("data/batch/*")
                    
                    model.train(i,test_names,batch_name)
                #####################################
                pre_train,pre_test=model.getfinal()
                data_info_path=glob.glob("../../model_result_info.pkl")
                if data_info_path !=[]:
                    with open(data_info_path[0],"r")as f:
                        model_result_info=pickle.load(f)
                else:
                    model_result_info={}
                model_result_info["cubt"]={"train":round(pre_train,4),"test":round(pre_test,4)}
                with open("../../model_result_info.pkl","w")as f:
                    pickle.dump(model_result_info,f)
    ###########################################
            if FLAGS.test:
                config.logging.info("test")
                test_names=[]
                if FLAGS.extra_test1!="file1":
                    print(FLAGS.extra_test1)
                    test_names.append(FLAGS.extra_test1)
                if FLAGS.extra_test2!="file2":
                    print(FLAGS.extra_test2)
                    test_names.append(FLAGS.extra_test2)
                #filenames=glob.glob(FLAGS.testfile)
                model.test(test_names,True)

