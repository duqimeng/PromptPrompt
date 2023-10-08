
#启动子路径：
Promoter_path = '../1_select_statistic/promoter'               #此物种启动子文件路径
non_Promoter_path = '../1_select_statistic/non-promoter'       #此物种非启动子文件路径


fea_0 = 'result/picture_0/'           #特征所在非启动子序列的位置
fea_1 = 'result/picture_1/'           #特征所在启动子序列的位置




#result文件夹中存放绘制出的图片：横坐标只0~81，纵坐标导表该特征在该位置出现的次数。
#index_pic文件夹中存放需要使用的代码，其中select_pic_50文件夹中存放的是从result文件夹中挑选出的特征，这些特征应该存在共性，比如他们都在同一位置出现的概率避灾其他位置出现的概率大