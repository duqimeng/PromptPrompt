


non-promoter:非启动子序列数据
promoter:启动子序列数据

1_select_statistic.py ：根据all_stastic文件夹中的数据筛选特征，在result_1文件夹中得到筛选结果

2_findbox.py：根据上一步到的结果（也就是result_1）中的文件，到得到box_result.txt这个文件

3_get_result.py：根据得到的box_result.txt中的特征对序列进行分类（代码中会有阈值的设定）