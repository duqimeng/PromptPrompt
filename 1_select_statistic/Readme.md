


non-promoter:non-promoter sequence data
promoter: promoter data.

1_select_statistic.py: filter features according to the data in all_stastic folder, and get the filter result in result_1 folder.

2_findbox.py: according to the results of the previous step (that is, result_1) in the file, to get box_result.txt this file

3_get_result.py: according to get box_result.txt in the characteristics of the sequence for classification (the code will have a threshold set)


**
The 1_select_statistic folder holds the data analysis code that provides the basis for the creation of the prediction tool
The 1_select_statistic/fea2pict_test folder is where the location information features are generated and utilized. Run the fea2pict.py file and the results are stored in the results folder, the path can be set under config. In the 'result/picture_1/' path to select the picture features that meet the requirements, stored in the index_pic/select_pic_50 file and folder under the run get_result.py (non-promoter and promoter file to store the promoter and non-promoter data respectively, this is an example)**
