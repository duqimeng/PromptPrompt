 #!/bin/sh
rm result/*
rm statistic/*
rm statistic_2/*
rm classify_data/*

echo "now start"
python 1getLabelCharUniBiTri2cnt.py --file train_data_1.txt
python 2getClassify_data.py  --file train_data_1.txt
python 3getTF.py
python 3.5getTF2.py


python 4getFeatureLimit.py
python 5getXY.py
echo "finished"

echo "now start"
rm classify_data/*
python 2getClassify_data.py  --file test_data_1.txt --which "test"
python 5getXY.py  --which "test"
echo "finished"

echo "now sending data"
mv result/*2id  ../LR_model/data/dict_info/
mv result/data_info.pkl ../LR_model/data/dict_info/
mv result/xy_t* ../LR_model/data

echo "done"   


 