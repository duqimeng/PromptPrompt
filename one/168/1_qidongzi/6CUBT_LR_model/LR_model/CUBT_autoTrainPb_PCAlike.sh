echo "now train"
python main.py --train True --extra_test1 data/xy_test --new_ckpt True
#grep -e 'test' log_* >> log/validate_log
#grep -e 'train' log_* >> log/train_log

#echo "now saving pb"
#python main.py --savePB True

rm *.pyc


#echo "now sending id files data info file and pb file to Flusk"
#cp data/dict_info/*2id ../FluskSerVice/dict/
#cp data/dict_info/data_info.pkl ../FluskSerVice/dict/
#cp model.pb ../FluskSerVice/dict/
#echo "done"

