


def flip_label(label):
    if label == '0':
        return '1'
    elif label == '1':
        return '0'
    else:
        return label
        
        
        
data2label = {}
raw2label = {}
iiii = 0
for line in open('cubt_predicts_result'):
    data,pre,rea = line.strip().split('\t')
    data2label[data] = rea


for line in open('../../Data/test_data_1.txt'): 
    data,rea = line.strip().split('\t')
    # pre = flip_label(rea)
    raw2label[data] = rea


ii = 0
for i in raw2label.keys():
    if data2label[i] == raw2label[i]:
        ii+=1
print(ii)
exit()
    # print(i)
    # exit()
print(len(data2label))
print(len(raw2label))
    
    
    
    # rea = flip_label(rea)
    # all_+=1
    # if pre == rea:
        # i+=1
        # print('0')
# print(all_)
# print(i)