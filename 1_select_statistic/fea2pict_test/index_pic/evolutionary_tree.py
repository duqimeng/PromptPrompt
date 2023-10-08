


def find_fea(seq,fea_lst):
    new_seq = "-" * len(seq)  # 初始化新序列，全部使用-代替
    for feature in fea_lst:
        index = seq.find(feature)  # 找到特征在序列中的位置
        if index != -1:  # 如果特征存在于序列中
            new_seq = new_seq[:index] + feature + new_seq[index+len(feature):]  # 将特征替换进新序列中
    return new_seq







if __name__ == '__main__':
    with open('to_evolutionary.txt','w',encoding = 'utf-8')as f:
        for line in open('结果/result_11.txt','r'):
            seq,fea_lst = line.strip().split('\t')
            fea_lst = eval(fea_lst)
            new_seq = find_fea(seq,fea_lst)
            f.write(new_seq + '\n')