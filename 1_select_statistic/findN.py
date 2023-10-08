a={}
for line in open('B100.txt','r',encoding='utf-8'):
    data,label = line.strip().split('\t')
    for i in data:
        if i == 'N' or i == 'Y' or i == 'w' or i == 'R':
            a[line]=i
print(a)
