'''步骤1、构建相关数据结构集储存上述相互作用对。
'''
#步骤1---------
import numpy as np
import pandas as pd
import time

info_rat1 =  pd.read_table(r'C:\\Users\\john\\Desktop\\csv\\net1.csv',sep=',')#rat 上游基因到kcnam1三层基因
info_rat2 =  pd.read_table(r'C:\\Users\\john\\Desktop\\csv\\net2.csv',sep=',')#rat 上游基因到kcnam1四层基因

info_homo =  pd.read_table(r'C:\\Users\\john\\Desktop\\csv\\net3.csv',sep=',')#homo 上游基因到kcnam1四层基因
info_mice1 =  pd.read_table(r'C:\\Users\\john\\Desktop\\csv\\net4.csv',sep=',')#mice 上游基因到kcnam1三层基因
info_mice2 =  pd.read_table(r'C:\\Users\\john\\Desktop\\csv\\net5.csv',sep=',')#mice 上游基因到kcnam1四层基因
#------创建键值对列表
#--rat
pairlist1 = {(x,i) for (x,i) in zip(info_rat1.list1,info_rat1.list2)}
pairlist2 = {(x,i) for (x,i) in zip(info_rat1.list2,info_rat1.list3)}
pairlist3 = {(x,i) for (x,i) in zip(info_rat2.list1,info_rat2.list2)}
pairlist4 = {(x,i) for (x,i) in zip(info_rat2.list2,info_rat2.list3)}
pairlist5 = {(x,i) for (x,i) in zip(info_rat2.list3,info_rat2.list4)}
#--homo
pairlist6 = {(x,i) for (x,i) in zip(info_homo.list1,info_homo.list2)}
pairlist7 = {(x,i) for (x,i) in zip(info_homo.list2,info_homo.list3)}
pairlist8 = {(x,i) for (x,i) in zip(info_homo.list3,info_homo.list4)}
#--mice
pairlist9  = {(x,i) for (x,i) in zip(info_mice1.list1,info_mice1.list2)}
pairlist10 = {(x,i) for (x,i) in zip(info_mice1.list2,info_mice1.list3)}

pairlist11 = {(x,i) for (x,i) in zip(info_mice2.list1,info_mice2.list2)}
pairlist12 = {(x,i) for (x,i) in zip(info_mice2.list2,info_mice2.list3)}
pairlist13 = {(x,i) for (x,i) in zip(info_mice2.list3,info_mice2.list4)}
key_value = pairlist1|pairlist2|pairlist3|pairlist4|pairlist5|pairlist6|pairlist7\
    |pairlist8|pairlist9|pairlist10|pairlist11|pairlist12|pairlist13  #构造键值对集合
print("number of key_value:",len(key_value))
#------创建基因列表
genelist = []
for i in key_value:
    genelist.append(i[0])
    genelist.append(i[1])
print('number of gene:',len(genelist))
genelist = set(genelist)
print('number of gene:',len(genelist))
genedict = list(enumerate(genelist)) #构造基因序列索引
genedict = dict(genedict)
#print(genedict)
'''步骤2：构建初始距离矩阵，代表每个基因与基因之间的距离。
  '-'代表无直接相互作用关系。
'''
#步骤2：----------------------------
initial_array = np.empty([373,373],dtype=str)
print(initial_array)
INFINITY = '-'
for x in range(373):
    for y in  range(373):
        if x == y:
            initial_array[x][y]=0
        else:
            if(genedict[x],genedict[y]) in key_value or (genedict[y],genedict[x]) in key_value:
                initial_array[x][y]=1
            else:initial_array[x][y]=INFINITY
#'''步骤三执行Floyd算法，更新距离矩阵，表示每个基因到每个基因之间的最短距离'''
#步骤3---------------------------------
def addWithInfinity(a,b):                     #由于存在无穷大值INFINITY，并且整个矩阵为字符矩阵,故对加法运算做调整
    if a == INFINITY or b == INFINITY:
        return INFINITY
    else: return int(a)+int(b)
def minWithInfinity(a,b):                     #同理对min()函数进行调整
    if a == INFINITY and b ==INFINITY:
        return INFINITY
    elif a == INFINITY and b != INFINITY:
        return b
    elif a != INFINITY and b == INFINITY:
        return a
    else :return min(int(a),int(b))
#-------计算ing
start = time.time()         #时间记录
for i in range(373):
    for r in range(373):
        for c in range(373):
             initial_array[r][c] = minWithInfinity(initial_array[r][c],
             addWithInfinity(initial_array[r][i],initial_array[i][c]))
distance_array = initial_array.astype(np.int16)
elapsed = time.time()-start
dt = pd.DataFrame(distance_array)
print(dt.head())
print(dt.tail())
#'''步骤4：算出每个基因到其他所有基因的距离之和，最小者即为hub基因'''
#步骤4---------------------
sum1 = dt.sum()
temp = {'value':sum1,'gene_id':list(genedict.values())}
sum1= pd.DataFrame(temp)
sum_sort = sum1.sort_values(by=['value','gene_id'])
print(sum_sort.head())#显示头五行
print('mean:',sum_sort.value.mean())
print('median:',sum_sort.value.median())
print('std:',sum_sort.value.std())
print('time elapsed:',elapsed)
#----output:
'''
number of key_value: 870
number of gene: 1740
number of gene: 373

    value  gene_id
248    740    50522
213    816     3191
130    884     6915
11     886    24654
99     897     4609
mean: 1137.3941018766757  #均值距离
median: 1107.0            #中位数距离
std: 129.82975895072994   #方差
time elapsed: 209.9327802658081   #373个基因规模运行的时间
'''
#kcnam1，基因跨三个物种ratid：83731，homoid：3778 miceid：16531
#从上游到kcnam1，共计373个基因，870对相互作用对，hub基因前五名如上表所示，geneid：50522，3191，6915，24654，2609


#后记：此次算法共计的373个基因是200个上游基因到kcnam1总计的，最短距离算法也是围绕着这373个基因进行的
#但是有很多基因不在这373个基因里面的，但和这373个基因依然存在千丝万缕的相互作用关系，理应也计算在内，
#本次实验只计算了上游基因 能跑到 kcnam1之前基因的相互作用网络，故本次实验是精简了原始的网络，所得hub基因也是狭义上的hub基因。
#若想从原始网络计算hub基因又经过kcnam1，可以对部分基因加权计算，凸显kcnam1的重要性。