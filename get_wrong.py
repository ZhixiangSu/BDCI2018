import pandas as pd
import jieba
import numpy as np
raw=pd.read_csv("train.csv")
content_id=raw["content_id"]
content=raw["content"]
subject=raw["subject"]
subject_dic={'刹车': 4, '空间': 8, '噪音': 9, '刹车盘': 4, '风噪': 9, '尾灯': 5, '发动机': 10, '舒适性': 9, 'abs': 4, '刹车灯': 4, '影像': 3, '油耗': 7, '轴承': 3, '全时': 6, '前脸': 5, '安卓': 3, '好看': 5, '动力': 10, '配置': 3, '车漆': 5, '安全性': 4, '消耗': 10, '内饰': 2, '车身': 5, '操控': 6, '气囊': 4, '空调': 9, '加速': 10, '做工': 2, '外观': 5, '异响': 9, '后备箱': 8, '刹车片': 4, '价格': 1, '车价': 1, '材料': 2, '中控': 3, '方向盘': 6, '导航': 3, '优惠': 1, '座椅': 2, '雷达': 3, '底盘': 6, '操控性': 6, '刹车油': 4, '省油': 7, '变速箱': 10, '颜色': 5}
standard_sub=["None","价格","内饰","配置","安全性","外观","操控","油耗","空间","舒适性","动力"]
subject_contains=[]
for i in range(len(content)):
    subject_contains.append(np.zeros(len(subject_dic)+1,dtype=np.int).tolist())
    for sub in subject_dic:
        num=content.values[i].count(sub)
        subject_contains[i][subject_dic[sub]]+=num
count=0
FN=[]
FP=[]
i=0
while(i<len(content)):
    k=0
    while(i+k<len(content) and content_id[i]==content_id[i+k]):
        if(subject_contains[i][subject_dic[subject[i+k]]]>0):
            subject_contains[i][subject_dic[subject[i + k]]]=0
        else:
            FN.append([content[i+k],subject[i+k]])
        k+=1
    for j in range(11):
        if(subject_contains[i][j]!=0):
            FP.append([content[i], standard_sub[j]])
    i+=k+1
'''
for i in range(len(subject_contains)):
    subject_contains[i].index(max(subject_contains[i])) == subject_dic[subject[i]]
    if  subject_contains[i].index(max(subject_contains[i]))!=subject_dic[subject[i]]:
        wrong_subject.append(subject.values[i])
        wrong_content.append(content.values[i])
'''
FN=pd.DataFrame(FN)
FP=pd.DataFrame(FP)

writer=pd.ExcelWriter("wrong.xlsx")
FN.to_excel(writer,sheet_name="漏判")
FP.to_excel(writer,sheet_name="多判")
writer.close()