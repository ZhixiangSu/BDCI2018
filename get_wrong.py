import pandas as pd
import jieba
import numpy as np
raw=pd.read_csv("train.csv")
content_id=raw["content_id"]
content=raw["content"]
subject=raw["subject"]
subject_dic={'优惠': 1, '车价': 1, '獠牙': 5, '后备箱': 8, '方向盘': 6, '车身': 5, '外观': 5, '做工': 2, '安全性': 4, '刹车灯': 4, '轴承': 3, '全时': 6,
     '安卓': 3, '雷达': 3, '加速': 10, '价格': 1, '空间': 8, '省油': 7, '刹车盘': 4, '影像': 3, '风噪': 9, '操控': 6, '座椅': 2, '动力': 10,
     '舒适性': 9, '好看': 5, '异响': 9, '消耗': 10, '变速箱': 10, '尾灯': 5, '材料': 2, '油耗': 7, '底盘': 6, '发动机': 10, '车漆': 5, '刹车片': 4,
     '导航': 3, 'abs': 4, '内饰': 2, '颜色': 5, '中控': 3, '刹车': 4, '操控性': 6, '配置': 3, '噪音': 9, '气囊': 4, '空调': 9, '性能': 6,
     '前脸': 5, '刹车油': 4}
subject_contains=[]
for i in range(len(content)):
    subject_contains.append(np.zeros(len(subject_dic)+1,dtype=np.int).tolist())
    for sub in subject_dic:
        num=content.values[i].count(sub)
        subject_contains[i][subject_dic[sub]]+=num
count=0
wrong_content=[]
wrong_subject=[]
for i in range(len(content)):
    k=0
    has=0
    for j in range(11):
        if(subject_contains[i][j]>0):
            k=0
            while (i+k<len(content) and content_id[i + k] == content_id[i]):
                if (subject_dic[subject[i + k]] == j):
                    has=1
                k += 1
    if(has==0):
        for m in range(k):
            wrong_subject.append(subject.values[i+m])
            wrong_content.append(content.values[i+m])
    i+=k
'''
for i in range(len(subject_contains)):
    subject_contains[i].index(max(subject_contains[i])) == subject_dic[subject[i]]
    if  subject_contains[i].index(max(subject_contains[i]))!=subject_dic[subject[i]]:
        wrong_subject.append(subject.values[i])
        wrong_content.append(content.values[i])
'''
wrong_content=pd.DataFrame(wrong_content)
wrong_subject=pd.DataFrame(wrong_subject)
wrong=pd.concat([wrong_content,wrong_subject],axis=1)

writer=pd.ExcelWriter("wrong.xlsx")
wrong.to_excel(writer)
writer.close()