import pandas as pd
import jieba
import numpy as np
raw=pd.read_csv("train.csv")
content=raw["content"]
subject=raw["subject"]
subject_dic={'刹车': 4, '空间': 8, '噪音': 9, '刹车盘': 4, '风噪': 9, '尾灯': 5, '发动机': 10, '舒适性': 9, 'abs': 4, '刹车灯': 4, '影像': 3, '油耗': 7, '轴承': 3, '全时': 6, '前脸': 5, '安卓': 3, '好看': 5, '动力': 10, '配置': 3, '车漆': 5, '安全性': 4, '消耗': 10, '内饰': 2, '车身': 5, '操控': 6, '气囊': 4, '空调': 9, '加速': 10, '做工': 2, '外观': 5, '异响': 9, '后备箱': 8, '刹车片': 4, '价格': 1, '车价': 1, '材料': 2, '中控': 3, '方向盘': 6, '导航': 3, '优惠': 1, '座椅': 2, '雷达': 3, '底盘': 6, '操控性': 6, '刹车油': 4, '省油': 7, '变速箱': 10, '颜色': 5}
subject_contains=[]
for i in range(len(content)):
    subject_contains.append(np.zeros(len(subject_dic)+1,dtype=np.int).tolist())
    for sub in subject_dic:
        num=content.values[i].count(sub)
        subject_contains[i][subject_dic[sub]]+=num
count=0
unknown_content=[]
unknown_subject=[]
for i in range(len(subject_contains)):
    subject_contains[i].index(max(subject_contains[i])) == subject_dic[subject[i]]
    if subject_contains[i].index(max(subject_contains[i]))==0 :
        unknown_subject.append(subject.values[i])
        unknown_content.append(content.values[i])
unknown_content=pd.DataFrame(unknown_content)
unknown_subject=pd.DataFrame(unknown_subject)
unknown=pd.concat([unknown_content,unknown_subject],axis=1)

writer=pd.ExcelWriter("unknown.xlsx")
unknown.to_excel(writer)
writer.close()