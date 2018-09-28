#coding=utf-8

import pandas as pd

import jieba
import numpy as np
raw=pd.read_csv("train.csv")
subject_dic_10={"价格":1,"内饰":2,"配置":3,"安全性":4,"外观":5,"操控":6,"油耗":7,"空间":8,"舒适性":9,"动力":10,
              "优惠": 1,   "便宜": 1,
              "座椅": 2,  "做工": 2,
             "导航": 3,  "手机": 3,  "中控": 3,
             "刹车": 4, "刹车片": 4, "安全": 4, "刹车盘": 4,  "问题": 4,
               "好看": 5, "喜欢": 5, "车身": 5, "颜色": 5,
              "底盘": 6, "四驱": 6, "方向盘": 6,  "性能": 6,  "感觉": 6, "全时": 6,
              "高速": 7, "公里": 7, "平均": 7,  "市区": 7, "左右": 7, "10": 7, "机油": 7, "省油": 7,
              "后备箱": 8, "后排": 8, "xv": 8,
             "空调": 9, "噪音": 9,  "声音": 9,
             "发动机": 10,  "变速箱": 10,
             }
subject_dic_5={"价格":1,"内饰":2,"配置":3,"安全性":4,"外观":5,"操控":6,"油耗":7,"空间":8,"舒适性":9,"动力":10,
                "优惠":1,
               "座椅":2,
                "导航":3,"手机":3,
                "刹车":4,"刹车片":4,"安全":4,
                "好看":5,
                "底盘":6,"四驱":6,"方向盘":6,
                "高速":7,"公里":7,"平均":7,
                "后备箱":8,"后排":8,"xv":8,
                "空调":9,"噪音":9,"声音":9,
                "发动机":10,"机油":10,"变速箱":10}
subject_dic= \
    {'刹车': 4, '空间': 8, '噪音': 9, '刹车盘': 4, '风噪': 9, '尾灯': 5, '发动机': 10, '舒适性': 9, 'abs': 4, '刹车灯': 4, '影像': 3, '油耗': 7,
     '轴承': 3, '全时': 6, '前脸': 5, '安卓': 3, '好看': 5, '动力': 10, '配置': 3, '车漆': 5, '安全性': 4, '消耗': 10, '内饰': 2, '车身': 5,
     '操控': 6, '气囊': 4, '空调': 9, '加速': 10, '做工': 2, '外观': 5, '异响': 9, '后备箱': 8, '刹车片': 4, '价格': 1, '车价': 1, '材料': 2,
     '中控': 3, '方向盘': 6, '导航': 3, '优惠': 1, '座椅': 2, '雷达': 3, '底盘': 6, '操控性': 6, '刹车油': 4, '省油': 7, '变速箱': 10, '颜色': 5}
standard_sub={None:0,"价格":1,"内饰":2,"配置":3,"安全性":4,"外观":5,"操控":6,"油耗":7,"空间":8,"舒适性":9,"动力":10,}
content_id=raw["content_id"]
content=raw["content"]
subject=raw["subject"]
subject_contains=[]
for i in range(len(content)):
    subject_contains.append(np.zeros(11,dtype=np.int).tolist())
    for sub in subject_dic:
        num=content.values[i].count(sub)
        subject_contains[i][subject_dic[sub]]+=num
result=[]
count=0
TP=0
TN=0
FP=0
FN=0
i=0
while(i<len(content)):
    k=0
    while(i+k<len(content) and content_id[i]==content_id[i+k]):
        if(subject_contains[i][subject_dic[subject[i+k]]]>0):
            TP+=1
            subject_contains[i][subject_dic[subject[i + k]]]=0
        else:
            FN+=1
        k+=1
    for j in range(11):
        if(subject_contains[i][j]!=0):
            FP+=1
    i+=k+1
P=TP/(TP+FP)
R=TP/(TP+FN)
F1=2*P*R/(P+R)
print(P,R)
print(F1)
