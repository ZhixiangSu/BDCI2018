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
    {'优惠': 1, '车价': 1, '獠牙': 5, '后备箱': 8, '方向盘': 6, '车身': 5, '外观': 5, '做工': 2, '安全性': 4, '刹车灯': 4, '轴承': 3, '全时': 6,
     '安卓': 3, '雷达': 3, '加速': 10, '价格': 1, '空间': 8, '省油': 7, '刹车盘': 4, '影像': 3, '风噪': 9, '操控': 6, '座椅': 2, '动力': 10,
     '舒适性': 9, '好看': 5, '异响': 9, '消耗': 10, '变速箱': 10, '尾灯': 5, '材料': 2, '油耗': 7, '底盘': 6, '发动机': 10, '车漆': 5, '刹车片': 4,
     '导航': 3, 'abs': 4, '内饰': 2, '颜色': 5, '中控': 3, '刹车': 4, '操控性': 6, '配置': 3, '噪音': 9, '气囊': 4, '空调': 9, '性能': 6,
     '前脸': 5, '刹车油': 4}
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
for i in range(len(content)):
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
P=TP/(TP+FP)
R=TP/(TP+FN)
F1=2*P*R/(P+R)
print(P,R)
print(F1)
