#coding=utf-8
import pandas as pd
import jieba
import numpy as np
raw=pd.read_csv("train.csv")
content_id=raw["content_id"]
content=raw["content"]
subject=raw["subject"]
subject_dic_raw={'颜色': 5, '配置': 3, '空调': 9, '装甲': 6, '舒适性': 9, '油耗': 7, '导航': 3, '獠牙': 5, '车漆': 5, '材料': 2, '外观': 5, '刹车盘': 4, '车身': 5, '空间': 8, '动力': 10, '价格': 1, '异响': 9, '内饰': 2, '加速': 10, '后备箱': 8, '尾灯': 5, '影像': 3, '全时': 6, '刹车油': 4, '中控': 3, '好看': 5, '轴承': 3, '省油': 7, '风噪': 9, '刹车': 4, '操控性': 6, '安卓': 3, '操控': 6, '雷达': 3, '方向盘': 6, '做工': 2, '变速箱': 10, '刹车片': 4, 'abs': 4, '噪音': 9, '消耗': 10, '安全性': 4, '前脸': 5, '发动机': 10, '车价': 1, '优惠': 1, '气囊': 4, '性能': 6, '刹车灯': 4,
     '座椅': 2,  '平均': 7, '声音': 9,  '底盘': 6,  '后排': 8,}

def get_accurancy(dic):
    subject_contains = []
    for i in range(len(content)):
        subject_contains.append(np.zeros(11,dtype=np.int).tolist())
        for sub in dic:
            num=content.values[i].count(sub)
            subject_contains[i][dic[sub]]+=num
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    i = 0
    while (i < len(content)):
        k = 0
        while (i + k < len(content) and content_id[i] == content_id[i + k]):
            if (subject[i+k]in dic and subject_contains[i][dic[subject[i + k]]] > 0):
                TP += 1
                subject_contains[i][dic[subject[i + k]]] = 0
            else:
                FN += 1
            k += 1
        for j in range(11):
            if (subject_contains[i][j] != 0):
                FP += 1
        i+=k+1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    return F1
has_better=True
result_dic=subject_dic_raw.copy()
while(has_better):
    has_better=False
    standard=get_accurancy(result_dic)
    for item in result_dic:
        subject_dic = result_dic.copy()
        subject_dic.pop(item)
        accurancy=get_accurancy(subject_dic)
        if(accurancy>standard):
            result_dic.pop(item)
            has_better=True
            break
    print(result_dic)
    print(accurancy,standard)
