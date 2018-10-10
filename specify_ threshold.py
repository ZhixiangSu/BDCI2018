#coding=gbk
import pandas as pd
import torch
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf=pd.read_csv("tf-idf.csv",encoding="gbk")
raw=pd.read_csv("train.csv")
content_id=raw["content_id"]
content=raw["content"]
subject=raw["subject"]
subject_dic={"�۸�":0,"����":1,"����":2,"��ȫ��":3,"���":4,"�ٿ�":5,"�ͺ�":6,"�ռ�":7,"������":8,"����":9}
standard_sub=["�۸�","����","����","��ȫ��","���","�ٿ�","�ͺ�","�ռ�","������","����"]
def get_accuracy(threshold=0.19):
    subject_contains = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    i = 0
    while(i<len(content)):
        k=0
        while(i+k<len(content) and content_id[i+k]==content_id[i]):
            if(tf_idf.values[i][subject_dic[subject.values[i+k]]]>threshold):
                TP+=1
            else:
                FN+=1
            tf_idf.values[i][subject_dic[subject.values[i]]]=0
            k+=1
        for j in range(len(standard_sub)):
            if(tf_idf.values[i][j]>threshold):
                FP+=1
        i+=k
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    return F1,FN,FP

for i in range(len(tf_idf)):
    sum = 0
    for j in range(len(tf_idf.values[i])):
        sum += tf_idf.values[i][j]
    for j in range(len(tf_idf.values[i])):
        tf_idf.values[i][j]/=sum