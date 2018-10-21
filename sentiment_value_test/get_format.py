# coding=gbk
import pandas as pd
import torch
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
raw = pd.read_csv("../train.csv",encoding="UTF-8")
subject_dic = {"�۸�": 1, "����": 2, "����": 3, "��ȫ��": 4, "���": 5, "�ٿ�": 6, "�ͺ�": 7, "�ռ�": 8, "������": 9, "����": 10}
standard_sub = ["�۸�", "����", "����", "��ȫ��", "���", "�ٿ�", "�ͺ�", "�ռ�", "������", "����"]
key_words=["��ͥ","��ϲ","����","���°�","����","���","�Ϳ�","����","����","5000","ɲ��","����","ɲ����","��ȫ��","100","����","����","����","��","̫С","�Ż�","ͼ��","��׿","����","���","����","17","��·","2019","ѭ��","����","����","��ɲ","ɲ����","�Ƽ�","����","cx5","�۸�","����","ָ����","����","����","cd","�����","Ӱ��","�յ�","��β","ʵ��","����","ά��","gps","����","80","����","ƽ���ٶ�","����","��ɲ��","����","����","�ļ�","����","Ϩ��","����","����","�ٿ���","����","һ��Ǯ","����","����","����","�ÿ�","���","ζ��","����","����","���","����","��ײ","����","����","ë��","����","����","����","����","����","֧��","����","�ٿ�","����","����","����","�ѿ�","����","ʵʱ","����","����","������","����","��Ʒ","��Ǯ","ͣ��","ȫʱ","����","����","����","����","����","����","��˰","��ɫ","�ͱ�","����","¥��","�㹻","ָ��","14","����","�˱�","����","ɲ��Ƭ","����","���","����","�ܴ�","·��","����","ԭ��","��ó","����","����","����","�仯","����","����","�Լ۱�","��ɫ","�ر�","abs","��̼","fb","�ƶ�","����","2w","�ٹ���","�ĸ�","11","��","����","�㳵","����","����","����","40","�ռ�","��;","�Ӵ�","��Ӽ�","����","ɲ����","�״�","����","����","�綯","����","���ʦ","����","����","���Ӿ�","���޶�","��ϵ","����","����","����","����","Ư��","�ߵ�","��װ","12","�˴�","ǰ��","ѹ����","ȫ��","ej","�γ�","����","�ڲ��ռ�","ʵ��","����","����","����","̥ѹ","�ۺ�","����","�ٶ�","����","�ŵ�","����","������","�п�","β��","gla","����","30","���","ȫϵ","����","����","13","ÿ��","ʡ��","Ч��","©��","��λ","��ͼ","ϸ��","�ͺ�","ϣ��","��û��","��ϴ","0w20","����","���","����","ɽկ","�Ͽ�","��λ","��ʨ","����","����","��ǯ","����","����","����","���","es","����","���","�ƾ�","˲ʱ","usb","�Ƕ�","������","q5","16","������","��ѩ","β��","����","����","С��","����","����","����","����","���","wd40","����","������","������","����","�촰","����","������","��Ƥ","�Ƹ�","ȷʵ","ԭ��","19","���","һ��","����","20","����ͷ","������","����","��Ұ","���","����","cvt","����","����","ӥ��","����","ǰ��","ֻ��","��̥","Ӱ��","����","������","û��","�Ȱ�","����","rav4","����","����","����","���","����","cx","����","������","�۸����","����","ɲ��̤��","����","����","�濥","xt","�³�","��ɫ","����","����","����","Խ��Խ","����","������","���","����","����","����","����","����","dixcel","����","�ֽ�",
]
temp=pd.read_csv("../tf_idf.csv",encoding="gbk")
words_tfidf = temp[key_words]
values = [[] for i in range(len(raw))]
sp_content=[[] for i in range(len(raw))]
sp_word=[]
for i in range(len(raw)):
    for j in range(1, len(subject_dic) + 1):
        v = 0
        max_tfidf=0
        max_word=""
        words = jieba.lcut(raw["content"].values[i])
        for word in words:
            if (word in words_tfidf.columns.tolist()):
                if(words_tfidf[word].values[j]>max_tfidf):
                    max_tfidf=words_tfidf[word].values[j]
                    max_word=word
                v += words_tfidf[word].values[j]
        values[i].append(v)
        sentences=re.split("[��������?]",raw["content"].values[i])
        sp_word.append(max_word)
        for sentence in sentences:
            if(sentence.count(max_word)>0):
                sp_content[i].append(sentence)
                break
values = pd.DataFrame(values)
sp_content=pd.DataFrame(sp_content)
values.columns = standard_sub
# values["subject"]=raw["subject"]
tf_idf=values
for i in range(len(tf_idf)):
    sum = 0
    for j in range(len(tf_idf.values[i])):
        sum += tf_idf.values[i][j]
    for j in range(len(tf_idf.values[i])):
        tf_idf.values[i][j] /= sum
threshold=0.19
result_tfidf=[]
for i in range(len(tf_idf)):
    if(i>0 and raw["content_id"][i]==raw["content_id"][i-1]):
        continue
    has=False
    for j in range(len(standard_sub)):
        if (tf_idf.values[i][j] > threshold):
            has=True
            result_tfidf.append([raw["content_id"].values[i],sp_content.values[i][j],standard_sub[j],raw["sentiment_value"].values[i],None,sp_word[i]])
    if(has==False):
        result_tfidf.append([raw["content_id"].values[i],sp_content.values[i][0],standard_sub[0], raw["sentiment_value"].values[i], None,sp_word[i]])
result_tfidf=pd.DataFrame(result_tfidf)
result_tfidf.columns=["content_id","content","subject","sentiment_value","sentiment_word","sp_word"]
result_tfidf["sentiment_value"]=result_tfidf["sentiment_value"].astype(int)
result_tfidf.to_csv("result_tfidf1.csv",encoding="UTF-8",index=False)