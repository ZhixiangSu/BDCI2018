import pandas as pd
import jieba
import numpy as np
raw=pd.read_csv("train.csv")
content=raw["content"]
subject=raw["subject"]
subject_dic={"价格":1,"内饰":2,"配置":3,"安全性":4,"外观":5,"操控":6,"油耗":7,"空间":8,"舒适性":9,"动力":10,
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