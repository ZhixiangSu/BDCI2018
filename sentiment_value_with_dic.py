import re
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import jieba.posseg as pseg
import jieba
raw=pd.read_csv("train.csv")
result=pd.read_csv("./commit/result.csv")
content=raw['content']
subject=raw['subject']
positive_words=open("positive_words.txt",encoding="UTF-8").read().split("\n")
negative_words=open("negative_words.txt",encoding="UTF-8").read().split("\n")
split_content=[]
sentiment_value=[]
for i in range(len(content)):
    split_content.append(re.split("[，。]",content[i]))
    sentiment_value.append(0)
    for j in range(len(split_content[i])):
        if(split_content[i][j].count(subject[i])!=0):
            split_words=jieba.lcut(split_content[i][j],cut_all=True)
            sentiment_value_words=1
            has=False
            for word in positive_words:
                times =split_words.count(word)
                if(times!=0):
                    if (subject[i] == "价格" and word == "高"):
                        sentiment_value_words *= pow(-1,times)
                    elif (subject[i] == "油耗" and word == "高"):
                        sentiment_value_words *= pow(-1,times)
                    elif (subject[i] == "价格" and word == "大"):
                        sentiment_value_words *= pow(-1,times)
                    elif (subject[i] == "油耗" and word == "大"):
                        sentiment_value_words *= pow(-1,times)
                    has=True
            for word in negative_words:
                times=split_words.count(word)
                if(times!=0):
                    has=True
                    if (subject[i] == "价格" and word == "低"):
                        sentiment_value_words *= pow(-1,times)
                    elif (subject[i] == "油耗" and word == "低"):
                        sentiment_value_words *= pow(-1,times)
                    elif (subject[i] == "价格" and word == "小"):
                        sentiment_value_words *= pow(-1,times)
                    elif (subject[i] == "油耗" and word == "小"):
                        sentiment_value_words *= pow(-1,times)
                    sentiment_value_words*=pow(-1,times)
            if(has==False):
                sentiment_value_words=0.
            sentiment_value[i]+=sentiment_value_words
for i in range(len(sentiment_value)):
    if(sentiment_value[i]>0):
        sentiment_value[i]=1
    elif sentiment_value[i]<0:
        sentiment_value[i]=-1
num=0
sentiment_value=pd.DataFrame(sentiment_value,dtype=np.int)
for i in range(len(content)):
    if(raw["sentiment_value"][i]==sentiment_value.values[i]):
        num+=1
print(num/len(content))
result["sentiment_value"]=sentiment_value
result["sentiment_value"]=result["sentiment_value"].astype(int)
result.to_csv("predict_value.csv",encoding="UTF-8",index=False)