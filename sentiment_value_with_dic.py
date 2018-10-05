import re
import pandas as pd
from gensim.models.word2vec import Word2Vec
import jieba.posseg as pseg
import jieba
raw=pd.read_csv("train.csv")
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
for value in sentiment_value:
    if(value>0):
        value=1
    elif value<0:
        value=-1
num=0
sentiment_value=pd.DataFrame(sentiment_value)
for i in range(len(content)):
    if(raw["sentiment_value"][i]==sentiment_value.values[i]):
        num+=1
print(num/len(content))
raw["sentiment_value"]=sentiment_value
raw.to_csv("predict_value.csv")