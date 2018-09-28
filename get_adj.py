import nltk
import jieba
import pandas as pd
import numpy as np
import jieba.posseg as pseg
raw=pd.read_csv("train.csv")
content=raw['content']
a_data=[]
for line in content:
    words=pseg.cut(line)
    a_words=[]
    for word,flag in words:
        if(flag=='a'):
            a_words.append(word)
    a_data.append(a_words)
a_out=pd.DataFrame(a_data)
print(a_out)
writer=pd.ExcelWriter("adj.xlsx")
a_out.to_excel(writer)
writer.close()
