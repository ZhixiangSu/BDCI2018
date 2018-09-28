import pandas as pd
from gensim.models.word2vec import Word2Vec
import jieba.posseg as pseg
raw=pd.read_csv("train.csv")
content=raw['content']
sentences=[]
for line in content:
    words=pseg.cut(line)
    a_words=[]
    for word,flag in words:
            a_words.append(word)
    sentences.append(a_words)
print(sentences)
model=Word2Vec()
model.build_vocab(sentences)
model.train(sentences,total_examples = model.corpus_count,epochs = model.iter)
model.save("word2vec_model.gz")