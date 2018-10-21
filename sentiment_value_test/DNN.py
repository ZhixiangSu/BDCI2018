import pandas as pd
import jieba
from gensim.models.word2vec import Word2Vec
import jieba.posseg as pseg
import torch
import torch.nn as nn
import random
import time
import math
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
model=Doc2Vec.load("d2v.model")
raw=pd.read_csv("result_tfidf1.csv")
content=raw["content"]
sentiment_value=raw["sentiment_value"]
class RNN(nn.Module):
    def __init__(self,input_size,embedding1_size,embedding2_size,output_size):
        super(RNN,self).__init__()
        self.embedding1_size=embedding1_size
        self.embedding2_size = embedding2_size
        self.i2e1= nn.Linear(input_size, embedding1_size)
        self.i2e2 = nn.Linear(embedding1_size, embedding2_size)
        self.e22o = nn.Linear(embedding2_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.LogSigmoid()
    def forward(self, input):
        embedding1=self.i2e1(input)
        embedding1=self.softmax(embedding1)
        embedding2=self.i2e2(embedding1)
        embedding2=self.softmax(embedding2)
        output=self.e22o(embedding2)
        output=self.softmax(output)
        return output
rnn=RNN(50,25,100,3)
criterion = nn.NLLLoss()
learning_rate = 0.005
def train(category_tensor, line_tensor):
    for i in range(100):
        rnn.zero_grad()
        output= rnn(line_tensor)
        loss = criterion(output, category_tensor)
        loss.backward()
        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)
        return output
def doc_vector2tensor(i):
    vector=model.infer_vector(content[i]).tolist()
    tensor=torch.tensor([vector])
    return tensor
def sentiment2tensor(i):
    tensor=torch.tensor([sentiment_value[i]+1],dtype=torch.long)
    return tensor
def TrainingExample(i):
    doc_tensor=doc_vector2tensor(i)
    sentiment_tensor=sentiment2tensor(i)
    return doc_tensor,sentiment_tensor
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()-1
    return category_i
def evaluate( line_tensor):
    rnn.zero_grad()
    output= rnn(line_tensor)
    return output
for j in range(100):
    for iter in range(len(content)):
        i=random.randint(0,len(content)-1)
        doc_tensor,sentiment_tensor=TrainingExample(i)
        output=train(sentiment_tensor,doc_tensor)
T=0
for iter in range(len(content)):
    doc_tensor,sentiment_tensor=TrainingExample(iter)
    output=evaluate(doc_tensor)
    guess=categoryFromOutput(output)
    print(guess,sentiment_value[iter])
    if(guess==sentiment_value[iter]):
        T+=1
print(T/len(content))