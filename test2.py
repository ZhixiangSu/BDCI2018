import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
from torch import optim
import torch.nn.functional as F
raw=pd.read_csv("train.csv")
adj=pd.read_excel("words.xlsx")
adj.dropna(axis=0, how='all')
sentiment_value=raw["sentiment_value"]
dic={}
num=[0,0,0]
for i in range(len(adj)):
    num[sentiment_value[i]+1]+=1
    for j in range(len(adj.values[i])):
        if (adj.values[i][j] not in dic):
            dic[adj.values[i][j]]=len(dic)
DIC_SIZE=len(dic)
print(num)
def word_to_vector(word):
    tensor=torch.LongTensor(dic[word])
    return tensor
def line_to_vector(line):
    line1=[]
    for i in range(len(line)):
        if line[i] is not np.nan or i<1:
            line1.append(dic[line[i]])
    tensor=torch.LongTensor(line1)
    #print(tensor)
    return tensor
class RNN(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.gru=nn.GRU(embedding_size,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    def initHidden(self):
        return torch.zeros(1,1, self.hidden_size)

n_hidden = 100
n_categories=3
embedding_size=50
rnn = RNN(DIC_SIZE,embedding_size, n_hidden,n_categories)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i.item()
    return  category_i

def randomTrainingExample():
    x=random.randint(0,len(adj)-1)
    category=sentiment_value.values[x]+1
    line=adj.values[x]
    category_tensor=torch.tensor([category], dtype=torch.long)
    line_tensor = line_to_vector(line)
    return category, line, category_tensor, line_tensor
criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor, encoder_optimizer):
    hidden = rnn.initHidden()
    encoder_optimizer.zero_grad()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)/num[category_tensor.data]
    #print(output)
    loss.backward()
    encoder_optimizer.step()


    return output, loss.item()
import time
import math

n_iters = 100000
print_every = 500
plot_every = 100



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
rnn_optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    #print(category, line, category_tensor, line_tensor)
    output, loss = train(category_tensor, line_tensor,rnn_optimizer)
    current_loss += loss

    # #print iter number, loss, name and guess
    if iter % print_every == 0:
        guess= categoryFromOutput(output)
        print(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
torch.save(RNN, 'net.pkl')
'''
RNN=torch.load('net.pkl')
tests=1000
correct_num=0
rnn_optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
for i in range(tests):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    # print(category, line, category_tensor, line_tensor)
    output, loss = train(category_tensor, line_tensor, rnn_optimizer)
    current_loss += loss
    guess = categoryFromOutput(output)
    print(output)
    if guess == category :
        correct = '✓'
        correct_num += 1
    else :
        correct='✗ (%s)' % category
    print(guess, correct)
print(correct_num/tests)
'''