import pandas as pd
import torch
import torch.nn as nn
import random
raw=pd.read_csv("train.csv")
adj=pd.read_excel("adj.xlsx")
sentiment_value=raw["sentiment_value"]
dic={}
for i in range(len(adj)):
    for j in range(len(adj.values[i])):
        if adj.values[i][j] not in dic:
            dic[adj.values[i][j]]=len(dic)
DIC_SIZE=len(dic)
def word_to_vector(word):
    tensor=torch.zeros(1,DIC_SIZE)
    tensor[0][dic[word]]=1
    return tensor
def line_to_vector(line):
    tensor=torch.zeros(len(line),1,DIC_SIZE)
    for i in range(len(line)):
        tensor[i][0][dic[line[i]]]+=1
    return tensor
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 100
n_categories=3
rnn = RNN(DIC_SIZE, n_hidden,n_categories)
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return  category_i

def randomTrainingExample():
    x=random.randint(0,len(adj)-1)
    category=sentiment_value.values[x]+1
    line=adj.values[x]
    category_tensor=torch.tensor([category], dtype=torch.long)
    line_tensor = line_to_vector(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    #print(output)
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
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

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    #print(category, line, category_tensor, line_tensor)
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # #print iter number, loss, name and guess
    if iter % print_every == 0:
        guess= categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
torch.save(RNN, 'net.pkl')