# Define and train rnn Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from IPython import embed
import json
import os
import time
import random

start_time = time.time()

# Define the architecture

class BiRNN(nn.Module):
    # 我还并不懂这里的num_hiddens和num_layers是什么。
    def __init__(self, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()

        # 双向LSTM
        self.encoder = nn.LSTM(input_size = embed_size, hidden_size = num_hiddens, num_layers = num_layers, bidirectional = True)

        # 一个全连接层
        self.decoder = nn.Linear(2 * 2 * num_hiddens, 8)
        # self.decoder = nn.Linear(num_hiddens, 8)
    
    def forward(self, x):
        # LSTM
        x = self.encoder(x)

        # 此时消弭了sequence这一维（原为第0维）。
        x = torch.cat((x[0][0], x[0][-1]), 1)

        # FC to 8
        x = self.decoder(x)

        # the return shape is (batch, 8)
        x = F.softmax(x, dim=1)

        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == '__main__':
    embed_size, num_layers = 300, 2
    net = BiRNN(embed_size, embed_size, num_layers).float()
    net.apply(init_weights)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # 0.01 是书中的参数
    optimizer = optim.Adam(net.parameters(), lr = 0.01)

    print('Network is defined.')
    print('Running time: ', time.time() - start_time)

    # Train the network

    with open('./data/train-labels.json', 'r') as f:
        labelSet = json.load(f)
        for i in range(len(labelSet)):
            labelSet[i] = torch.from_numpy(np.array(labelSet[i]))

    print('Label set is set.')
    print('Running time: ', time.time() - start_time)

    with open('./data/train-inputs' + str(embed_size) + 'd.json', 'r') as f:
        inputSet = json.load(f)
        for i in range(len(inputSet)):
            inputSet[i] = torch.from_numpy(np.array(inputSet[i]))

    print('Input set is set.')
    print('Running time: ', time.time() - start_time)

    for epoch in range(1000):
        # random shuffle
        random_list = list(zip(labelSet, inputSet))
        random.shuffle(random_list)
        labelSet, inputSet = zip(*random_list)

        for i in range(len(labelSet) // 50):
            running_loss = 0

            # zero the parameter gradients
            optimizer.zero_grad()

            # 书中没有给minibatch的大小，我还是用50吧。
            for j in range(i*50, (i+1)*50):
                outputs = net(inputSet[j].view(inputSet[j].size()[0], 1, -1).float())
                loss = criterion(outputs.float(), torch.unsqueeze(labelSet[j], 0))
                running_loss += loss

                loss.backward()

            optimizer.step()

            print('Trained. epoch', epoch, 'minibatch', i)
            print('running_loss =', running_loss / 50)
            print('Running time: ', time.time() - start_time)

        # save model of every epoch
        torch.save(net, os.path.join('./model', 'rnn-' + str(epoch) + '.pth'))