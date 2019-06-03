# 我需要来先写这个py，因为这是最核心的。
# 先来照着paper稍微train一train，然后把它存下来，用demo和test看一看！

# Define and train the cnn model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from sklearn.utils import shuffle
from IPython import embed
import json
import os
import time

start_time = time.time()

# Define the architecture

# Hyper Parameters


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv = nn.ModuleList([nn.Conv2d(1, 100, (3, 300)), nn.Conv2d(1, 100, (4, 300)), nn.Conv2d(1, 100, (5, 300))])
        self.conv = nn.ModuleList([nn.Conv2d(1, 100, (3, 300)), nn.Conv2d(
            1, 100, (4, 300)), nn.Conv2d(1, 100, (5, 300))])
        self.do = nn.Dropout()
        self.fc = nn.Linear(300, 8)

    def forward(self, x):
        # the witdth of filter window h = 3, 4, 5 with 100 feature maps each
        a = []
        for i in range(3):
            # Convulational layer with multiple filter widths and feature maps
            a.append(self.conv[i](x))
            a[i] = a[i].view(a[i].size()[:-1])

            # Max-over-time pooling
            a[i] = F.max_pool1d(a[i], kernel_size=a[i].size()[-1:])
            a[i] = a[i].view(a[i].size()[:-1])

        # # drop out rate p = 0.5 (default)
        x = self.do(torch.cat((a[0], a[1], a[2]), 1))

        # Fully connected layer
        x = F.relu(self.fc(x))

        # softmax output
        x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':
    net = Net()
    net = net.float()

    # Define the loss function and optimizer

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters())

    print('Network is defined.')
    print('Running time: ', time.time() - start_time)

    # Train the network

    with open('./data/train-labels.json', 'r') as f:
        labelSet = json.load(f)
        for i in range(len(labelSet)):
            labelSet[i] = torch.from_numpy(np.array(labelSet[i]))

    print('Label set is set.')
    print('Running time: ', time.time() - start_time)

    with open('./data/train-inputs300d.json', 'r') as f:
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

            for j in range(i*50, (i+1)*50):
                outputs = net(inputSet[j].view(
                    (1, 1) + inputSet[j].size()).float())
                loss = criterion(outputs.float(), torch.unsqueeze(
                    labelSet[j], 0))

                running_loss += loss

                loss.backward()

            optimizer.step()

            # l2 constraint s = 3
            weightMatrix = net.state_dict()['fc.weight']
            for j in range(weightMatrix.size()[0]):
                weightMatrix[j] *= 3 / torch.norm(weightMatrix[j])

            print('Trained. epoch', epoch, 'minibatch', i)
            print('running_loss =', running_loss / 50)
            print('Running time: ', time.time() - start_time)

        # save model of every epoch
        torch.save(net, os.path.join('./model', str(epoch) + '.pth'))
