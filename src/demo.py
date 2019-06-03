# Demonstrate the results.

import torch
import json
import numpy as np
from IPython import embed
from trainCNN import Net
from trainRNN import BiRNN

# net = torch.load('./model/43.pth')
net = torch.load('./model/rnn.pth')
net.eval()

with open('./data/demo-inputs30d.json', 'r') as f:
    inputSet = json.load(f)
    for i in range(len(inputSet)):
        inputSet[i] = torch.from_numpy(np.array(inputSet[i]))

with open('./data/demo-labels.json', 'r') as f:
    labelSet = json.load(f)
    for i in range(len(labelSet)):
        labelSet[i] = torch.from_numpy(np.array(labelSet[i]))

for i in range(len(inputSet)):
    # out = net(inputSet[i].view((1, 1) + inputSet[i].shape).float())
    out = net(inputSet[i].view((inputSet[i].shape[0], 1, -1)).float())
    print(out)
    print(i, ':', int(torch.argmax(out[0])), ', ', int(labelSet[i]))
