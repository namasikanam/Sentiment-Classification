# Test the model, calculate accuracy, macro F-score, and correlation coefficient.

import torch
import json
from trainCNN import Net
import os
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import numpy as np
import math
from IPython import embed
import math

net = torch.load(os.path.join('./model', 'cnn.pth'))
net.eval()

with open('./data/test-inputs300d.json', 'r') as f:
    inputSet = json.load(f)
    for i in range(len(inputSet)):
        inputSet[i] = torch.from_numpy(np.array(inputSet[i]))

with open('./data/test-labels.json', 'r') as f:
    labelSet = json.load(f)
    for i in range(len(labelSet)):
        labelSet[i] = torch.from_numpy(np.array(labelSet[i]))

print('Loaded')

numberOfTruePrediction = 0
argmaxLabel, argmaxOutput = [], []
sumOfCorrelationCoefficient = 0
totalCounter = 0
for i in range(len(inputSet)):
    hasNaN = False
    for x in labelSet[i]:
        if math.isnan(x):
            hasNaN = True
    if hasNaN:
        continue

    totalCounter += 1

    out = net(inputSet[i].view((1, 1) + inputSet[i].shape).float())[0].detach()

    # Accuracy
    # 假设out没有多重最大值
    numberOfTruePrediction += int(
        torch.argmax(labelSet[i]) == torch.argmax(out))

    # F-score
    argmaxLabel.append(torch.argmax(labelSet[i]))
    argmaxOutput.append(torch.argmax(out))

    # Correlation Coefficient
    sumOfCorrelationCoefficient += abs(pearsonr(out.numpy(),
                                                labelSet[i].numpy())[0])

    print(totalCounter, ': numberOfTruePrediction =', numberOfTruePrediction,
          ', sumOfCorrelationCoefficient =', sumOfCorrelationCoefficient, 'argmaxOutput =', torch.argmax(out), 'argmaxLabel =', labelSet[i])

print('Accuracy =', numberOfTruePrediction / totalCounter)
print('Macro f1 =', f1_score(np.array(argmaxLabel),
                             np.array(argmaxOutput), average='macro'))
print('Correlation Coefficient =', sumOfCorrelationCoefficient / totalCounter)
