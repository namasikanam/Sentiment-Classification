# 可能需要暴露一个给文件名的接口，因为demo也要用你
# 不对啊，可是我不仅需要处理input data，还需要处理word vector呢……
# 需要把原始数据处理成词向量
# 不存在的词就给它随机一个

# Preprocessing the data and save them as tensor

import json
import os
import random
import numpy as np
import math

# load Chinese word vector

wordVectors = {}

with open('./data/sgns.sogou.word', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        splitLine = line.split(' ')
        wordVectors[splitLine[0]] = list(
            map(lambda x: float(x), splitLine[1:-1]))
vectors = list(wordVectors.values())

print('Chines Word Vectors are loaded successfully.')


def trans(filename):
    counterTmp = 0

    labels = []
    inputs = []
    with open(os.path.join('./data', filename), 'r') as f:
        lines = f.readlines()
        for line in lines:
            splitLine = line.split('\t')

            sentiments = splitLine[1].split(' ')

            maxSentiment, maxIndex = 0, -1
            for i, sentiment in enumerate(sentiments[1:]):
                s = int(sentiment.split(':')[1])
                if s > maxSentiment:
                    maxSentiment = s
                    maxIndex = i
                elif s == maxSentiment:
                    maxIndex = -1
            if maxIndex == -1:
                continue
            labels.append(maxIndex)

            # currentSentiment = []
            # for sentiment in sentiments[1:]:
            #     currentSentiment.append(int(sentiment.split(':')[1]))
            # currentSentiment = np.array(currentSentiment)
            # currentSentiment = list(
            #     np.exp(currentSentiment) / sum(np.exp(currentSentiment)))

            # labels.append(currentSentiment)

            words = splitLine[2].replace('\n', '').split(' ')
            currentWordVectors = []
            for word in words:
                if word in wordVectors:
                    currentWordVectors.append(wordVectors[word])
                else:
                    currentWordVectors.append(random.choice(vectors))
            inputs.append(currentWordVectors)

            counterTmp = counterTmp + 1
            print('News ', counterTmp, ' is worked successfully.')
    with open(os.path.join('./data/', filename.split('.')[1] + '-labels.json'), 'w') as f:
        json.dump(labels, f)
    with open(os.path.join('./data/', filename.split('.')[1] + '-inputs300d.json'), 'w') as f:
        json.dump(inputs, f)

# Translate `sinanews.demo`, `sinanews.test` and `sinanews.train` to word vector


trans('sinanews.demo')
trans('sinanews.test')
trans('sinanews.train')
