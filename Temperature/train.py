'''
온도 세 개를 전부 학습합니다.
'''
from hparams import *
from setting import *

import torch
import pandas as pd

import matplotlib.pyplot as plt

'''
data load
'''
train_data_ts, target_data_ts = MakeTensor()
dataloader = MakeDataLoader(train_data_ts, target_data_ts)

'''
setting for training
'''
model, criterion, optimizer = ModelCostOptimizer()
model_params = list(model.parameters())

'''
run training
'''
train_accuracy, train_loss = run_train(dataloader, model, criterion, optimizer)
plt.subplot(1,2,1)
plt.plot(train_loss)
plt.title('train_loss - 1950~2020')

plt.subplot(1,2,2)
plt.plot(train_accuracy)
plt.title('train_accuracy - 1950~2020')

plt.savefig('./data/train_result.png')
