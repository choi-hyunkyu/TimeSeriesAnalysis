from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
GPU 사용 선언
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
랜덤시드 고정
'''
torch.manual_seed(777)

'''
하이퍼파라미터 선언
'''
sequence_length = 6
input_size = 2
hidden_size = 16
output_size = 1
num_layers = 2
learning_rate = 1e-3
nb_epochs = 300

'''
데이터 불러오기
'''
original_data_df = pd.read_csv('./data/Champagne_Sales.csv')
print(original_data_df.shape)
print(original_data_df)

'''
데이터프레임 컬럼 이름 변경
'''
original_data_df.columns = ['Month', 'Sales']
print(original_data_df.head())

'''
날짜 하이픈 제거
'''
original_data_df['Month'] = original_data_df['Month'].str.replace(pat=r'[^\w\s]', repl=r'', regex=True) # 하이픈제거
original_data_df = original_data_df.astype('int')

'''
train, test data 분리
'''
train_size = int(len(original_data_df) * 0.7)
train_set = original_data_df[0:train_size]
test_set = original_data_df[train_size - sequence_length:]

'''
스케일링함수 선언
'''
def minmax_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)

    return numerator / (denominator + 1e-7)

'''
데이터셋 함수 선언
'''
def build_data(time_series, sequence_length):
    x_data = []
    y_data = []

    for i in range(0, len(time_series) - sequence_length):
        _x = time_series.values[i: i + sequence_length, :]
        _y = time_series.values[i + sequence_length, [-1]]
        x_data.append(_x)
        y_data.append(_y)

    return np.array(x_data), np.array(y_data)

'''
데이터로더 함수 선언
'''
def MakeDataLoader(x_np, y_np):
    '''
    totensor
    '''
    x_ts = torch.FloatTensor(x_np)
    y_ts = torch.FloatTensor(y_np)

    '''
    dataset & dataloader
    '''
    dataset = TensorDataset(x_ts, y_ts)
    dataloader = DataLoader(dataset, batch_size = sequence_length, shuffle = True)

    return dataloader

train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)

x_train_np, y_train_np = build_data(train_set, sequence_length)
x_test_np, y_test_np = build_data(test_set, sequence_length)

train_dataloader = MakeDataLoader(x_train_np, y_train_np)
test_dataloader = MakeDataLoader(x_test_np, y_test_np)

'''
모델 설계
'''
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        self.fc = nn.Linear(
            hidden_size,
            output_size,
            bias = True
        )

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        return x

model = Net(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

'''
학습
'''
for epoch in range(nb_epochs):
    train_loss = 0.0
    for i, samples in enumerate(train_dataloader):
        x_train, y_train = samples
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = criterion(hypothesis, y_train)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    print("Epoch: {} | Loss: {:.6f}".format(epoch, loss.item()))

torch.save(model, './data/model.pt')

'''
평가
'''
with torch.no_grad():
    predicted_data_list = []
    label_list = []
    for i, samples in enumerate(test_dataloader):
        x_test, y_test = samples
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        prediction = model(x_test)
        predicted_data_list.append(prediction.tolist())
        label_list.append(y_test.tolist())#.cpu().data.numpy())
        loss = criterion(prediction, y_test)

'''
시각화
'''
predicted_data_np = np.array(sum(sum(predicted_data_list, []), []))
label_np = np.array(sum(sum(label_list, []), []))

#%matplotlib inline
plt.plot(label_np)
plt.plot(predicted_data_np)
plt.legend(['original', 'prediction'])
plt.show()