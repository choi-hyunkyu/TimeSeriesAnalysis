from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
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
데이터 불러오기
'''
original_data_df = pd.read_csv('./data/original_data.csv')
original_data_df.drop(['location'], axis = 1, inplace = True)
original_data_df.head()

'''
check missing value
'''
original_data_df.isnull().sum()

'''
change missing value from behind value
'''
original_data_df.fillna(method = 'ffill', inplace = True)
original_data_df.isnull().sum()

'''
change dataframe type to numeric
'''
original_data_df = original_data_df.apply(pd.to_numeric)

'''
season data insert

봄 : 3월 ~ 5월 - 0.05
여름 : 6월 ~ 8월 - 0.25
가을 : 9월 ~ 11월 - 0.42
겨울 : 12월 ~ 2월 - 0.28
'''
season_data = pd.DataFrame(original_data_df['month'])
season_data.columns = ['season']
original_data_df = pd.concat([original_data_df, season_data], axis = 1)
original_data_df.info()

'''
season data clustering
'''
for season in original_data_df:
    original_data_df.loc[(original_data_df['season'] >= 1) & (original_data_df['season'] < 3), 'season'] = 0.05
    original_data_df.loc[(original_data_df['season'] >= 3) & (original_data_df['season'] < 6), 'season'] = 0.25
    original_data_df.loc[(original_data_df['season'] >= 6) & (original_data_df['season'] < 9), 'season'] = 0.42
    original_data_df.loc[(original_data_df['season'] >= 9) & (original_data_df['season'] < 12), 'season'] = 0.28
    original_data_df.loc[(original_data_df['season'] >= 12), 'season'] = 0.05

original_data_df.head()

'''
data scaling
'''
max_abs_scaler = MaxAbsScaler()

scaled_data_np = max_abs_scaler.fit_transform(original_data_df)
scaled_data_df = pd.DataFrame(scaled_data_np)
scaled_data_df.columns = ['frontyear', 'backyear', 'month', 'day', 'temp_avg', 'temp_min', 'temp_max', 'season']
scaled_data_df.head()

'''
train, test data 분리
'''
train_size = int(len(scaled_data_df) * 0.7)
train_data_df = scaled_data_df[0:train_size].reset_index(drop = True)
validation_data_df = scaled_data_df[:int(len(scaled_data_df) * 0.15):].reset_index(drop = True)
test_data_df = scaled_data_df[-int(len(scaled_data_df) * 0.15):].reset_index(drop = True)

print("train size:", train_size)
print("result df shape:", scaled_data_df.shape)
print("train df shape:", train_data_df.shape)
print("validation df shape:", validation_data_df.shape)
print("test df shape:", test_data_df.shape)

'''
unnecessary index drop, train & target seperate
'''
x_train_data_df = train_data_df[['frontyear', 'backyear', 'month', 'day', 'season']]
y_train_data_df = train_data_df[['temp_avg', 'temp_min', 'temp_max']]

x_validation_data_df = validation_data_df[['frontyear', 'backyear', 'month', 'day', 'season']]
y_validation_data_df = validation_data_df[['temp_avg', 'temp_min', 'temp_max']]

x_test_data_df = test_data_df[['frontyear', 'backyear', 'month', 'day', 'season']]
y_test_data_df = test_data_df[['temp_avg', 'temp_min', 'temp_max']]

print("x train shape:", x_train_data_df.shape)
print("y train shape:", y_train_data_df.shape)
print("x validation shape:", x_validation_data_df.shape)
print("y validation shape:", y_validation_data_df.shape)
print("x test shape", x_test_data_df.shape)
print("y test shape", y_test_data_df.shape)

'''
GPU 사용
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
랜덤시드 고정
'''
torch.manual_seed(777)

'''
하이퍼파라미터

입력 형태: (Batch Size, Sequence Length, Input Dimension)
'''
batch_size = 256 # 64년치 데이터
sequence_length = 1 # 한 개의 batch 당 몇 개의 sequence가 들어있는 개수
input_size = 5
hidden_size = 32
num_layers = 3
output_size = 3 # Trend, Seasonal, Residual
learning_rate = 1e-5
nb_epochs = 400

'''
데이터셋함수
'''
def MakeDataSet(x_data_df, y_data_df):
    x_ts = torch.FloatTensor(np.array(x_data_df))
    y_ts = torch.FloatTensor(np.array(y_data_df))
    dataset_ts = TensorDataset(x_ts, y_ts)

    return dataset_ts

'''
데이터로더
'''
def MakeDataLoader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    return dataloader

'''
데이터셋
'''
train_dataset_ts = MakeDataSet(x_train_data_df, y_train_data_df)
validation_dataset_ts = MakeDataSet(x_validation_data_df, y_validation_data_df)
test_dataset_ts = MakeDataSet(x_test_data_df, y_test_data_df)

'''
데이터로더
'''
train_dataloader = MakeDataLoader(train_dataset_ts, batch_size)
validation_dataloader = MakeDataLoader(validation_dataset_ts, batch_size)
test_dataloader = MakeDataLoader(test_dataset_ts, batch_size)

'''
사용할 데이터 확인
'''
for i, samples in enumerate(test_dataloader):
    x, y = samples
    print("Batch:", i + 1)
    print("Input:",x.shape)
    print("Target:",y.shape)
    if i == 4:
        break

'''
model 설계
'''
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = 0.3,
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


'''
model, cost, optimizer
'''
model = Net(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

'''
모델 테스트
'''
x, y = list(train_dataloader)[0]
x = x.view(-1, sequence_length, input_size).to(device) # RNN input: batch size, sequence length, input size
y = y.to(device)
hypothesis = model(x)
loss = criterion(hypothesis, y)

print("X shape:", x.shape)
print("Y shape:", y.shape)
print("Hypothesis:", hypothesis.shape)
print("Optimizer:", optimizer)
print("Loss:", loss)

'''
학습
'''
trn_loss_list = []
val_loss_list = []
# Train
for epoch in range(nb_epochs):
    # Train Parameters
    trn_loss = 0.0
    for i, samples in enumerate(train_dataloader):
        # Train Data
        x_train, y_train = samples
        x_train = x_train.view(-1, sequence_length, input_size).to(device)
        y_train = y_train.to(device)
        
        # Train
        model.train()
        hypothesis = model(x_train)
        optimizer.zero_grad()
        train_loss = criterion(hypothesis, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Train Loss
        trn_loss += train_loss.item() / len(train_dataloader)
        
    trn_loss_list.append(trn_loss)

    # Evaluation
    with torch.no_grad():
        # Validation Parameters
        val_loss = 0.0
        for ii, validation_samples in enumerate(validation_dataloader):
            # Validation Data
            x_validation, y_validation = validation_samples
            x_validation = x_validation.view(-1, sequence_length, input_size).to(device)
            y_validation = y_validation.to(device)
            
            # Evaluation
            model.eval()
            prediction = model(x_validation)
            validation_loss = criterion(prediction, y_validation)
            
            # Validation Loss
            val_loss += validation_loss.item() / len(validation_dataloader)
            
    val_loss_list.append(val_loss)
    
    print("Epoch: {:3d} | Train Loss: {:.6f} | Val Loss: {:.6f}".format(epoch + 1, trn_loss, val_loss))

torch.save(model, './data/model.pt')

print("train loss list length:", len(trn_loss_list))
print("validation loss list length:", len(val_loss_list))

'''
결과 데이터 저장
'''
loss_result_df = pd.DataFrame({'Train Loss': trn_loss_list, 'Validation Loss': val_loss_list})
loss_result_df.to_csv('./data/loss_result_df.csv')

'''
train, validation loss 시각화
'''
plt.figure(figsize = (16, 9))
plt.plot(trn_loss_list, label = 'Train Loss')
plt.plot(val_loss_list, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()

'''
평가
'''
original = []
result = []
for i, batch in enumerate(test_dataloader):
    x, y = batch
    x = x.view(-1, sequence_length, input_size).to(device)
    y = y.to(device)
    pred = model(x)
    label = y
    loss = criterion(pred, label)
    original.append(y.tolist())
    result.append(pred.tolist())

print(len(result))
print(len(original))

test_original_np = np.array(sum(sum(original, []), []))
test_result_np = np.array(sum(sum(result, []), []))

'''
데이터프레임 reshape
'''
test_original_df = pd.DataFrame(test_original_np.reshape(-1, 3))
test_result_df = pd.DataFrame(test_result_np.reshape(-1, 3))

print(test_original_df.shape)
print(test_result_df.shape)

'''
결과 데이터 데이터프레임 결합
'''
reshaped_test_original_df = pd.concat([x_test_data_df, test_original_df], axis = 1)
reshaped_test_result_df = pd.concat([x_test_data_df, test_result_df], axis = 1)

'''
예측 데이터 데이터프레임 변환
'''
inversed_test_original_np = max_abs_scaler.inverse_transform(reshaped_test_original_df)
inversed_test_original_df = pd.DataFrame(inversed_test_original_np)

inversed_test_result_np = max_abs_scaler.inverse_transform(reshaped_test_result_df)
inversed_test_result_df = pd.DataFrame(inversed_test_result_np)

'''
데이터프레임 columns 이름 변경
'''
inversed_test_original_df.columns = [['frontyear', 'backyear', 'month', 'day', 'season', 'O_temp_avg', 'O_temp_min', 'O_temp_max']]
inversed_test_result_df.columns = [['frontyear', 'backyear', 'month', 'day', 'season', 'P_temp_avg', 'P_temp_min', 'P_temp_max']]

dropped_test_original_df = inversed_test_original_df[['O_temp_avg', 'O_temp_min', 'O_temp_max']]
dropped_test_result_df = inversed_test_result_df[['P_temp_avg', 'P_temp_min', 'P_temp_max']]

'''
결과 데이터 시각화
'''
plt.figure(figsize = (16, 9))
plt.plot(dropped_test_original_df[['O_temp_avg']], label = 'Original')
plt.plot(dropped_test_result_df[['P_temp_avg']], label = 'Prediction')
plt.legend(loc = 'upper right')
plt.show()
