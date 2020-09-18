from hparams import *
from usegpu import *
from model import DNN

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

'''
setting for data
'''
def MakeDataSet():
    original_test_df = pd.read_csv(test_data_route)
    test_day_df = pd.read_csv(normalized_test_data_route)
    test_day_np = np.array(test_day_df)
    test_day_list = test_day_np.tolist()
    test_dataset = torch.FloatTensor(test_day_list)

    return test_dataset

def MakeTensor():
    '''
    train data
    '''
    train_data_df = pd.read_csv(train_data_route)
    target_data_df = pd.read_csv(target_data_route)

    '''
    change data type
    '''
    train_data_np = np.array(train_data_df)
    target_data_np = np.array(target_data_df)

    train_data_ts = torch.FloatTensor(train_data_np)
    target_data_ts = torch.FloatTensor(target_data_np)

    return train_data_ts, target_data_ts

def MakeDataLoader(train_data_ts, target_data_ts):
    '''
    dataset & dataloader
    '''
    dataset = TensorDataset(train_data_ts, target_data_ts)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    return dataloader

'''
setting for training
'''
def ModelCostOptimizer():
    device = UseGPU()
    '''
    gpu, model, loss function, optimizer
    '''
    model = DNN().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    return model, criterion, optimizer

def accuracy(prediction, y_train):
    return (prediction.argmax(dim=1) == y_train).float().mean().item()

def run_train(dataloader, model, criterion, optimizer):
    device = UseGPU()
    train_loss = []
    train_accuracy = []
    for epoch in range(nb_epochs + 1):
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            prediction = model(x_train)
            accuracy = abs(prediction/y_train).float().mean().item()
            train_accuracy.append(accuracy)
            cost = criterion(prediction, y_train)
            optimizer.zero_grad()
            cost.backward()
            train_loss.append(cost.float().item())
            optimizer.step()
            print('Batch: {}/{} Epoch: {:4d}/{} Accuracy: {:4f} Loss: {:.6f}'.format(
                epoch, nb_epochs, 
                batch_idx+1, len(dataloader),
                accuracy,
                cost.item()
                ))
    
    '''
    model save
    '''
    # 모델 저장
    torch.save(model, PATH + 'model.pt')

    return train_accuracy, train_loss

'''
test
'''
def test(test_dataset):
    '''
    use gpu
    '''
    device = UseGPU()

    '''
    load model
    '''
    model =  torch.load(PATH + 'model.pt')

    with torch.no_grad():
        input_data = test_dataset.to(device)
        prediction = model(input_data).to(device)

    '''
    change data type to make file with csv type
    '''
    original_test_df = pd.read_csv(original_data_route)
    original_test_df = original_test_df[['temp_avg', 'temp_min', 'temp_max']]
    validation_data_df = pd.read_csv(validation_data_route)
    validation_data_df = validation_data_df[['temp_avg', 'temp_min', 'temp_max']][len(validation_data_df)-750 : len(validation_data_df)]
    prediction_np = prediction.cpu().numpy()
    prediction_df = pd.DataFrame(prediction_np * 1000)
    prediction_df.columns = ['temp_avg', 'temp_min', 'temp_max']
    test_result_df = pd.concat([validation_data_df, prediction_df, original_test_df], axis = 0, ignore_index= True)
    test_result_df.to_csv('./data/test_result.csv', index = False)

    return validation_data_df, prediction_df, test_result_df
