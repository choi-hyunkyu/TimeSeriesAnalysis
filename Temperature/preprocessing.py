# 전쟁 중 빠져있는 temp_avg, temp_min, temp_max 예측
# 연도 별 온도 상승률 (평균, 최저, 최고 기온)
# http://www.climate.go.kr/home/09_monitoring/assets/download/korea_100years_climatechange_report.pdf
from hparams import *

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

'''
load data
'''
data = pd.read_csv(original_data_route, dtype = 'unicode')
test = pd.read_csv(test_data_route, dtype = 'unicode')

'''
check missing value

1950.09.01 ~ 1953.11.30 전쟁 기간동안 온도에 대한 기록이 없음
'''
data.isnull().sum()
test.isnull().sum()

'''
change missing value from behind value
'''
data.fillna(method = 'ffill', inplace = True)
data.isnull().sum()

'''
change dataframe type to numeric
'''
data = data.drop(['location'], axis = 1)
test = test.drop(['location'], axis = 1)
data = data.apply(pd.to_numeric)
test = test.apply(pd.to_numeric)

'''
season data insert

봄 : 3월 ~ 5월 - 0.05
여름 : 6월 ~ 8월 - 0.25
가을 : 9월 ~ 11월 - 0.42
겨울 : 12월 ~ 2월 - 0.28
'''
season_data = pd.DataFrame(data['month'])
season_data.columns = ['season']
data = pd.concat([data, season_data], axis = 1)
data.info()

season_test = pd.DataFrame(test['month'])
season_test.columns = ['season']
test = pd.concat([test, season_test], axis = 1)
test.info()

'''
season data clustering
'''
for season in data:
    data.loc[(data['season'] >= 1) & (data['season'] < 3), 'season'] = 0.05
    data.loc[(data['season'] >= 3) & (data['season'] < 6), 'season'] = 0.25
    data.loc[(data['season'] >= 6) & (data['season'] < 9), 'season'] = 0.42
    data.loc[(data['season'] >= 9) & (data['season'] < 12), 'season'] = 0.28
    data.loc[(data['season'] >= 12), 'season'] = 0.05

for season in test:
    test.loc[(test['season'] >= 1) & (test['season'] < 3), 'season'] = 0.05
    test.loc[(test['season'] >= 3) & (test['season'] < 6), 'season'] = 0.25
    test.loc[(test['season'] >= 6) & (test['season'] < 9), 'season'] = 0.42
    test.loc[(test['season'] >= 9) & (test['season'] < 12), 'season'] = 0.28
    test.loc[(test['season'] >= 12), 'season'] = 0.05
    
data.head()
test.head()

'''
unnecessary index drop, train & target seperate
'''
train = data.drop(['temp_avg', 'temp_min', 'temp_max'], axis = 1)
target = data.drop(['frontyear', 'backyear', 'month', 'day', 'season'], axis = 1)

'''
train data scaling
'''
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

train[['frontyear', 'backyear', 'month', 'day']] = min_max_scaler.fit_transform(train[['frontyear', 'backyear', 'month', 'day']]) # 연월일의 주기성을 나타내기 위해 minmax scaler 사용
# target[['temp_avg', 'temp_min', 'temp_max']] = min_max_scaler.fit_transform(target[['temp_avg', 'temp_min', 'temp_max']]) # 영상 영하 기온을 표현하기 위해서 standard scaler 사용
target = target / 1000
test[['frontyear', 'backyear', 'month', 'day']] = min_max_scaler.fit_transform(test[['frontyear', 'backyear', 'month', 'day']]) # 연월일의 주기성을 나타내기 위해 minmax scaler 사용

'''
datafile save
'''
train.to_csv('./data/train.csv', index = False)
target.to_csv('./data/target.csv', index = False)

test.to_csv('./data/test.csv', index = False)
