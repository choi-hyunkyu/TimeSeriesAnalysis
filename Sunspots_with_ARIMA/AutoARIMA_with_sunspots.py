from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from pmdarima.arima import ADFTest, auto_arima
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import warnings

'''
데이터 불러오기
'''
original_data_df = pd.read_csv('./data/Sunspots.csv', header = 0, parse_dates = [0], index_col = 0, squeeze = True)
original_data_df = original_data_df.reset_index(drop = True)
original_data_df.head()

'''
데이터프레임 컬럼 이름 변경
'''
original_data_df.columns = ['Month', 'Sunspot num']
print(original_data_df.head())

'''
날짜 datetime 타입으로 변환
'''
original_data_date_list = original_data_df['Month'].tolist()
original_data_date_list = pd.to_datetime(original_data_date_list)
original_data_df['Month'] = original_data_date_list
print(original_data_df.head())
type(original_data_df['Month'][0])

'''
Month 컬럼 인덱스 부여
'''
original_data_df = original_data_df.set_index('Month').astype(int) # Month 컬럼에 인덱스 부여
original_data_df = original_data_df[:1000]
print(original_data_df.head())

'''
결측치 확인
'''
print(original_data_df.isnull().sum())

'''
train, test 데이터 분리
'''
train_data_df = original_data_df[:round(len(original_data_df) * 0.8)]
test_data_df = original_data_df[round(len(original_data_df) * 0.8):]
print(len(original_data_df)) # 3252 train:2500, test:752
print(len(train_data_df) + len(test_data_df))

'''
데이터 시각화
'''
plt.figure(figsize = (8, 5))
plt.plot(train_data_df, label = 'Training')
plt.plot(test_data_df, label = 'Test')
plt.legend(loc = 'upper left')
plt.show()

'''
decomposition 확인 (trend, seasonal, resid)
'''
result = seasonal_decompose(train_data_df, model = 'additive')
re_plot = result.plot()
plt.show()

'''
adfuller() 사용
'''
# stationary 확인
train_data_np = train_data_df.values
result = adfuller(train_data_np)
print("ADF Statistic: {:.6f}".format(result[0]))
print("p-value: {:.6f}".format(result[1], end = ''))

if result[1] <= 0.05:
    print(" => Stationary Data")
else:
    print(" => Non-Stationary Data")

'''
train 데이터 분석
'''
adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(train_data_df)

'''
경고 무시
'''
warnings.filterwarnings('ignore')

'''
ARIMA 모델 학습

Auto ARIMA 탐색 범위 설정
  * ARIMA 차수 p : 0 ~ 5
  * ARIMA 차수 d : 1 고정
  * ARIMA 차수 q : 0 ~ 5
  * Seasonality 차수 P : 0 ~ 5
  * Seasonality 차수 D : 1 고정
  * Seasonality 차수 Q : 0 ~ 5
  * Seasonality 간격 m : 12 (연 단위 반복)
'''
arima_model = auto_arima(
    train_data_df,

    # ARIMA 차수
    start_p = 0, d = 1, start_q = 0, # ARIMA 차수 d = 1
    max_p = 5, max_d = 5, max_q = 5,

    # Seasonality 차수
    start_P = 0, D = 1, start_Q = 0, # Seasonality 차수 d = 1
    max_P = 5, max_D = 5, max_Q = 5,
    m = 12, # window = 12
    seasonal = True,

    # 기타 설정
    error_action = 'warn',
    trace = True,
    supress_warnings = True,
    stepwise = True,
    random_state = 100,
    n_fits = 50
)

'''
모델 정보
'''
arima_model.summary()

'''
예측 및 성능 평가
'''
predict_result = arima_model.predict(n_periods = len(test_data_df))

prediction = pd.DataFrame(predict_result,index = test_data_df.index)
prediction.columns = ['predicted Sunspot number']
prediction

'''
결과 시각화
'''
plt.figure(figsize = (8, 5))
plt.plot(train_data_df, label = "Training")
plt.plot(test_data_df, label = "Test")
plt.plot(prediction, label = "Predicted")
plt.legend(loc = 'upper left')
plt.show()

'''
오차 및 정확도
'''
rmse = sqrt(mean_squared_error(test_data_df, prediction))
print('RMSE: %.3f' % rmse)

r2_score_ret = r2_score(test_data_df, prediction)
print('R2 Score: %.5f' % r2_score_ret)