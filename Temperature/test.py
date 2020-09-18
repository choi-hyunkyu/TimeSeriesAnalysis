from hparams import *
from setting import *

import pandas as pd
import torch

'''
change data type to make test dataset
'''
test_dataset = MakeDataSet()

'''
test without training
'''
validation_data_df, prediction_df, test_result_df = test(test_dataset)

'''
visualization
'''
prediction_df.plot().plot()
test_result_df[:2250].plot()
test_result_df[600:1600].plot()
test_result_df.plot().get_figure().savefig('./data/test_result.png')
