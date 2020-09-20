#<20.09.19> by KH


from lib.loadData import DataPreprocessing
from lib.scaling import MinMax_scaler
from lib.LSTM import LSTM

import pandas as pd
import numpy as np

if __name__=="__main__":

    '''
    input_l 
    input list
    0 : file name @ data directory
    1 : index colonm number, start from 0 if you don't want this, just put string type 
    2 : object colonm number, default 0
    3 : split percent
    4 : window number
    '''
    ################ 필수 작성 ####################
    input_l = ["price", 0, 0, 80, 20]
    ##############################################

    loadData= DataPreprocessing(input_l[0])
    loadData.dataLoading()
    loadData.set_index(input_l[1])
    loadData.excel2numeric()
    loadData.selectcolumn(input_l[2])
    loadData.scaling()
    loadData.shift4window(loadData.selected_data,10)
    loadData.splitTrainTestData(input_l[3])

    x_train = loadData.train.drop(loadData.train.columns[0],axis=1)
    y_train = loadData.train[loadData.train.columns[0]]

    x_test = loadData.test.drop(loadData.test.columns[0], axis=1)
    y_test = loadData.test[loadData.test.columns[0]]


    lstm_model = LSTM()
    lstm_model.operation(x_train.values, y_train.values, x_test.values, y_test.values)

# LSTM 의 input shape도 input list와 연동되도록 하기.

    # Graph
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(lstm_model.model.predict(x_test.values.reshape(np.shape(x_test)[0],np.shape(x_test)[1],1)), label = 'predict')
    plt.plot(y_test.values, label = 'y_test')
    plt.legend()
