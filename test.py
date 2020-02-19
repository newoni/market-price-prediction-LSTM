import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler

from keras.layers import LSTM
from keras.models import Sequential
from keras. layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping

"""
data preprocessing
"""
# 데이터 불러오기
data = pd.read_csv("data\\price.csv")
data.set_index(data.columns[0],inplace = True)

data = data[::-1]

for col in data.columns:
    data[col] = data[col].str.replace(",", "")
    data[col] = pd.to_numeric(data[col])

pr_data = data[data.columns[0]]





# train, test 데이터 나누기
split_date = "2019/12/14"

train = pr_data.loc[:split_date]
train_arr = pr_data.loc[:split_date].values.reshape(-1,1)
test = pr_data.loc[split_date:]
test_arr = pr_data.loc[split_date:].values.reshape(-1,1)

# 데이터 스케일링
sc = MinMaxScaler()

sc.fit(train_arr)
train_scaled = sc.transform(train_arr)
test_scaled = sc.transform(test_arr)

train_sc_df = pd.DataFrame(train_scaled, columns = ["Scaled"],index = train.index)
test_sc_df = pd.DataFrame(test_scaled, columns = ["Scaled"],index = test.index)

# window 만들기

for s in range(1,13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df[train_sc_df.columns[0]].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df[train_sc_df.columns[0]].shift(s)


# 트레이닝셋과 테스트셋 만들기
x_train = train_sc_df.dropna().drop(train_sc_df.columns[0],axis = 1)
y_train = train_sc_df.dropna()[train_sc_df.columns[0]]

x_test = test_sc_df.dropna().drop(test_sc_df.columns[0],axis = 1)
y_test = test_sc_df.dropna()[test_sc_df.columns[0]]

x_train = x_train.values
y_train = y_train.values.reshape(-1,1)
x_test = x_test.values
y_test = y_test.values.reshape(-1,1)

x_train_t = x_train.reshape(x_train.shape[0],12,1)
x_test_t = x_test.reshape(x_test.shape[0],12,1)

"""
set model
"""

K.clear_session()
model = Sequential() #Sequential Model
model.add(LSTM(20, input_shape = (12,1))) # (timestep, feature)
model.add(Dense(1)) # output = 1
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.summary()

'''
model fitting
'''

early_stop = EarlyStopping(monitor = 'loss', patience = 1, verbose = 1)
model.fit(x_train_t, y_train, epochs = 100, batch_size = 50, verbose = 1, callbacks = [early_stop])

y_pred = model.predict(x_test_t)

print(y_test - y_pred)