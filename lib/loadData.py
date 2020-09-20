# <20.09.17> by KH
import pandas as pd
import datetime
from . import scaling as SC

"""
data preprocessing
"""

class DataPreprocessing:
    def __init__(self,dataname):
        self.name = dataname

    def dataLoading(self): # 데이터 불러오기

        try:
            self.data = pd.read_csv("data\\" + self.name +".csv")

        except:
            print("data's file type is not csv type. tring excel file type")
            self.data = pd.read_excel("data\\" + self.name + ".xlsx")


    def set_index(self,column_number):
        try:
            self.data.set_index(self.data.columns[column_number],inplace = True)

        except:
            print("index inplace canceld")

    def excel2numeric(self):
        # change data frame's values from string type to numeric type
        for col in self.data.columns:
            self.data[col] = self.data[col].str.replace(",", "")
            self.data[col] = pd.to_numeric(self.data[col])

    def selectcolumn(self,column_Number):
        self.selected_data = self.data[self.data.columns[column_Number]]


    def scaling(self):
        sc = SC.MinMax_scaler()
        self.selected_data =pd.DataFrame(sc.fit_transform(self.selected_data.values.reshape((-1,1))))

        return self.selected_data

    def shift4window(self,data, windowNum):
        # window 만들기
        # this will make window number +1 data frame because of train or test y
        for s in range(1,windowNum+1):
            data['shift_{}'.format(s)] = data[data.columns[0]].shift(s)

        data.dropna(inplace=True)
        self.shifted_data = data
        return self.shifted_data

    def splitTrainTestData(self,std_percent):
        '''
        split Train, Test data with standard date
         have to put std_data with string type like this
        "2019/12/14"
        '''

        split_percent = std_percent
        len_data = len(self.shifted_data)
        std = int(len_data*split_percent/100)
        self.train = self.shifted_data[:std]
        self.test = self.shifted_data[std:]