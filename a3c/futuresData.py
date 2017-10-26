# -*- coding: utf-8 -*-
import numpy as np
import csv
import coinmarketcap_usd_history
import os
from functools import reduce
from a3c.config import *


class futuresData:
    def __init__(self):
        self.mFuturesNum = -1
        self.mInforFieldsNum = -1
        self.mLength = -1
        self.mPoundage = -1
        self.mData = []
        self.mDate = []
        self.mPrice = []

    def loadCryptocurrency(self, use_test_data):
        Cryptosymbols = ['Bitcoin', 'Ethereum']
        start_date = '2016-01-01'
        mid_date = '2017-05-01'
        end_date = '2017-10-31'
        columnnames = ['Close', 'Averageclose', 'Volume']
        rollingwindows = 5

        if use_test_data:
            data_dir = './data/Cryptocurrency_test.csv'
        else:
            data_dir = './data/Cryptocurrency_train.csv'

        keys_dict = {}
        dfcrypto = []
        if not os.path.exists('./data/Cryptocurrency.csv'):
            for sym in Cryptosymbols:
                dftemp = coinmarketcap_usd_history.main([sym, start_date, end_date, '--dataframe'])
                dftemp['Averageclose'] = pd.rolling_mean(dftemp['Close'], window=rollingwindows)
                dftemp.set_index('Date', inplace=True)
                dftemp = dftemp[columnnames]
                if len(dfcrypto) == 0:
                    dfcrypto = dftemp
                else:
                    dftemp = pd.merge(dfcrypto, dftemp, left_index=True, right_index=True, sort=True)

            dftemp.to_csv('./data/Cryptocurrency.csv')
            dftemp_train = dftemp.loc[dftemp.index <= pd.to_datetime(mid_date)]
            dftemp_train.to_csv('./data/Cryptocurrency_train.csv')
            dftemp_test = dftemp.loc[dftemp.index > pd.to_datetime(mid_date)]
            dftemp_test.to_csv('./data/Cryptocurrency_test.csv')

        with open(data_dir, encoding='utf8') as f:
            print('[A3C_data]Loading data from data/Cryptocurrency.csv ...')
            self.mFuturesNum = len(Cryptosymbols)
            self.mInforFieldsNum = len(columnnames)
            self.mLength = 0
            self.mPoundage = 0
            if not use_test_data:
                pddata = pd.read_csv(data_dir)
                args.mean = pddata.mean()
                args.std = pddata.std()

            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i <= rollingwindows:
                    i += 1
                    continue
                else:
                    baddata = False
                    idata = np.zeros([self.mFuturesNum, self.mInforFieldsNum])
                    iprice = np.zeros(self.mFuturesNum)
                    for j in range(0, self.mFuturesNum):
                        dateidx = j * (self.mInforFieldsNum)
                        for k in range(0, self.mInforFieldsNum):
                            istring = row[dateidx + k + 1]
                            if len(istring) == 0:
                                baddata = True
                                break
                            # try:
                            idata[j][k] = (float(istring) - args.mean[dateidx + k]) / args.std[dateidx + k]
                            # except Exception as e:
                            #     pass
                        if baddata == True:
                            break
                        iprice[j] = float(row[dateidx + 1])
                    if baddata == True:
                        i += 1
                        continue
                    self.mData.append(idata.reshape(self.mFuturesNum * self.mInforFieldsNum))
                    self.mDate.append(row[0])
                    self.mPrice.append(iprice)
                    i += 1
                    self.mLength += 1

        print('[A3C_data]Successfully loaded ' + str(self.mLength) + ' data')

    def loadData_moreday0607(self, use_test_data):
        if use_test_data:
            data_dir = './data/moreday0607_test.csv'
        else:
            data_dir = './data/moreday0607_train.csv'
        with open(data_dir, encoding='utf8') as f:
            print('[A3C_data]Loading data from data/moreday0607.csv ...')
            self.mFuturesNum = 6
            self.mInforFieldsNum = 10
            self.mLength = 0
            self.mPoundage = 0
            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i < 4:
                    i += 1
                    continue
                else:
                    baddata = False
                    idata = np.zeros([self.mFuturesNum, self.mInforFieldsNum])
                    iprice = np.zeros(self.mFuturesNum)
                    for j in range(0, self.mFuturesNum):
                        dateidx = j * (self.mInforFieldsNum + 2)
                        for k in range(0, self.mInforFieldsNum):
                            istring = row[dateidx + k + 1]
                            if len(istring) == 0:
                                baddata = True
                                break
                            idata[j][k] = float(istring)
                        if baddata == True:
                            break
                        iprice[j] = idata[j][1]
                    if baddata == True:
                        i += 1
                        continue
                    self.mData.append(idata.reshape(self.mFuturesNum * self.mInforFieldsNum))
                    self.mDate.append(row[0])
                    self.mPrice.append(iprice)
                    i += 1
                    self.mLength += 1

        print('[A3C_data]Successfully loaded ' + str(self.mLength) + ' data')

    def getObservation(self, time):
        return self.mData[time]

    def getPrice(self, time):
        return self.mPrice[time]


import pandas as pd
from collections import defaultdict
import numpy as np

path_list = [
    'data/IC00.csv',
    'data/TF.csv',
]


class Futures_cn(object):
    def __init__(self):
        self.data_df = None
        self.future_num = 0
        self.info_field_num = 0
        self.days = []

    def load_tranform(self, path_list, used_items=['date', 'time', 'open', 'high', 'low', 'close', 'amount', 'volume']):
        '''
        transform the futures data.
        :param path_list:
        :param used_items: the used columns must contain 'date' and 'time'!
        :return:None
        '''
        df_list = []
        count = 0
        for path in path_list:
            temp_df = pd.read_csv(path)[used_items]
            # temp_df.columns = [ item+'_'+str(count) for item in used_items]
            df_list.append(temp_df)
            count += 1
        merge_df = df_list[0]
        for i in range(1, len(df_list)):
            # removed_columns.extend(['date_'+str(i),'time_'+str(i)])
            temp_df = df_list[i]
            merge_df = pd.merge(merge_df, temp_df, how='inner', on=['date', 'time'])

        merge_df.sort_values(['date', 'time'])
        self.data_df = merge_df
        self.future_num = count
        self.info_field_num = len(used_items) - 2
        self.days = list(self.data_df['date'].values)

    def extract_day(self, day=None, replace=True):
        '''
        extract one day info
        :param day: given day to extract or random extract
        :param replace: replace or not
        :return: ndarray (minutes per day, future_num * info_field_num)
        '''
        if len(self.days) == 0:
            self.days = list(self.data_df['date'].values)

        choice = day if day else np.random.choice(self.days)
        if replace:
            self.days.remove(choice)
        tempdata = self.data_df[self.data_df['date'] == choice].values[:, 2:]
        reshaped_data = tempdata.reshape((tempdata.shape[0], self.future_num, self.info_field_num))
        price = reshaped_data[:, :, 0]
        return choice, price, reshaped_data

    def extract_day_for_directTrain(self, day=None, replace=True):
        '''
        extract one day info for direct_train.py
        :param day: given day to extract or random extract
        :param replace: replace or not
        :return:[
          date that choiced,
          price of first minutes as ndarray (1, future_num),
          price of other minutes as ndarray (minutes_per_day-1, future_num),
          info of first (minutes_per_day-1) minutes as ndarray (minutes_per_day - 1, future_num * info_field_num)
          ]

        '''
        if len(self.days) == 0:
            self.days = list(self.data_df['date'].values)

        choice = day if day else np.random.choice(self.days)
        if replace:
            self.days.remove(choice)
        tempdata = self.data_df[self.data_df['date'] == choice].values[:, 2:]
        reshaped_data = tempdata.reshape((tempdata.shape[0], self.future_num, self.info_field_num))
        price = reshaped_data[:, :, 0]
        firstprice = price[0:1, :]
        nextprice = price[1:, :]
        tempdata_nolast = tempdata[0:len(tempdata) - 1, :]
        return choice, firstprice, nextprice, tempdata_nolast


if __name__ == '__main__':
    c = futuresData()
    c.loadCryptocurrency(True)

    # c = Futures_cn()
    # c.load_tranform(path_list)
    # for i in range(10):
    # choice,price,reshaped_data = c.extract_day()
    # print(choice)
    # print(reshaped_data.shape)
    # print(price.shape)
    # print(price)
    # print(reshaped_data[0])

    r1, r2, r3, r4 = c.extract_day_for_directTrain()
    print(r2.shape)
    print(r3.shape)
    print(r4.shape)
