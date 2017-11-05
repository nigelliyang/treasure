# -*- coding: utf-8 -*-
import numpy as np
import csv
import coinmarketcap_usd_history
import os
from functools import reduce
from a3c.config import *
import datetime as dt
import re

from WindPy import *


class futuresData:
    def __init__(self):
        self.mFuturesNum = -1
        self.mInforFieldsNum = -1
        self.mLength = -1
        self.mPoundage = -1
        self.mData = []
        self.mDate = []
        self.mPrice = []

    def loadIndexData(self, use_test_data):
        if use_test_data:
            data_dir = './data/IndexData_test.csv'
        else:
            data_dir = './data/IndexData_train.csv'
        windcode = ["H11001.CSI", "000300.SH", "000905.SH", "000906.SH", "000016.SH", "399006.SZ",
                    "NH0300.NHF", "NH0400.NHF", "NH0500.NHF", "NH0600.NHF", "NH0008.NHF"]
        start_date = '2013-01-01'
        mid_date = '2017-01-01'
        end_date = (pd.datetime.now() + dt.timedelta(-1)).strftime("%Y-%m-%d")
        inputdata = []
        rollingwindows = 5

        if not os.path.exists('./data/IndexData.csv'):
            w.start()
            for code in windcode:
                wsd_data = w.wsd(code, "close,volume", start_date, end_date)
                wsdtemp_df = pd.DataFrame(data=np.mat(wsd_data.Data).T, columns=wsd_data.Fields, index=wsd_data.Times)
                # wsdtemp_df = wsdtemp_df.dropna()
                # rm = re.match(r"([a-zA-Z]+)([0-9]+)(.)([a-zA-Z]+$)", sec)
                # subsec = rm.group(1) + rm.group(3) + rm.group(4)
                wsdtemp_df['inputClose'] = wsdtemp_df['CLOSE']
                wsdtemp_df['Averageclose'] = wsdtemp_df['CLOSE'].rolling(window=rollingwindows).mean()
                # wsdtemp_df['Averagevolume'] = wsdtemp_df['VOLUME'].rolling(window=rollingwindows).mean()
                wsdtemp_df.index = pd.to_datetime(wsdtemp_df.index)
                if len(inputdata) == 0:
                    inputdata = wsdtemp_df
                else:
                    inputdata = pd.merge(inputdata, wsdtemp_df, left_index=True, right_index=True, sort=True)

            inputdata.to_csv('./data/IndexData.csv')

            dftemp_train = inputdata.loc[inputdata.index <= pd.to_datetime(mid_date)]
            dftemp_train.to_csv('./data/IndexData_train.csv')
            dftemp_test = inputdata.loc[inputdata.index > pd.to_datetime(mid_date)]
            dftemp_test.to_csv('./data/IndexData_test.csv')

        inputdata = pd.read_csv(data_dir)
        inputdata.set_index(inputdata.columns[0], inplace=True)
        inputdata.index = pd.to_datetime(inputdata.index)

        print('[A3C_data]Loading data from data/IndexData.csv ...')
        self.mFuturesNum = len(windcode)
        self.mInforFieldsNum = 3
        args.asset_num = self.mFuturesNum
        args.info_num = self.mInforFieldsNum
        args.input_size = args.asset_num * args.info_num
        self.mLength = 0
        self.mPoundage = 0.001
        if not use_test_data:
            args.mean = inputdata.mean()
            args.std = inputdata.std()

        i = 0
        for index in inputdata.index:
            if i <= rollingwindows:
                i += 1
                continue
            else:
                baddata = False
                idata = np.zeros([self.mFuturesNum, self.mInforFieldsNum])
                iprice = np.zeros(self.mFuturesNum)
                for j in range(0, self.mFuturesNum):
                    dateidx = j * (self.mInforFieldsNum + 1)
                    for k in range(0, self.mInforFieldsNum):
                        istring = inputdata.loc[index][dateidx + k + 1]
                        # if len(istring) == 0:
                        #     baddata = True
                        #     break
                        # try:
                        idata[j][k] = (float(istring) - args.mean[dateidx + k + 1]) / args.std[dateidx + k + 1]
                        # idata[j][k] = float(istring)
                        # except Exception as e:
                        #     pass
                    if baddata == True:
                        break
                    iprice[j] = float(inputdata.loc[index][dateidx])
                if baddata == True:
                    i += 1
                    continue
                self.mData.append(idata.reshape(self.mFuturesNum * self.mInforFieldsNum))
                self.mDate.append(index.strftime("%Y-%m-%d"))
                self.mPrice.append(iprice)
                i += 1
                self.mLength += 1

        print('[A3C_data]Successfully loaded ' + str(self.mLength) + ' data')

    def loadFuturesData(self, use_test_data):
        if use_test_data:
            data_dir = './data/FuturesData_test.csv'
        else:
            data_dir = './data/FuturesData_train.csv'

        if not os.path.exists('./data/FuturesData.csv'):
            w.start()

            start_date = '2013-01-01'
            mid_date = '2017-01-01'
            end_date = (pd.datetime.now() + dt.timedelta(-1)).strftime("%Y-%m-%d")
            wset_data = w.wset("sectorconstituent", "date=" + end_date + ";sectorid=1000015510000000")
            wset_df = pd.DataFrame(data=np.mat(wset_data.Data).T, columns=wset_data.Fields)

            inputdata = []
            # wset_df.set_index("date",inplace=True)
            wsd_df = pd.DataFrame(columns=['WINDCODE', 'SEC_NAME', 'PCT_CHG1', 'PCT_CHG2'])

            for sec in wset_df["wind_code"]:
                if sec.find('RS') != -1 or sec.find('B') != -1 or sec.find('WH') != -1 or sec.find(
                        'WR') != -1 or sec.find(
                    'BB') != -1 or sec.find('FB') != -1 or sec.find('FU') != -1 or sec.find('JR') != -1 or sec.find(
                    'LR') != -1 or sec.find('PM') != -1 or sec.find('SF') != -1 or sec.find('RI') != -1:
                    continue
                print(sec)

                wsd_data = w.wsd(sec, "close,volume", start_date, end_date)
                wsdtemp_df = pd.DataFrame(data=np.mat(wsd_data.Data).T, columns=wsd_data.Fields, index=wsd_data.Times)
                # wsdtemp_df = wsdtemp_df.dropna()
                rm = re.match(r"([a-zA-Z]+)([0-9]+)(.)([a-zA-Z]+$)", sec)
                subsec = rm.group(1) + rm.group(3) + rm.group(4)
                wsdtemp_df['Averageclose'] = wsdtemp_df['CLOSE'].rolling(window=5).mean()
                wsdtemp_df['Averagevolume'] = wsdtemp_df['VOLUME'].rolling(window=5).mean()
                wsdtemp_df.index = pd.to_datetime(wsdtemp_df.index)
                wsdtemp_df.index[0]
                if len(inputdata) == 0:
                    inputdata = wsdtemp_df
                else:
                    inputdata = pd.merge(inputdata, wsdtemp_df, left_index=True, right_index=True, sort=True)

            inputdata.to_csv('./data/FuturesData.csv')

            dftemp_train = inputdata.loc[inputdata.index <= pd.to_datetime(mid_date)]
            dftemp_train.to_csv('./data/FuturesData_train.csv')
            dftemp_test = inputdata.loc[inputdata.index > pd.to_datetime(mid_date)]
            dftemp_test.to_csv('./data/FuturesData_test.csv')

        inputdata = pd.read_csv(data_dir)

        with open(data_dir, encoding='utf8') as f:
            print('[A3C_data]Loading data from data/FuturesData.csv ...')
            self.mFuturesNum = len(Cryptosymbols)
            self.mInforFieldsNum = len(columnnames)
            args.asset_num = self.mFuturesNum
            args.info_num = self.mInforFieldsNum
            args.input_size = args.asset_num * args.info_num
            self.mLength = 0
            self.mPoundage = 0.0025
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
                            # idata[j][k] = float(istring)
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

    def loadCryptocurrency(self, use_test_data):
        Cryptosymbols = ['Bitcoin', 'Ethereum']
        start_date = '2016-01-01'
        mid_date = '2017-05-01'
        end_date = '2017-10-31'
        columnnames = ['Close', 'Averageclose', 'Averagevolume']
        # columnnames = ['Close', 'priceratio', 'volumeratio']
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
                dftemp['Averagevolume'] = pd.rolling_mean(dftemp['Volume'], window=rollingwindows)
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
            args.asset_num = self.mFuturesNum
            args.info_num = self.mInforFieldsNum
            args.input_size = args.asset_num * args.info_num
            self.mLength = 0
            self.mPoundage = 0.0025
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
                            # idata[j][k] = float(istring)
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
            self.mPoundage = 0.0025
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
    c.loadIndexData(False)
    # c.loadCryptocurrency(True)

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
