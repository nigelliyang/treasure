#environment class for finantial RL by @jialiang.cui
import gym.spaces
import numpy as np

from futuresData import futuresData

class futuresGame:
    def __init__(self,data):
        self.mSupportEmpty = 1
        self.mData = data
        self.mFuturesNum = self.mData.mFuturesNum
        self.mInforFieldsNum = self.mData.mInforFieldsNum
        self.ction_space = gym.spaces.box.Box(np.linspace(0,0,self.mFuturesNum+self.mSupportEmpty),np.linspace(1,1,self.mFuturesNum+1))
        self.observation_space = gym.spaces.box.Box(np.zeros(self.mInforFieldsNum * self.mFuturesNum),np.linspace(100000000,100000000,self.mInforFieldsNum * self.mFuturesNum))

    def reset(self, initProperty = 100000.0):
        self.totalReward = 0
        self.mProperty = initProperty
        self.mAssetAllocation = np.zeros(self.mFuturesNum + self.mSupportEmpty)
        if self.mSupportEmpty:
            self.mAssetAllocation[-1] = 1
        observation = self.mData.getObservation(0)
        self.mPrice = self.mData.getPrice(0)
        self.time = 1
        return observation


    def step(self, action):
        assert self.time <= self.mData.mLength - 1
        assert len(action) == self.mFuturesNum + self.mSupportEmpty
        #update property and assert allocation

        newPrice = self.mData.getPrice(self.time)
        newProperty = 0.0
        reward = 0.0
        newContrib = np.zeros(self.mFuturesNum + self.mSupportEmpty)
        for i in range(0,self.mFuturesNum):
            reward -= self.mData.mPoundage * abs(self.mAssetAllocation[i] - action[i]) * self.mProperty
            oldContribi = self.mProperty  * action[i]
            newContrib[i] = oldContribi / self.mPrice[i] * newPrice[i]
            newProperty += newContrib[i]
        if self.mSupportEmpty == 1:
            newProperty += action[-1] * self.mProperty
        reward += newProperty - self.mProperty
        percentageReward = reward / self.mProperty

        #update
        for i in range(0,self.mFuturesNum):
            self.mAssetAllocation[i] = newContrib[i] / newProperty
        if self.mSupportEmpty == 1:
            self.mAssetAllocation[-1] = action[-1] * self.mProperty / newProperty
        self.mPrice = newPrice
        self.mProperty = newProperty

        #update observation
        observation = self.mData.getObservation(self.time)

        self.time += 1
        if self.time >= self.mData.mLength:
            done = True
        else:
            done = False
        info = {}
        self.totalReward += reward
        return [observation, self.mAssetAllocation, percentageReward, done, info]


class FuturesGame_cn(object):

    def __init__(self,data,step = 30):
        self._data = data
        self._step = step

        self.support_empty = 1 #mush be 1 or 0
        self.future_num = data.future_num
        self.info_field_num = data.info_field_num

        self.day_sign = None
        self.price = None #shape=(minutes_num,asset_num)
        self.daydata = None #shape=(minutes_num,asset_num,info_field_num)
        self.minutes_num = None

        self.total_property = None
        self.allocation = None
        self.time = None #current time
        self.terminate = True


    def reset(self, init_property = 50000):
        self.total_property = init_property
        # random extract ont day dato as an episode
        self.day_sign,self.price,self.daydata = self._data.extract_day()
        self.minutes_num = self.price.shape[0]
        # current time
        # the state is before current time open price
        # 30 min state is 0 - 29 minutes open, close, etc. we buy futures condition on the history info
        self.time = self._step
        self.terminate = False
        self.allocation = np.zeros(shape=(self.future_num + self.support_empty))
        if self.support_empty:
            self.allocation[-1] = 1

        return self.daydata[self.time - self._step:self.time].reshape((self._step,self.future_num * self.info_field_num))


    def step(self,action):
        assert not self.terminate
        assert len(action) == self.future_num + self.support_empty

        last_price = self.price[self.time - 1] # last minute's open price
        next_price = self.price[self.time] #at the end of this minute we make the open price as the price
        next_allocation = np.zeros(shape=(self.future_num + self.support_empty))
        next_property_contrib = []
        for i_asset in range(self.future_num):
            temp_quntity = self.total_property * action[i_asset] / last_price[i_asset] #quntity of asset
            temp_next_property = next_price[i_asset] * temp_quntity
            next_property_contrib.append(temp_next_property)
        if self.support_empty == 1:
            next_property_contrib.append(self.total_property * action[-1])
        next_property = sum(next_property_contrib)
        reward_percent = next_property / self.total_property

        for i in range(self.future_num):
            next_allocation[i] = next_property_contrib[i] / next_property
        if self.support_empty == 1:
            next_allocation[-1] = next_property_contrib[-1] / next_property
        #update data
        self.time += 1
        self.total_property = next_property
        self.allocation = next_allocation
        next_state = self.daydata[self.time - self._step:self.time].reshape((self._step,self.future_num * self.info_field_num))
        #check terminate
        if self.time >= self.minutes_num:
            self.terminate = True

        return next_state,self.allocation,reward_percent,self.terminate,{}

if __name__ == '__main__':
    import futuresData as fd
    c = fd.Futures_cn()
    c.load_tranform(fd.path_list)
    f = FuturesGame_cn(c,step=50)
    f.reset()
    for i in range(1000):
        action = np.ones(f.future_num + f.support_empty) / (f.future_num + f.support_empty)
        state,allo,reward,terminate,_ = f.step(action)
        print('step=',i)
        print('stateshape=',state.shape)
        print('allo=',allo)
        print('reward=',reward)
        print('terminate=',terminate)









