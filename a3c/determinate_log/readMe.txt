每个检测的记录的形状是: [period, step, asset+2]
asset+2 指的是 各个asset的资产占比+当前杠杆+本次交易回报率
period中，第一个是benchmark，是平均分配资产时的收益情况，相当于大盘走势
第二个是初始值，即随机初始化后的情况