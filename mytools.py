# -*- coding: utf-8 -*-
import datetime
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
class Time_calculater(object):
    def __init__(self):
        self.start=time.time()
        self.last_time=self.start
        self.remain_time=0
    #定义将秒转换为时分秒格式的函数
    def time_change(self,time_init):
        time_list = []
        if time_init/3600 > 1:
            time_h = int(time_init/3600)
            time_m = int((time_init-time_h*3600) / 60)
            time_s = int(time_init - time_h * 3600 - time_m * 60)
            time_list.append(str(time_h))
            time_list.append('h ')
            time_list.append(str(time_m))
            time_list.append('m ')

        elif time_init/60 > 1:
            time_m = int(time_init/60)
            time_s = int(time_init - time_m * 60)
            time_list.append(str(time_m))
            time_list.append('m ')
        else:
            time_s = int(time_init)

        time_list.append(str(time_s))
        time_list.append('s')
        time_str = ''.join(time_list)
        return time_str
    def time_cal(self,i,N):
        now_time=time.time()
        self.remain_time=(now_time-self.last_time)*(N-i-1)
        self.last_time=now_time
        print("剩余时间："+self.time_change(self.remain_time))


if __name__ == '__main__':
    time_calculater=Time_calculater()
    N=10#实际使用时用相应变量替换掉
    for i in range(N):
        time.sleep(1)#为了测试效果添加的
        time_calculater.time_cal(i,N)