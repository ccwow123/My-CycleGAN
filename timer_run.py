from threading import Timer
# from train_dev2 import *
from train_dev2 import *
import torch
from time import sleep
# 设置延时时间
def set_timer(hour=0, min=0, sec=0):
    # 小时转秒
    def hour2sec(hour):
        return hour * 60 * 60
    # 分钟转秒
    def min2sec(min):
        return min * 60
    return hour2sec(hour) + min2sec(min) + sec
# 执行单个train
def loop(cfg_path):
    torch.cuda.empty_cache()
    # 参数解析
    args = parse_args(cfg_path)
    # 创建模型
    model = Trainer(args)
    # 模型训练
    model.run()
    torch.cuda.empty_cache()
# 执行多个train
def my_job(jobs,repeat=1):
    for key,_ in jobs.items():
        for i in range(repeat):
            print('-' * 50, '现在执行：', key, '-' * 50)
            loop(key)
            sleep(5)
if __name__ == '__main__':
    repeat = 1 #重复次数
    jobs ={
        "cycleGAN_ex": '',

    }

    Timer(set_timer(sec=3),my_job,(jobs,repeat)).start()

