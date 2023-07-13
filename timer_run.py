from threading import Timer
from train_pix2pix import *
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
def loop(item):
    opt = parser_args()
    main(opt)
# 执行多个train
def my_job(jobs,repeat=1):
    for model in jobs:
        for i in range(repeat):
            print('-' * 50, '现在执行：', model, '-' * 50)
            loop(model)
            sleep(20)
if __name__ == '__main__':
    repeat = 1 #重复次数
    jobs =['1']# 任务列表

    Timer(set_timer(sec=1),my_job,(jobs,repeat)).start()
    # Timer(set_timer(hour=5),my_job,(jobs2,repeat)).start()

# "Unet0": '',
# "Unet_mobile_s": '',
# 'lraspp_mobilenetv3_large': '',
# "FCN": '',
# "SegNet": '',
# "DenseASPP": '',
# 'deeplabV3p': '',