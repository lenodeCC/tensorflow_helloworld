# 读取天气情况
import pandas as pd

# year,moth,day,week分别表示的具体的时间
# temp_2：前天的最高温度值
# temp_1：昨天的最高温度值
# average：在历史中，每年这一天的平均最高温度值
# actual：这就是我们的标签值了，当天的真实最高温度
# friend：这一列可能是凑热闹的，你的朋友猜测的可能值，咱们不管它就好了

datas = pd.read_csv("./data/temps.csv")

print(datas.head())  # 看看数据长什么样子
