# 回归测试预测天气
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from lib.loadTemps import datas
import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing

warnings.filterwarnings("ignore")

# 分别得到年月日
years = datas["year"]
months = datas["month"]
days = datas["day"]

dates = [str(str(int(year)) + "-" + str(int(month)) + "-" + str(int(day)))
         for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
# print(dates[:5])

# 将提取特征
datas = pd.get_dummies(datas)

# 标签
labels = np.array(datas['actual'])

# 在特征中去掉标签
datas = datas.drop('actual', axis=1)

# 名字单独保存一下，以备后患
feature_list = list(datas.columns)

# 转换成合适的格式
datasArr = np.array(datas)


input_features = preprocessing.StandardScaler().fit_transform(datas)

print(input_features)

layers = tf.keras.layers

model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer="random_normal",
                       kernel_regularizer=tf.keras.regularizers.l2(0.03)))  # 权重参数随机，添加正则化
model.add(layers.Dense(32, kernel_initializer="random_normal",
                       kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1, kernel_initializer="random_normal",
                       kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
              loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25,
          epochs=10, batch_size=64)

model.summary()

predict = model.predict(input_features)
