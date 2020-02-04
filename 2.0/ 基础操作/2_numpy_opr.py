import tensorflow as tf
import numpy as np

x = tf.constant([[1, 9]])
y = tf.add(x, 1)

print(y.numpy())  # 转换为numpy


x1 = np.ones([1, 2])
y1 = tf.add(x1, 2)
print(y1)
