# numpy数据集转换为tensorflow数据集
import numpy as np
import tensorflow as tf
a = np.zeros((3,4))
ta = tf.convert_to_tensor(a)

with tf.Session() as sess:
    print(sess.run(ta))