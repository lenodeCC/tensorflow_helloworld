# tensorflow自带helloworld训练集（mnist）
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data', one_hot=True)

batch_size = 100

batch_xs, batch_ys = mnist.train.next_batch(batch_size)

print("batch data %s" % (type(batch_xs)))
print("batch label %s" % (type(batch_ys)))
print("batch data shape %s" % (batch_xs.shape,))
print("batch data label %s" % (batch_ys.shape,))


testimg = mnist.test.images
testlabel = mnist.test.labels
nsample = 5

randidx = np.random.randint(testimg.shape[0], size=nsample)
print(randidx)
for i in randidx:
    curr_img = np.reshape(testimg[i, :], (28, 28))
    curr_label = np.argmax(testlabel[i, :])
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    print(""+str(i)+"th 训练数据"+"标签是 "+str(curr_label))
    plt.show()
