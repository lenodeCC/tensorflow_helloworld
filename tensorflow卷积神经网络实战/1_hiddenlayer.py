# 隐藏层(手写字内容识别)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data', one_hot=True)

numClass = 10  # 训练类目
inputSize = 784  # 输入图像 28*28=784
numHiddenUnits = 50  # 隐藏层输出数据个数，即所谓的神经元个数
trainingIterations = 10000  # 训练次数
batchSize = 100  # 一次读取数量，根据内存或显存大小决定,一般>=64

# 输入参数，其大小为：输入大小*batch大小
x = tf.placeholder(tf.float32, shape=[None, inputSize])
# 获得结果，其大小为：类目*batch大小
y = tf.placeholder(tf.float32, shape=[None, numClass])

# 参数初始化
# 第一层权重参数，神经元个数*输入大小
W1 = tf.Variable(tf.random_normal([inputSize, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])

# 输出层权重参数，类目*神经元个数
W2 = tf.Variable(tf.random_normal([numHiddenUnits, numClass], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1), [numClass])

# 网络结构
# 隐藏层输出 relu(W1*x+B1)
hiddenLayerOutput = tf.matmul(x, W1)+B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
# 最终层输出
finalOutput = tf.matmul(hiddenLayerOutput, W2) + B2
finalOutput = tf.nn.relu(finalOutput)

# 计算损失 交叉商损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y, logits=finalOutput))

# 梯度下降
opt = tf.train.GradientDescentOptimizer(.1).minimize(loss)

# 是否正确预测
correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y, 1))

# 计算准确度 tf.cast用于将预测结果（bool）转换为float类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    # 初始化全局的变量
    sess.run(tf.global_variables_initializer())
    # 循环trainingIterations次训练
    for i in range(trainingIterations):
        # 获得下一次的batch
        batch = mnist.train.next_batch(batchSize)
        # 获取输入内容，即图像
        batchInput = batch[0]
        # 获取标签 10*1的矩阵，若图片内容为2，则第3项为1，其余为0
        batchLabels = batch[1]
        # 执行训练
        _, trianingLoss = sess.run([opt, loss], feed_dict={
                                   x: batchInput,
                                   y: batchLabels
                                   })
        # 每1000次打印一下
        if i % 1000 == 0:
            # 计算一下当前的准确率
            train_accuracy = accuracy.eval(session=sess, feed_dict={
                x: batchInput,
                y: batchLabels
            })
            print("step %d, training accuracy %g" % (i, train_accuracy))

    # 使用测试集测试下当前模型下的正确率
    batch = mnist.test.next_batch(batchSize)
    testAccuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
    print("test accuracy %g" % (testAccuracy))
