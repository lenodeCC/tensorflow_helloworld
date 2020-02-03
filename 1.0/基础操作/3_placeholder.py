# 占位符
import tensorflow as tf

input1 = tf.placeholder(tf.float32,(1,2))
input2 = tf.placeholder(tf.float32,(2,1))
output = tf.matmul(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={
        input1: [[1.,2.]],
        input2: [[2.],[1.]]
    }))
