# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:50:25 2018

code on book TensorFlow page79-80：Custom loss function


@author: hxh
"""
import tensorflow as tf
import numpy as np
from numpy.random import RandomState


batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义一个单层的神经网络前向传播的过程，这里是简单的加权
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 预测少了和多了的成本
loss_less = 10
loss_more = 1
# 书中tf.select现已更新为tf.where，功能相同
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size,2)
# 设置回归的正确值为两个输入的和加上一个随机量。加上随机量是为了加入不可预测的噪音，
# 否则不同的损失函数的意义就不大了，因为不同的损失函数都会在能完全预测正确的时候最低。
# 一般来说噪音为一个均值为0的小量，所以这里的噪音设置为 -0.5 ~ 0.5 的随机数。
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1,x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    yy = np.zeros(shape=(128,1))
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min((start + batch_size), dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_:Y[start:end]})
        iyy = sess.run(y, feed_dict={x: X[start:end], y_:Y[start:end]})
        for j in range(len(iyy)): 
            yy[start + j] = iyy[j]
        ww = sess.run(w1)
    
    # 输出观察对比计算值和真实值
    for i in range(128):
        print("计算值:",yy[i],"  真实值:",Y[i])
    









