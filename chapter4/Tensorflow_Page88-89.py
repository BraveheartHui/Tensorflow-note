# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:50:25 2018

code on book TensorFlow page88-89：
    5层神经网络带L2正则化的损失函数计算方法

python: 3.6.2
tensorflow:1.4.0

@author: hxh
"""
import tensorflow as tf
import numpy as np
from numpy.random import RandomState



# 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为'losses'的集合中
def get_weights(shape, lambda_):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    
    # add_to_collection 函数将这个新生成变量的L2正则化损失项加入集合
    # 这个函数第一个参数'losses'是集合的名字，第二个参数是要加入这个集合的内容
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda_)(var))
    #返回生成的变量
    return var


x = tf.placeholder(tf.float32, shape=(None, 2),name='x-input')    
y_ = tf.placeholder(tf.float32, shape=(None, 1),name='y-input')
batch_size = 8

# 定义了每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]

# 神经网络的层数
n_layers = len(layer_dimension)

# 该变量维护前向传播时最深层的节点，开始时是输入层
cur_layer = x

# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成一个5层的全连接神经网络
for i in range(1,n_layers):
    print(i," cur_layer：",cur_layer)
    # layer_dimension[i]为下一层的节点数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合中
    weight = get_weights([in_dimension,out_dimension], 0.01)
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    # 使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前节点的个数
    in_dimension = layer_dimension[i]

# 定义神经网络前向传播的同时已经将所有L2正则化损失加入图上的集合
# 这里只需要计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection 返回一个列表，这个列表是所有这个集合中的元素
# 在这个样例中，这些元素就是损失函数的不同部分，将它们加起来可以得到最终的损失函数了
loss = tf.add_n(tf.get_collection('losses'))







