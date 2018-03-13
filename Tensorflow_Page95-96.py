# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:40:25 2018

下载并转化MNIST数据的格式，将数据从原始的数据包中解析成训练和测试神经网络的格式

python: 3.6.2
tensorflow:1.4.0

@author: hxh
"""

from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集，如果指定地址下没有已经下载好的数据集
# 那么TensorFlow会自动从给出的网址中下载

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# 打印Training data size : 55000
print("Training data size : ",mnist.train.num_examples)

# 打印Validating  data size : 5000
print("Validating data size: ", mnist.validation.num_examples)

# 打印Testing data size : 10000
print("Testing data size : ", mnist.test.num_examples)

# 打印
print("Example training data : ", mnist.train.images[0])

# 打印Testing data size : 
print("Testing data size : ",mnist.train.labels[0])





