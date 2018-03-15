# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 08:28:07 2018

chapter5_5.5
结合变量管理机制和模型持久化机制进行的完整的神经网络训练实践

mnist_inference.py : 前向传播的过程以及神经网络中的参数

python: 3.6.2
tensorflow:1.4.0

@author: hxh
"""

import tensorflow as tf

# 定义神经网络结构的相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


 
def get_weight_variable(shape, regularizer):
    '''通过tf.get_variable函数来获取变量'''
    #print("name:",tf.get_variable_scope().name)
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    # 当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    '''定义神经网络的前向传播过程'''
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    
    # 声明第二层
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    
    for ele1 in tf.trainable_variables():  
        print(ele1.name)
    return layer2
    








