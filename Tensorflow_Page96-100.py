# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:14:21 2018

使用MNIST数据集训练一个三层的神经网络

python: 3.6.2
tensorflow:1.4.0

@author: hxh
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Mnist数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 500    # 隐藏节点个数

BATCH_SIZE = 100    

LEARNING_RATE_BASE = 0.8   # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 描述模型的复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 10000        # 训练轮数
MOVING_AVERAGE_DECAY = 0.99         # 滑动平均衰减率

def inference(input_tensor, avg_class, weights1, biases1,weights2, biases2):
    if avg_class == None:
        # 当没有提供滑动平均类时，直接使用参数当前的取值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


### 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    # 计算在当前参数下神经网络前向传播的效果。这里给出的用于计算滑动平均的类为None
    y = inference(x,None,weights1,biases1,weights2,biases2)
    
    # 定义存储训练轮数的变量，指定为不可训练的变量(trainable=False)
    global_step = tf.Variable(0, trainable=False)
    
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    
    # 在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量则不需要
    # 当传递trainable = True时，Variable（）构造函数会自动将新变量添加到GraphKeys.TRAINABLE_VARIABLES图形集合中。 
    # 这个函数可以返回该集合的内容。
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #计算使用了滑动平均之后的前向传播的结果。 
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络边上全中的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    
    # 总损失为交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization
    
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY,staircase=True)


    # 使用
    # 包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
    # 又需要更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 判断两个张量在每一维上是否相等
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    # 这个运算首先将布尔型的数值转换为实数型，然后计算平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 初始化会话并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据。一般在神经网络的训练过程中会通过数据来大致判断
        # 停止的条件和评判训练的效果
        validate_feed = {x : mnist.validation.images, y_ : mnist.validation.labels}
        test_feed = {x : mnist.test.images, y_ : mnist.test.labels}
        
        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证集上的测试结果
            if i % 1000 == 0:
                # 计算滑动平均模型在验证集上的结果，因为MNIST数据集比较小，
                # 所以一次可以处理所有的验证数据。
                validate_acc = sess.run(accuracy, feed_dict = validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d steps, validation accuracy is %g and test accuracy is %g " % (i, validate_acc,test_acc))
                
            # 产生这一轮使用的一个batch的训练数据，并巡行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_ : ys})
            
        # 在训练结束之后，在测试数据上检验正确率
        
        #print("After %d training steps, test accuracy using average mode is %g " % (TRAINING_STEPS, test_acc))
    

def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)

# TensorFlow提供一个主程序入口，tf.app.run()会调用上面的main函数
if __name__ == '__main__':
    main()











