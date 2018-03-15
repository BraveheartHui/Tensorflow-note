# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 08:47:07 2018

chapter5_5.5
结合变量管理机制和模型持久化机制进行的完整的神经网络训练实践

mnist_train.py : 神经网络的训练部分

python: 3.6.2
tensorflow:1.4.0


@author: hxh
"""

import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播的函数
import mnist_inference as mi

# 配置神经网络中的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    
    # 清理图中保存的变量，重置图
    tf.reset_default_graph()
    
    # 定义输入输出
    
    x = tf.placeholder(tf.float32, [None,mi.INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32, [None,mi.OUTPUT_NODE],name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    # 直接使用inference中定义的前向传播过程
    y = mi.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    # 定义损失函数、学习率、滑动平均操作以及训练过程
    
    #print("train name:",tf.get_variable_scope().name)
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')   
    
    #初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs,y_:ys})
            
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出在当前batch上的损失
                print("After "+str(step)+" training steps, loss on training batch is "+str(loss_value)+". ")
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step = global_step)




def main():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)



if __name__ == '__main__' :
    main()




