# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:42:01 2018

tensorboard 监控指标可视化

python: 3.6.2
tensorflow:1.4.0

@author: hxh
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "tensorboard/log"
BATCH_SIZE = 100
TRAIN_STEPS = 3000

# 生成变量监控信息并定义生成监控信息日志的操作。
# var--给出了需要记录的张量
# name--在可视化结果中显示的图表名称，这个名称一般与变量名一致
def variable_summaries(var, name):
    # 将生成监控信息的操作放在同一命名空间下
    with tf.name_scope('summaries'):
        # 记录张量中元素的取值分布
        tf.summary.histogram(name, var)
        
        # 计算变量的平均值
        mean= tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev' + name, stddev)
    


# 生成一层全链接层神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 将同一层神经网络放在一个统一的命名空间下
    with tf.name_scope(layer_name):
        # 声明神经网络边上的权重，并调用生成权重监控信息日志的函数
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
        
        # 声明神经网络偏置项
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')
        
        # 记录神经网络输出节点在经过激活函数之前的分布
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name+'/preactivations', preactivate)
        activations = act(preactivate, name='activation')
        
        # 纪录神经网络输出节点在经过激活以后的分布
        tf.summary.histogram(layer_name+ '/activations', activations)
        return activations



def main():
    # 清理图中保存的变量，重置图
    tf.reset_default_graph()
    
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    #定义输出
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    
    hidden = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden, 500, 10, 'layer2', act=tf.identity)
    
    
    # 计算交叉熵并定义生成交叉熵监控日志
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('cross_entropy', cross_entropy)
    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    
    # 计算正确率
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    
    merge = tf.summary.merge_all()
    
    
    with tf.Session() as sess:
        # 初始化写日志的writer，并将当前图写入日志
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()
        
        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _ = sess.run([merge, train_step], feed_dict={x:xs, y_: ys})
            summary_writer.add_summary(summary, i)
    
    summary_writer.close()
    
 


if __name__ == '__main__':
    main()



