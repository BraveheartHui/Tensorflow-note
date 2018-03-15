# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:07:17 2018

chapter5_5.5
结合变量管理机制和模型持久化机制进行的完整的神经网络训练实践

mnist_eval.py : 神经网络的测试部分

python: 3.6.2
tensorflow:1.4.0

@author: hxh
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载常量
import mnist_inference as mi
import mnist_train as mt


# 每10秒家在一次最新的模型，并在测试数据集上测试最新的模型正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出格式
        x =tf.placeholder(tf.float32, [None,mi.INPUT_NODE], name = 'x-input')
        y_ =tf.placeholder(tf.float32, [None, mi.OUTPUT_NODE], name = 'y-input')
        validate_feed = {x:mnist.validation.images, y_ : mnist.validation.labels}
        
        # 直接通过调用封装好的函数来加载模型
        y = mi.inference(x,None)
        
        correct_predition = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
        
        # 通过变量重命名的方式来家在模型
        variable_averages = tf.train.ExponentialMovingAverage(mt.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mt.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After " + str(global_step) +" training steps, validation accuracy = " + str(accuracy_score))
                else:
                    print("No checkpoint file found.")
                    return
            time.sleep(EVAL_INTERVAL_SECS)
        
        
def main():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()
        
        
        