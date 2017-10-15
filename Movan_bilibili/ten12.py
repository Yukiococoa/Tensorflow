# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:09:44 2017

@author: 羽落苍穹
"""
# matplot的time类找不到 改用Spyder运行查看结果





import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time



def add_layer(inputs,in_size,out_size,activation_function=None):


    # 初始值随机比全部为零要好
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    # 初始值全部为0.1 同上
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape) # 噪音
y_data = np.square(x_data) - 0.5 + noise

# 有点儿像函数内的变量
    # 也可以用来选择部分数据训练模型
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

# 第一个隐含层
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(l1,10,1,activation_function=None)
# lose function
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
# 优化器参数 学习率，目标是minimize loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 结果可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
# 可以任意plot 不会终止整个函数
plt.show(block=False)
plt.ion()

for i in range(1000):
    # 训练
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))   
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        time.sleep(0.1)


