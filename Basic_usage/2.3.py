import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# x = n*784
x = tf.placeholder("float",[None,784])
# W and b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# y
y = tf.nn.softmax(tf.matmul(x,W) + b)
# y_ true label
y_ = tf.placeholder("float",[None,10])

# loss function
Loss = -tf.reduce_sum(y_*tf.log(y))

# active train
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(Loss)

# reset
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))




