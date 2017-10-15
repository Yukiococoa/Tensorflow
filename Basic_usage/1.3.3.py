import tensorflow as tf

in1 = tf.constant(3)
in2 = tf.constant(2)
in3 = tf.constant(5)

add_ = tf.add(in2,in3)
mul_ = tf.multiply(in1,in2)

with tf.Session() as sess:
    result = sess.run([mul_,add_])
    print(result)
