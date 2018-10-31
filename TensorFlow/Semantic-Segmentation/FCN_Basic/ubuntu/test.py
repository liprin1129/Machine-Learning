import tensorflow as tf

a = tf.zeros([1, 300, 300, 1], tf.float32)

print(a)

b = tf.reshape(a, [-1])

print(b)