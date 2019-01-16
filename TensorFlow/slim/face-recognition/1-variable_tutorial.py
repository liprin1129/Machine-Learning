import tensorflow as tf
import matplotlib.pyplot as plt
''' # 1. Native TF: initialize
bias_1 = tf.Variable(tf.zeros(shape=[200]), name="b1")
weight_1 = tf.Variable(tf.lin_space(start=0.0, stop=12.0, num=3),  name="w1")
weight_2 = tf.Variable(tf.range(start=0.0, limit=12.0, delta=3),  name="w2")
weight_3 = tf.Variable(tf.random_normal(shape=[784, 200], mean=1.5, stddev=0.35), name="w3")
weight_4 = tf.Variable(tf.truncated_normal(shape=[784, 200], mean=1.5, stddev=0.35), name="w4")
print(weight_1)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  val_b1 = sess.run(bias_1)
  val_w1, val_w2, val_w3, val_w4 = sess.run([weight_1, weight_2, weight_3, weight_4])
  print(type(val_b1))
  # >> <type 'numpy.ndarray'>
  print(val_w1.shape)
  # >> (3,)
  # 그래프로 변수 확인하기
  plt.subplot(221)
  plt.hist(val_w1)
  plt.title('val_w1_linspace')
  plt.grid(True)
  plt.subplot(222)
  plt.hist(val_w2)
  plt.title('val_w2_range')
  plt.grid(True)
  plt.subplot(223)
  plt.hist(val_w3)
  plt.title('val_w3_random_normal')
  plt.grid(True)
  plt.subplot(224)
  plt.hist(val_w4)
  plt.title('val_w2_truncated_normal')
  plt.grid(True)
  plt.show()
'''

''' # Native TF: assign device placement
with tf.device("/cpu:0"):
  bias_2= tf.Variable(tf.ones(shape=[200]), name="b2")
print(bias_1)
# >> Tensor("b1/read:0", shape=(200,), dtype=float32)
print(bias_2)
# >> Tensor("b2/read:0", shape=(200,), dtype=float32, device=/device:CPU:0)
'''

''' # Native TF: saving and restoring
model_path = "/tmp/tx-01.ckpt"
# 저장
bias_3 = tf.add(bias_1, bias_2, name='b3')
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(init_op)
  val_b3 = sess.run(bias_3)
  print(val_b3)
  save_path = saver.save(sess, model_path)
  print("Model saved in file: %s" % save_path)
# 로드
saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, model_path)
  print("Model restored")
  # access tensor by name directly
  val_b3 = sess.run('b3:0')
  print(val_b3)
  # get tensor by name
  graph = tf.get_default_graph()
  b3 = graph.get_tensor_by_name("b3:0")
  val_b3 = sess.run(b3)
  print(val_b3)
'''

import tensorflow.contrib.slim as slim
''' # 2. TF-slim

weight_4 = slim.variable('w4', shape=[784, 200], initializer=tf.truncated_normal_initializer(mean=1.5, stddev=0.35), device='/device:GPU:0')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)

  val_w4 = sess.run(weight_4)

  print(val_w4.shape)
  
  plt.hist(val_w4)
  plt.show()
'''

# 3. TF-slim: model variable and regular variable
# 모델 변수 생성하기
weight_5 = slim.model_variable('w5',
                                  shape=[10, 10, 3, 3],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  regularizer=slim.l2_regularizer(0.05),
                                  device='/CPU:0')
model_variables = slim.get_model_variables()
print([var.name for var in model_variables])
# >> [u'w5:0']
# 일반 변수 생성하기
my_var_1 = slim.variable('mv1',
                          shape=[20, 1],
                          initializer=tf.zeros_initializer(),
                          device='/device:GPU:0')

model_variables = slim.get_model_variables()
all_variables = slim.get_variables()
print([var.name for var in model_variables])
# >> [u'w5:0']
print([var.name for var in all_variables])
# >> [u'w4:0', u'w5:0', u'mv1:0'] 
  