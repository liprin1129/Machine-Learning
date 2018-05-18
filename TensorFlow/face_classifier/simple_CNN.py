import tensorflow as tf
from data_wrangling import PickleHelper
from tqdm import tqdm
import numpy as np

import platform
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
    
epoch = 5000
batch_size = 500
rl = 0.001 #learning rate

data = PickleHelper.load_pickle("../../Data/Face/", "faces-obj-32x32-features-norm.pkl")
labels = PickleHelper.load_pickle("../../Data/Face/", "faces-obj-32x32-labels-norm.pkl")
test_data = PickleHelper.load_pickle("../../Data/Face/", "blob-itamochi-32x32.pkl")

print("DATA: ", data.shape)

# FOR CODE TESTING
train_data = data[:100]
train_labels = labels[:100]
eval_data = data[100:120]
eval_labels = labels[100:120]

# CODE IMPLEMENTATION
train_data = data[:80000]
train_labels = labels[:80000]
#eval_data = data[80000:]
#eval_labels = labels[80000:]

feature_node = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
label_node = tf.placeholder(tf.float32, shape=[None, 2])
    
img_channel = 3
# CONVOLUTIONAL 1
conv1_W = tf.Variable(
    tf.truncated_normal([3, 3, 3, 32], stddev=0.1, dtype=tf.float32))
conv1_b = tf.Variable(tf.truncated_normal([32], stddev=0.1, dtype=tf.float32))

conv1 = tf.layers.conv2d(inputs=feature_node, filters=32, kernel_size=[3, 3], strides=(1, 1),  padding="valid", activation=tf.nn.relu)

#conv1 = tf.nn.conv2d(feature_node, conv1_W, strides=[1, 1, 1, 1], padding='VALID')
#relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
#pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)

# FLATTEN
relu1_shape = conv1.get_shape().as_list()
conv1_flat = tf.reshape(conv1, [-1, relu1_shape[1]*relu1_shape[2]*relu1_shape[3]])

# DENSE 1
dense1 = tf.layers.dense(inputs=conv1_flat, units=128, activation=tf.nn.relu)#tf.add(tf.matmul(

# DENSE 2
dense2 = tf.layers.dense(inputs=dense1, units=2)#tf.add(tf.matmul(

# CONST FUNCTION
loss = tf.losses.softmax_cross_entropy(onehot_labels=label_node, logits=dense2)

# OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=rl)
train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

# PREDICTION
prediction = tf.argmax(dense2, 1)

# ACCURACY
accuracy = tf.equal(prediction, tf.argmax(label_node, 1))

print("\n< ===================== TENSOR INFO. ===================== >")
print("conv1_W: ", conv1_W)
print("conv1_b: ", conv1_b)
print("relu1: ", conv1)#relu1.get_shape().as_list())
print("conv1_flat: ", conv1_flat)
print("dense 1: ", dense1)
print("dense 2: ", dense2)
print("loss: ", loss)
print("============================================================\n")

plt.ion()
plt.close('all')
fig_pred, ax_pred = plt.subplots(5, 5, figsize=(10, 10))

plt.show()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in tqdm(range(epoch)):
        batch_index = np.random.choice(len(train_data), size=batch_size)
        batch_x = train_data[batch_index]
        batch_y = train_labels[batch_index]

        sess.run(train_op, feed_dict={feature_node: batch_x, label_node: batch_y})

        if iter % 100 == 0:
            pred1 = sess.run(prediction, feed_dict={feature_node: test_data})
            pred2 = sess.run(prediction, feed_dict={feature_node: batch_x})
            #print("PRED: ", np.argmax(pred[:5], axis=1))
            #print("TRUE: ", np.argmax(train_labels[:5], axis=1))

            train_acc = sess.run(accuracy, feed_dict={feature_node: train_data[:batch_size], label_node: train_labels[:batch_size]})
            eval_acc = sess.run(accuracy, feed_dict={feature_node: eval_data[:batch_size], label_node: eval_labels[:batch_size]})
            print("({0}) TRAIN ACC.: {1} % | EVAL ACC.: {2} %"\
                  .format(iter, np.sum(train_acc)/len(train_acc)*100, np.sum(eval_acc)/len(eval_acc)*100))

            #print(pred[:10])
            
            for i in range(5):
                for j in range(5):
                    if (i == 0) and (j==0):
                        ax_pred[i, j].imshow(test_data[i*5+j])

                        if pred1[i*5+j] == 0:
                            ax_pred[i, j].set_title("Face")
                        else:
                            ax_pred[i, j].set_title("Non")
                    else:
                        ax_pred[i, j].imshow(batch_x[i*5+j])
                        #print("===> ", pred[i*5+j])
                        if pred2[i*5+j] == 0:
                            ax_pred[i, j].set_title("Face")
                        else:
                            ax_pred[i, j].set_title("Non")

            plt.tight_layout()
            plt.pause(0.0001)
            plt.draw()

'''
if __name__ == "__main__":
    cnn_fn(train_data, train_labels, None)
'''
