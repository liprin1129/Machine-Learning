import tensorflow as tf
from gradientzoo import TensorflowGradientzoo

with tf.Session() as sess:
#    #tf.saved_model.loader.load(sess, )
    #ckpt = tf.train.get_checkpoint_state('./')
    #print(ckpt)
    #saver = tf.train.Saver()
    #saver.restore(sess, "mnist_model.ckpt")
    ckpt = tf.train.get_checkpoint_state('./')
    print(ckpt)
    # Load latest weights from Gradientzoo
    #TensorflowGradientzoo('ericflo/mnist').load(sess)