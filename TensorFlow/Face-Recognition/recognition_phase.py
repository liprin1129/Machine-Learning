sys.path.append("/home/shared-data/Personal_Dev/Machine-Learning/TensorFlow/Common_helper")

import tensorflow as tf
import sys
import tfrecorder_helper as tfr
from tqdm import tqdm
import facenet

def architectur():


if __name__=="__main__":
    image_helper = tfr.TFRecord_Helper(height=224, width=224, verbose=False)

    epoch_num = 10
    batch_size = 10
    iterator = image_helper.convert_from_tfrecord_with_tf_dataset('/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', batch_size, "train")
    #print(iterator)
    
    logits = facenet(iterator["image"], train_validation_phase=True)
    
    
    '''
    #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = iterator["class_idx"], logits = logits))
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = iterator["class_idx"], logits = logits))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

    predictions = tf.argmax(logits, 1, output_type = tf.int32)
    #accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, iterator["class_idx"]), tf.float32))
    #accuracy = tf.reduce_sum(tf.equal(predictions, iterator["class_idx"]))

    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries','facenet')):
        os.mkdir(os.path.join('summaries','face'))

    with tf.Session() as sess:#, tqdm(total = iterations) as pbar:
        summ_writer = tf.summary.FileWriter(os.path.join('summaries','facenet'), sess.graph)
        
        """
        sess.run(tf.global_variables_initializer())

        try:
            while True:
                image_data = sess.run(iterator)

                #print("Extracted image name: ", tfr.np.shape(image_data['image'])); sys.stdout.flush()
                #break

                print("==============> Training")
                _, acc = sess.run([optimizer, predictions])
                #pbar.update(batch_size)
                #print("Extracted image name:\t", image_data['class_idx']); sys.stdout.flush()
                #print("Predicted image name:\t", acc); sys.stdout.flush()

                print(sess.run(tf.reduce_sum(tf.cast(tf.equal(predictions, image_data['class_idx']), tf.float32))))

        except tf.errors.OutOfRangeError:
            #print("End of an {0}".format(os.path.basename(tfrecord_file)))
            print("ERROR")
            pass
        """
    '''