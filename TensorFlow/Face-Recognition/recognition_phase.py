from facenet import FacenetModel
import os
import tensorflow as tf
import sys
import numpy as np
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_path', '/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', "root directory including images and tfrecords")

data_dir_root_str = FLAGS.image_path
data_image_dir_str = os.path.join(data_dir_root_str, 'images')

# Image folder list which is used to identify the number of net's output logits
data_image_list = os.listdir(data_image_dir_str)
data_image_list = [os.path.join(data_image_dir_str, x) for x in data_image_list]
data_image_list = [os.path.isdir(x) for x in data_image_list]

# Create a folder to save tensorboard data
if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','facenet')):
    os.mkdir(os.path.join('summaries','face'))

""" ################
# Hyper parameters #
################ """
epoch = 60
batch = 20
learning_rate = 0.001

num_outputs = sum(data_image_list)

model = FacenetModel(epoch, batch, learning_rate, num_outputs) # Initialize FacenetModel class to set shapes of layers

model.block1["unit1"] = [64, 64, 64]
model.block1["unit2"] = [64, 64, 64]
model.block1["unit3"] = [64, 64, 64]

model.block2["unit1"] = [128, 128, 256]
model.block2["unit2"] = [128, 128, 256]
model.block2["unit3"] = [128, 128, 256]
model.block2["unit4"] = [128, 128, 256]

model.block3["unit1"] = [256, 256, 512]
model.block3["unit2"] = [256, 256, 512]
model.block3["unit3"] = [256, 256, 512]
model.block3["unit4"] = [256, 256, 512]
model.block3["unit5"] = [256, 256, 512]
model.block3["unit6"] = [256, 256, 512]

model.block4["unit1"] = [512, 512, 2048]
model.block4["unit2"] = [512, 512, 2048]
model.block4["unit3"] = [512, 512, 2048]

def training():
    model(data_dir_root_str, 224, 224)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    #config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config_proto) as sess:

        tf.summary.FileWriter(os.path.join('summaries','facenet'), sess.graph) # Write tensorboard
        
        sess.run(tf.global_variables_initializer())

        training_handle = sess.run(model.train_iterator.string_handle())
        validation_handle = sess.run(model.validation_iterator.string_handle())
        for current_epoch_int in tqdm(range(epoch)):
            try:
                print("=== Training ==="); sys.stdout.flush()
                sess.run(model.train_iterator.initializer)
                total_accuracy = 0
                count = 0
                print("Processing: ", end=""); sys.stdout.flush()
                while True:
                    count += 1
                    print(".", end=""); sys.stdout.flush()

                    extracted_data = sess.run(model.get_next_in_interators, feed_dict={model.handle_placeholder: training_handle})
                    _ = sess.run(model.optimizer, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.labels_dataset_placeholder:extracted_data[1], model.train_valid_placeholder:True})

                    pred = sess.run(model.predictions, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.train_valid_placeholder:True})
                    total_accuracy += np.sum(np.equal(pred, extracted_data[1])) / batch

                    #print("{0} accuracy: {1}".format(np.sum(equality), equality)); sys.stdout.flush()
                    #print("expected:\t{0} \npredicted:\t{1}".format(extracted_data[1], pred)); sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                print("\n{0} %".format((total_accuracy/count)*100)); sys.stdout.flush()
                pass

            try:
                print("=== Validation ==="); sys.stdout.flush()
                sess.run(model.validation_iterator.initializer)
                total_accuracy = 0
                count = 0
                print("Processing: ", end=""); sys.stdout.flush()
                while True:
                    count += 1
                    print(".", end=""); sys.stdout.flush()

                    extracted_data = sess.run(model.get_next_in_interators, feed_dict={model.handle_placeholder: validation_handle})
                    pred = sess.run(model.predictions, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.train_valid_placeholder:True})
                    total_accuracy += np.sum(np.equal(pred, extracted_data[1])) / batch

                    #equality = np.equal(pred, extracted_data[1])
                    #print("{0} accuracy: {1}".format(np.sum(equality), equality)); sys.stdout.flush()
                    #print("expected:\t{0} \npredicted:\t{1}".format(extracted_data[1], pred)); sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                print("\n{0} %".format((total_accuracy/count)*100)); sys.stdout.flush()
                # Save the variables to disk.
                #save_path = saver.save(sess, '/tmp/%s_%03d-of-%03d.ckpt' % ("facenet", current_epoch_int, epoch))
                save_path = saver.save(sess, "/tmp/model.ckpt")
                print("Model saved in path: %s" % save_path)

                pass

            
    
def main(argv):
    training()

if __name__=="__main__":
    tf.app.run(main)
